# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in PPO/controller integration."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import AdaptiveTreeTrainingConfig, NormConfig, PPOActorConfig
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.trainer.ppo.actor import PPOActor, PPOActorController
from areal.trainer.ppo.adaptive import (
    ROW_ID,
    AdaptivePPOBridge,
    global_minibatch_indices,
    stamp_trajectory_ids,
    validate_adaptive_actor_config,
)
from areal.utils import stats_tracker
from areal.utils.data import RolloutGroup, concat_batch


def config(**kwargs):
    return PPOActorConfig(
        backend="megatron:d4",
        enable_tree_training=True,
        adaptive_tree=AdaptiveTreeTrainingConfig(
            local_token_budget=128, max_tree_tokens=512
        ),
        **kwargs,
    )


def data():
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]], dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.bool),
            "rewards": torch.tensor([0.0, 1.0]),
            "logprobs": torch.zeros(2, 3),
            "is_truncated": torch.zeros(2, dtype=torch.bool),
            "rollout_group": RolloutGroup((1, 1)),
        }
    ]


def test_yaml_config_is_opt_in_and_roundtrips():
    assert PPOActorConfig().adaptive_tree is None
    typed = OmegaConf.merge(
        OmegaConf.structured(PPOActorConfig),
        {
            "experiment_name": "test",
            "trial_name": "test",
            "backend": "megatron:d4",
            "adaptive_tree": {"cp_sizes": [1, 2], "replan_interval": 5},
        },
    )
    restored = OmegaConf.to_object(typed)
    runtime = restored.adaptive_tree.to_runtime_config()
    assert runtime.cp_sizes == (1, 2) and runtime.replan_interval == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m2_threshold": 0.5},
        {"offload": True},
        {"weight_update_mode": "awex"},
        {"ppo_n_minibatches": 0},
    ],
)
def test_unsupported_actor_semantics_fail_before_collectives(kwargs):
    with pytest.raises(ValueError):
        validate_adaptive_actor_config(config(**kwargs))


def test_global_minibatches_preserve_rows_and_step_count():
    rows = global_minibatch_indices(11, 3)
    assert tuple(map(len, rows)) == (4, 4, 3)
    assert sum(rows, ()) == tuple(range(11))
    with pytest.raises(ValueError):
        global_minibatch_indices(2, 3)


def test_row_ids_use_rtensor_shapes_without_localizing():
    item = {
        "attention_mask": RTensor(
            shard=TensorShardInfo(shard_id="mask", node_addr="test.invalid"),
            data=torch.empty((3, 9), dtype=torch.bool, device="meta"),
        )
    }
    stamp_trajectory_ids([item])
    torch.testing.assert_close(item[ROW_ID], torch.arange(3), rtol=0, atol=0)
    with pytest.raises(ValueError, match="reserved"):
        stamp_trajectory_ids([item])


def test_staging_preserves_groups_rejects_reorder_and_consumes_once():
    bridge = object.__new__(AdaptivePPOBridge)
    bridge.runtime = SimpleNamespace(rank=0, failed=False)
    bridge.pending, bridge.cycle, bridge.closed = None, None, False
    batch = data()
    stamp_trajectory_ids(batch)
    assert bridge.stage(batch, "first")["rows"] == 2
    with pytest.raises(RuntimeError, match="outstanding"):
        bridge.stage(batch, "second")
    tensors, meta = bridge._take("first")
    assert meta.rollout_groups == [RolloutGroup((1, 1))]
    assert tensors["input_ids"].device.type == "cpu"
    with pytest.raises(ValueError, match="consumed"):
        bridge._take("first")
    batch[0][ROW_ID] = batch[0][ROW_ID].flip(0)
    with pytest.raises(ValueError, match="reordered"):
        bridge.stage(batch, "third")


def test_normalization_group_reaches_reward_and_advantage_paths(monkeypatch):
    actor = PPOActor(
        PPOActorConfig(
            recompute_logprob=False, reward_norm=NormConfig(), adv_norm=NormConfig()
        ),
        SimpleNamespace(),
    )
    sentinel = object()
    observed = []
    original = actor.reward_norm.affine_parameters

    def record(
        x,
        mask=None,
        high_precision=True,
        reduce_group=None,
        group_sizes=None,
        group_member_counts=None,
    ):
        observed.append(reduce_group)
        return original(x, mask, high_precision, None, group_sizes, group_member_counts)

    monkeypatch.setattr(actor.reward_norm, "affine_parameters", record)
    original_adv = actor.adv_norm.__class__.__call__

    class Capture:
        def __call__(self, *args, **kwargs):
            observed.append(kwargs.pop("reduce_group", None))
            return original_adv(real_adv, *args, **kwargs)

    real_adv = actor.adv_norm
    actor.adv_norm = Capture()
    tensors, meta = concat_batch(data())
    actor._compute_advantages(tensors, meta, normalization_group=sentinel)
    assert observed == [sentinel, sentinel]
    stats_tracker.export_all()


def test_nonleader_loss_has_same_gradient_without_duplicate_statistics():
    actor = PPOActor(PPOActorConfig(recompute_logprob=False), SimpleNamespace())
    inputs = {
        "logprobs": torch.zeros(2, 3),
        "prox_logp": torch.zeros(2, 3),
        "advantages": torch.ones(2, 3),
        "loss_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    values = torch.full((2, 3), -0.1, requires_grad=True)
    stats_tracker.export_all()
    silent = actor._make_loss_fn(0, record_stats=False)(
        values, torch.zeros_like(values), inputs
    )
    (grad_silent,) = torch.autograd.grad(silent, values)
    assert stats_tracker.export_all() == {}
    recorded = actor._make_loss_fn(0)(values, torch.zeros_like(values), inputs)
    (grad_recorded,) = torch.autograd.grad(recorded, values)
    torch.testing.assert_close(grad_silent, grad_recorded, rtol=0, atol=0)
    assert stats_tracker.export_all()["n_valid_tokens"] == 6


@pytest.mark.parametrize("fail_stage", [False, True])
def test_controller_waits_for_source_ack_before_collective_dispatch(fail_stage):
    controller = object.__new__(PPOActorController)
    controller.workers = [SimpleNamespace(id=i) for i in range(4)]
    controller.workers_is_dp_head = [True] * 4
    controller._engine_name = lambda i: f"actor/{i}"
    events = []

    async def call(worker, method, engine, *args, rpc_meta, **kwargs):
        assert rpc_meta == {"broadcast": False}
        if method == "stage_adaptive_batch":
            assert worker == 0
            await asyncio.sleep(0)
            if fail_stage:
                raise RuntimeError("fetch failed")
            events.append("prepared")
            return {"handle": args[1]}
        if method == "adaptive_compute_logp":
            assert events[0] == "prepared"
            assert isinstance(args[0], str)
            events.append(worker)
            return "source-output" if worker == 0 else None
        assert method == "release_adaptive_batch" and worker == 0
        events.append("released")

    controller.scheduler = SimpleNamespace(async_call_engine=call)
    if fail_stage:
        with pytest.raises(RuntimeError, match="fetch failed"):
            controller._adaptive_call("adaptive_compute_logp", data())
        assert events == []
    else:
        assert (
            controller._adaptive_call("adaptive_compute_logp", data())
            == "source-output"
        )
        assert events == ["prepared", 0, 1, 2, 3, "released"]


def test_recovery_extra_state_roundtrips_and_old_manifests_load(tmp_path):
    from areal.api import StepInfo
    from areal.utils.recover import RecoverInfo

    state = {"adaptive_tree": {"last_cycle_id": 3, "cp_sizes": [1, 2, 4]}}
    info = RecoverInfo(StepInfo(0, 0, 0, 1), {}, {}, {}, {}, {}, extra_state=state)
    info.dump(str(tmp_path))
    assert RecoverInfo.load(str(tmp_path)).extra_state == state
    (tmp_path / "extra_state.json").unlink()
    assert RecoverInfo.load(str(tmp_path)).extra_state == {}


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_adaptive_ppo_four_gpu_oracle(precision, tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires four CUDA GPUs and Megatron")
    pytest.importorskip("megatron.core")
    env = dict(
        os.environ,
        AREAL_USE_TRITON_TREE_ATTN="0",
        AREAL_FLEX_ATTENTION_BLOCK_SIZE="128",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(Path(__file__).parent / "torchrun" / "run_adaptive_ppo.py"),
            "--precision",
            precision,
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
        ],
        env=env,
        check=True,
        timeout=900,
    )
