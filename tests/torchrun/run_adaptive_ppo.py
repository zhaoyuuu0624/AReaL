# SPDX-License-Identifier: Apache-2.0
"""Real four-rank GRPO/logprob/Adam oracle, including DCP cycle-boundary recovery.

Uses synthetic rollout trajectories and a small real MCore GPT. No inference
server or pretrained-model download is involved in this bounded integration test.
"""

import argparse
import copy
import faulthandler
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tests.torchrun.run_adaptive_tree_cp import (
    assert_close,
    check_optimizer,
    make_global_batch,
)

from areal.api.cli_args import AdaptiveTreeTrainingConfig, NormConfig, PPOActorConfig
from areal.tools.benchmark_tree_workload import build_engine
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.adaptive import AdaptivePPOBridge, stamp_trajectory_ids
from areal.utils.data import RolloutGroup, concat_batch


def build(precision, device, *, reference=False):
    args = SimpleNamespace(
        precision=precision,
        checkpoint=precision == "bf16",
        layers=2,
        hidden=128,
        cp=1,
        cap=512,
        model_seed=7,
        per_token_loss=True,
        distributed_optimizer=True,
        add_bias_linear=False,
        normalization="RMSNorm",
    )
    engine, model = build_engine(args, {"vocab_size": 64}, device)
    config = PPOActorConfig(
        backend="megatron:d4",
        dtype="bfloat16" if precision == "bf16" else "float32",
        enable_tree_training=True,
        pad_to_maximum=True,
        mb_spec=engine.config.mb_spec,
        gradient_checkpointing=precision == "bf16",
        ppo_n_minibatches=2,
        recompute_logprob=True,
        kl_ctl=0.0,
        reward_norm=NormConfig(
            mean_level="group",
            std_level="group",
            group_size=3,
            mean_leave1out=True,
            std_unbiased=True,
        ),
        adv_norm=NormConfig(mean_level="batch", std_level="batch", std_unbiased=True),
        adaptive_tree=AdaptiveTreeTrainingConfig(
            cp_sizes=[1] if reference else [1, 2, 4],
            local_token_budget=512 if reference else 128,
            max_tree_tokens=512,
            min_dwell_steps=0,
            replan_interval=8,
        ),
    )
    engine.config = config
    engine.actor = PPOActor(config, engine)
    engine.set_version(0)
    bridge = AdaptivePPOBridge(engine)
    return engine, model, bridge


def rollout(step):
    batch = make_global_batch(step)
    batch.pop("coefficients")
    # Workload fixture stores prediction-position masks; RL rollout masks mark
    # generated token positions, before PPO's one and only left shift.
    batch["loss_mask"] = torch.roll(batch["loss_mask"], 1, -1)
    rows = batch["input_ids"].shape[0]
    batch["logprobs"] = torch.full_like(batch["input_ids"], -4.0, dtype=torch.float32)
    batch["rewards"] = torch.linspace(-1, 1, rows)
    batch["is_truncated"] = torch.zeros(rows, dtype=torch.bool)
    batch["versions"] = torch.full_like(batch["input_ids"], step)
    groups, offset = [], 0
    for size in (3, 1, 2, 3, 2):
        item = {k: v[offset : offset + size].clone() for k, v in batch.items()}
        item["rollout_group"] = RolloutGroup((1,) * size)
        groups.append(item)
        offset += size
    assert offset == rows
    return groups


def run_cycle(engine, bridge, step, cp, source_data):
    rank = bridge.runtime.rank
    data = copy.deepcopy(source_data) if rank == 0 else None
    if rank == 0:
        stamp_trajectory_ids(data)
        bridge.stage(data, f"begin-{step}")
    bridge.begin(f"begin-{step}", step, cp_size=cp)
    assert bridge.cycle.forward_plan.cp_size == cp
    if rank == 0:
        bridge.stage(data, f"logp-{step}")
    logps = bridge.compute_logp(f"logp-{step}")
    adv = None
    if rank == 0:
        for item, logp in zip(data, logps, strict=True):
            item["prox_logp"] = logp
        bridge.stage(data, f"adv-{step}")
        adv = bridge.compute_advantages(f"adv-{step}")
        bridge.stage(adv, f"update-{step}")
    outcome = bridge.update(f"update-{step}")
    assert outcome["update_successful"]
    assert bridge.cycle is None and bridge.pending is None
    assert engine._active_tree_group is None and engine._active_tree_cap is None
    engine.set_version(step + 1)
    exported = engine.export_stats()
    if rank == 0:
        padded, _ = concat_batch(adv)
        assert exported["ppo_actor/n_seqs"] == padded["input_ids"].shape[0]
        count = int(padded["loss_mask"].count_nonzero())
        assert exported["ppo_actor/n_valid_tokens"] == count
        assert exported["ppo_actor/update/n_valid_tokens"] == count
        return logps, padded["advantages"], exported
    return None


def compare(actual, actual_model, reference, reference_model, bf16):
    for (name, pa), (other, pb) in zip(
        actual_model.named_parameters(), reference_model.named_parameters(), strict=True
    ):
        assert name == other
        assert_close(pa, pb, bf16, name, parameter=True)
    check_optimizer(actual, reference, bf16)


def main():
    faulthandler.dump_traceback_later(120, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    from megatron.core import parallel_state as mpu

    from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager

    mpu.initialize_model_parallel(context_parallel_size=1)
    actual, model, bridge = build(args.precision, device)
    reference, ref_model, ref_bridge = build(args.precision, device, reference=True)
    checkpointer = MegatronCheckpointManager(actual.model, actual.optimizer, None)
    rank, bf16 = dist.get_rank(), args.precision == "bf16"
    try:
        for step, cp in enumerate((1, 2, 4, 1)):
            source = rollout(step) if rank == 0 else None
            observed = run_cycle(actual, bridge, step, cp, source)
            expected = run_cycle(reference, ref_bridge, step, 1, source)
            if rank == 0:
                for a, b in zip(observed[0], expected[0], strict=True):
                    assert_close(a, b, bf16, "proximal logprob")
                assert_close(observed[1], expected[1], bf16, "advantages")
                for key in (
                    "ppo_actor/n_seqs",
                    "ppo_actor/n_groups",
                    "ppo_actor/n_valid_tokens",
                    "ppo_actor/update/n_valid_tokens",
                ):
                    assert observed[2][key] == expected[2][key], key
            compare(actual, model, reference, ref_model, bf16)
            assert bridge.runtime.planner.steps_since_switch == 1
            if step == 1:
                if rank == 0:
                    (args.checkpoint_dir / "model").mkdir(parents=True, exist_ok=True)
                    (args.checkpoint_dir / "adaptive.json").write_text(
                        json.dumps(bridge.state_dict())
                    )
                dist.barrier(group=actual.cpu_group)
                checkpointer.save_checkpoint(str(args.checkpoint_dir / "model"))
            if step == 2:
                # Restore model, optimizer and controller, then replay the next
                # complete cycle against the uninterrupted reference trajectory.
                checkpointer.load_checkpoint(str(args.checkpoint_dir / "model"))
                state = [
                    json.loads((args.checkpoint_dir / "adaptive.json").read_text())
                    if rank == 0
                    else None
                ]
                dist.broadcast_object_list(state, src=0, group=actual.cpu_group)
                bridge.load_state_dict(state[0])
                actual.set_version(2)
                assert bridge.last_cycle_id == 1
                replayed = run_cycle(actual, bridge, step, cp, source)
                if rank == 0:
                    for a, b in zip(replayed[0], observed[0], strict=True):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                compare(actual, model, reference, ref_model, bf16)
            if rank == 0:
                actual.logger.info(
                    "Adaptive PPO PASS precision=%s cycle=%s DPxCP=%sx%s (two Adam updates)",
                    args.precision,
                    step,
                    4 // cp,
                    cp,
                )
    except BaseException:
        # A failed rank must exit before collective teardown can hide its
        # original assertion behind peers waiting in a different collective.
        traceback.print_exc()
        os._exit(1)
    finally:
        faulthandler.cancel_dump_traceback_later()
        checkpointer.close()
        for item in (bridge, ref_bridge):
            item.close()
            item.runtime.close()
        dist.destroy_process_group(actual.cpu_group)
        dist.destroy_process_group(reference.cpu_group)
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
