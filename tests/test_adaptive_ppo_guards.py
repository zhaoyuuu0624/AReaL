# SPDX-License-Identifier: Apache-2.0
"""CPU staging identities and fail-fast scope guards for adaptive PPO."""

import copy
from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import (
    AdaptiveTreeTrainingConfig,
    PPOActorConfig,
    SchedulingStrategy,
)
from areal.trainer.ppo.adaptive import (
    ROW_ID,
    AdaptivePPOBridge,
    _group_signature,
    _identity,
    stamp_trajectory_ids,
)
from areal.utils.data import RolloutGroup, concat_batch


def _trajectories() -> list[dict]:
    """Two prompts, unequal group sizes, and removable right padding."""
    data = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0], [1, 2, 4, 0, 0]]),
            "attention_mask": torch.tensor([[True, True, True, False, False]] * 2),
            "rewards": torch.tensor([0.0, 1.0]),
            "logprobs": torch.zeros(2, 5),
            "rollout_group": RolloutGroup((1, 1)),
        },
        {
            "input_ids": torch.tensor([[7, 8, 0, 0]]),
            "attention_mask": torch.tensor([[True, True, False, False]]),
            "rewards": torch.tensor([2.0]),
            "logprobs": torch.zeros(1, 4),
            "rollout_group": RolloutGroup((1,)),
        },
    ]
    stamp_trajectory_ids(data)
    return data


def _bridge(*, rank: int = 0, requires_logp: bool = False) -> AdaptivePPOBridge:
    """Construct the CPU handoff boundary without starting distributed engines."""
    bridge = object.__new__(AdaptivePPOBridge)
    bridge.runtime = SimpleNamespace(rank=rank, failed=False)
    bridge.engine = SimpleNamespace(
        config=SimpleNamespace(should_compute_prox_logp=lambda: requires_logp)
    )
    bridge.actor = SimpleNamespace()
    bridge.pending = bridge.cycle = bridge.identity = bridge.group_signature = None
    bridge.normalization_group = object()
    bridge.logp_done = bridge.advantages_done = bridge.closed = False
    bridge.last_cycle_id = -1
    return bridge


def _bind_cycle(bridge: AdaptivePPOBridge, data: list[dict]) -> None:
    """Bind CPU identity metadata at the same boundary as a prepared cycle."""
    tensors, meta = concat_batch(data)
    bridge.cycle = SimpleNamespace(cycle_id=0)
    bridge.identity = _identity(tensors)
    bridge.group_signature = _group_signature(meta)


def test_source_staging_copies_and_detaches_the_snapshot():
    """Mutating localized input after its ACK cannot change a pending execution."""
    bridge = _bridge()
    data = _trajectories()
    data[0]["logprobs"].requires_grad_(True)
    expected, expected_meta = concat_batch(data)
    expected = {key: value.detach().clone() for key, value in expected.items()}
    bridge.stage(data, "snapshot")

    with torch.no_grad():
        for item in data:
            for value in item.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
    data.clear()
    actual, meta = bridge._take("snapshot")

    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        assert actual[key].device.type == "cpu"
        assert not actual[key].requires_grad
    assert meta == expected_meta


def test_staging_rejects_non_source_without_consuming_data():
    """A non-source worker cannot acquire ownership of a unique rollout batch."""
    bridge = _bridge(rank=1)
    with pytest.raises(RuntimeError, match="Only source"):
        bridge.stage(_trajectories(), "wrong-owner")
    assert bridge.pending is None


def test_identity_allows_transport_padding_trim_and_ignores_padding_values():
    """RTensor trimming preserves identity despite a changed padded width."""
    original = _trajectories()
    bridge = _bridge()
    _bind_cycle(bridge, original)
    trimmed = copy.deepcopy(original)
    for item, width in zip(trimmed, (3, 2), strict=True):
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                item[key] = value[:, :width]
    bridge.stage(trimmed, "trimmed")
    tensors, _ = bridge._take("trimmed")
    assert tensors["input_ids"].shape == (3, 3)
    assert _identity(tensors) == bridge.identity

    padded = copy.deepcopy(original)
    for item in padded:
        item["input_ids"][~item["attention_mask"]] = 999
    bridge.stage(padded, "padding-values")
    tensors, _ = bridge._take("padding-values")
    assert _identity(tensors) == bridge.identity


@pytest.mark.parametrize("change", ["token", "row_order", "valid_length"])
def test_staging_rejects_semantic_batch_changes_during_cycle(change):
    """Stable IDs cannot authorize different tokens, ordering, or valid lengths."""
    original = _trajectories()
    bridge = _bridge()
    _bind_cycle(bridge, original)
    changed = copy.deepcopy(original)
    if change == "token":
        changed[0]["input_ids"][0, 1] = 99
    elif change == "row_order":
        changed[0]["input_ids"] = changed[0]["input_ids"].flip(0)
    else:
        changed[0]["attention_mask"][0, 2] = False
    with pytest.raises(ValueError, match="batch identity changed"):
        bridge.stage(changed, "changed")
    assert bridge.pending is None


def test_staging_rejects_reordered_row_ids_before_binding_a_cycle():
    """The initial handoff must also retain global row identity and order."""
    bridge = _bridge()
    data = _trajectories()
    data[0][ROW_ID] = data[0][ROW_ID].flip(0)
    with pytest.raises(ValueError, match="reordered or duplicated"):
        bridge.stage(data, "reordered")
    assert bridge.pending is None


@pytest.mark.parametrize("change", ["prompt_groups", "logical_rollouts", "references"])
def test_staging_rejects_changed_group_metadata_with_identical_tokens(change):
    """Regrouping identical rows must not silently change reward normalization."""
    data = _trajectories()
    bridge = _bridge()
    _bind_cycle(bridge, data)
    changed = copy.deepcopy(data)
    if change == "prompt_groups":
        first = changed.pop(0)
        split = [
            {
                key: value[row : row + 1]
                for key, value in first.items()
                if isinstance(value, torch.Tensor)
            }
            for row in range(2)
        ]
        for item in split:
            item["rollout_group"] = RolloutGroup((1,))
        changed = split + changed
    elif change == "logical_rollouts":
        changed[0]["rollout_group"] = RolloutGroup((2,))
    else:
        changed[0]["rollout_group"] = RolloutGroup((1, 1), (0.0, 1.0))
    batch, _ = concat_batch(changed)
    assert _identity(batch) == bridge.identity
    with pytest.raises(ValueError, match="rollout group metadata changed"):
        bridge.stage(changed, "regrouped")
    assert bridge.pending is None


def test_advantages_may_execute_only_once_per_cycle():
    """A retry after successful preprocessing cannot normalize advantages twice."""
    data = _trajectories()
    bridge = _bridge()
    _bind_cycle(bridge, data)
    calls = []

    def compute(batch, meta, *, normalization_group):
        calls.append(normalization_group)
        return {**batch, "advantages": torch.ones_like(batch["logprobs"])}

    bridge.actor._compute_advantages = compute
    bridge.stage(data, "advantages-first")
    result = bridge.compute_advantages("advantages-first")
    assert calls == [bridge.normalization_group]
    assert [item["advantages"].shape for item in result] == [(2, 5), (1, 4)]
    bridge.stage(data, "advantages-again")
    with pytest.raises(RuntimeError, match="already computed"):
        bridge.compute_advantages("advantages-again")
    assert calls == [bridge.normalization_group]
    bridge.release("advantages-again")


def test_advantages_require_configured_proximal_logprob_before_consuming_handle():
    """Reject a missing required forward phase while preserving prepared input."""
    data = _trajectories()
    bridge = _bridge(requires_logp=True)
    _bind_cycle(bridge, data)
    bridge.stage(data, "advantages")
    with pytest.raises(RuntimeError, match="proximal logprob first"):
        bridge.compute_advantages("advantages")
    assert bridge.pending[0] == "advantages"


@pytest.fixture
def trainer(monkeypatch):
    """Run real configuration validation without building engines or schedulers."""
    from areal.trainer import rl_trainer

    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    instance = object.__new__(rl_trainer.PPOTrainer)
    instance.config = SimpleNamespace(
        actor=PPOActorConfig(
            backend="megatron:d4",
            enable_tree_training=True,
            adaptive_tree=AdaptiveTreeTrainingConfig(
                local_token_budget=128, max_tree_tokens=512
            ),
        ),
        rollout=SimpleNamespace(scheduling_strategy=SchedulingStrategy()),
        critic=None,
        teacher=None,
        mopd=None,
        ref=None,
    )
    return instance


def test_trainer_accepts_supported_actor_with_static_reference_capacity(trainer):
    """A separate static reference may cover the adaptive actor's maximum cap."""
    trainer.config.ref = SimpleNamespace(
        adaptive_tree=None, mb_spec=SimpleNamespace(max_tokens_per_mb=512)
    )
    trainer._validate_adaptive_tree_config()


@pytest.mark.parametrize("role", ["critic", "teacher", "mopd"])
def test_trainer_rejects_unsupported_adaptive_roles(trainer, role):
    """Unsupported roles fail before any distributed engine is initialized."""
    setattr(trainer.config, role, SimpleNamespace())
    with pytest.raises(ValueError, match="critic-free GRPO without teachers/MOPD"):
        trainer._validate_adaptive_tree_config()


def test_trainer_rejects_spmd_execution(trainer, monkeypatch):
    """The adaptive bridge requires its explicit single-controller RPC route."""
    from areal.trainer import rl_trainer

    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: False)
    with pytest.raises(ValueError, match="single-controller"):
        trainer._validate_adaptive_tree_config()


@pytest.mark.parametrize(
    "backend",
    [
        "megatron:d4t2",
        "megatron:d4p2",
        "megatron:d2c2",
        "megatron:(attn:d4|ffn:d2e2)",
    ],
)
def test_trainer_rejects_nontrivial_base_model_parallelism(trainer, backend):
    """Only the runtime's logical CP may vary; base MCore topology stays fixed."""
    trainer.config.actor.backend = backend
    with pytest.raises(ValueError, match="base TP=PP=EP=CP=1"):
        trainer._validate_adaptive_tree_config()


@pytest.mark.parametrize("cap", [None, 128, 511])
def test_trainer_rejects_reference_capacity_below_actor_cap(trainer, cap):
    """Long trajectories must not fail in the unchanged reference-logprob path."""
    trainer.config.ref = SimpleNamespace(
        adaptive_tree=None, mb_spec=SimpleNamespace(max_tokens_per_mb=cap)
    )
    with pytest.raises(ValueError, match="Static reference.*must cover"):
        trainer._validate_adaptive_tree_config()


def test_trainer_rejects_adaptive_reference(trainer):
    """The first integrated route adapts the actor only."""
    trainer.config.ref = SimpleNamespace(
        adaptive_tree=trainer.config.actor.adaptive_tree,
        mb_spec=SimpleNamespace(max_tokens_per_mb=512),
    )
    with pytest.raises(ValueError, match="reference engine must use a static"):
        trainer._validate_adaptive_tree_config()


@pytest.mark.parametrize("owner,target", [("actor", "rollout"), ("rollout", "actor")])
def test_trainer_rejects_colocation_in_either_direction(trainer, owner, target):
    """Both scheduling descriptions of shared residency are rejected."""
    getattr(trainer.config, owner).scheduling_strategy = SchedulingStrategy(
        type="colocation", target=target
    )
    with pytest.raises(ValueError, match="separate actor and rollout placement"):
        trainer._validate_adaptive_tree_config()


def test_disabled_adaptive_validation_leaves_existing_modes_untouched(monkeypatch):
    """Opting out bypasses every new adaptive-only backend and topology guard."""
    from areal.trainer import rl_trainer

    def unexpected():
        pytest.fail("Static execution should not inspect adaptive controller mode")

    monkeypatch.setattr(rl_trainer, "is_single_controller", unexpected)
    trainer = object.__new__(rl_trainer.PPOTrainer)
    trainer.config = SimpleNamespace(actor=SimpleNamespace(adaptive_tree=None))
    trainer._validate_adaptive_tree_config()
