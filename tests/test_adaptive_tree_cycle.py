# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for topology-independent PPO cycles and policy recovery."""

import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.engine.megatron_utils.adaptive_tree import AdaptiveTreeRuntime
from areal.models.tree_attn import adaptive
from areal.models.tree_attn.adaptive import AdaptiveTreeConfig, AdaptiveTreePlanner


def make_batch(length: int = 32, count: int = 8) -> dict[str, torch.Tensor]:
    """Use shared prefixes but distinct tails to exercise tree-aware partitions."""
    tokens = torch.full((count, length), 3, dtype=torch.long, device="cpu")
    tokens[:, 0] = torch.arange(count, device="cpu") // 2 + 1
    tokens[:, -1] = torch.arange(count, device="cpu") + 10
    return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}


def planner(**kwargs) -> AdaptiveTreePlanner:
    """Keep all candidate layouts feasible on small test batches."""
    return AdaptiveTreePlanner(
        4,
        AdaptiveTreeConfig(
            local_token_budget=128,
            max_tree_tokens=512,
            min_dwell_steps=0,
            **kwargs,
        ),
    )


def test_cycle_layout_preserves_global_optimizer_batch_ids():
    """CP changes partition a fixed optimizer batch, never the update schedule."""
    schedule = ((6, 0, 4, 2), (7, 1, 5, 3))
    chooser = planner()
    for cp in (1, 2, 4):
        cycle = chooser.plan_cycle(make_batch(), schedule, cycle_id=0, cp_size=cp)
        assert cycle.minibatch_indices == schedule
        assert all(plan.cp_size == cp for plan in cycle.optimizer_plans)
        for rows, plan in zip(schedule, cycle.optimizer_plans):
            actual = [rows[local] for shard in plan.shards for local in shard]
            assert sorted(actual) == sorted(rows)
        assert chooser.current_cp is None  # Planning cannot commit training state.


@pytest.mark.parametrize(
    "schedule",
    [(), ((), tuple(range(8))), ((0, 1, 2, 3), (4, 5, 6, 6)), ((0, 1),), ((True,),)],
)
def test_cycle_rejects_empty_missing_and_duplicate_optimizer_rows(schedule):
    """Malformed schedules fail before any candidate is executed."""
    with pytest.raises(ValueError, match="Optimizer minibatch"):
        planner().plan_cycle(make_batch(), schedule, cycle_id=0)


def test_cycle_small_optimizer_batch_forces_feasible_cp():
    """Full-batch feasibility alone cannot admit empty optimizer DP shards."""
    schedule = ((0,), tuple(range(1, 8)))
    chooser = planner()
    cycle = chooser.plan_cycle(make_batch(), schedule, cycle_id=0)
    assert cycle.cp_size == 4
    with pytest.raises(ValueError, match="infeasible"):
        chooser.plan_cycle(make_batch(), schedule, cycle_id=0, cp_size=1)


def test_cycle_skips_candidate_search_but_refreshes_current_batch(monkeypatch):
    """A stable workload packs only the selected CP and uses current row IDs."""
    chooser = planner(replan_interval=3, workload_change_threshold=0.25)
    schedule = (tuple(range(8)),)
    first = chooser.plan_cycle(make_batch(), schedule, cycle_id=0)
    chooser.commit_cycle(first)
    observed = []
    original = adaptive.make_plan

    def record(*args, **kwargs):
        observed.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(adaptive, "make_plan", record)
    batch = make_batch()
    batch["input_ids"] = batch["input_ids"].flip(0)
    for cycle_id in (1, 2):
        observed.clear()
        cycle = chooser.plan_cycle(batch, schedule, cycle_id=cycle_id)
        assert not cycle.replanned
        assert observed == [first.cp_size, first.cp_size]
        assert cycle.forward_plan.shards != first.forward_plan.shards
        chooser.commit_cycle(cycle)
    observed.clear()
    third = chooser.plan_cycle(batch, schedule, cycle_id=3)
    assert third.replanned
    assert set(observed) == {1, 2, 4}


def test_cycle_workload_growth_overrides_interval_and_infeasible_dwell():
    """Capacity pressure forces replanning even before the periodic check."""
    chooser = planner(replan_interval=100, workload_change_threshold=100)
    schedule = (tuple(range(8)),)
    first = chooser.plan_cycle(make_batch(), schedule, cycle_id=0, cp_size=1)
    chooser.commit_cycle(first)
    long = chooser.plan_cycle(make_batch(length=320), schedule, cycle_id=1)
    assert long.replanned and long.cp_size == 4


def test_cycle_tree_drift_triggers_search_without_length_change():
    """Removing shared prefixes changes tree work even at the same lengths."""
    chooser = planner(replan_interval=100, workload_change_threshold=0.1)
    batch = make_batch()
    batch["input_ids"][:, 0] = 1
    schedule = (tuple(range(8)),)
    first = chooser.plan_cycle(batch, schedule, cycle_id=0)
    chooser.commit_cycle(first)
    batch["input_ids"][:, 0] = torch.arange(8, device="cpu") + 1
    changed = chooser.plan_cycle(batch, schedule, cycle_id=1)
    assert changed.replanned


def test_cycle_commits_once_and_policy_roundtrips_through_json():
    """Checkpoint restore preserves cadence, drift reference, and cycle epochs."""
    chooser = planner()
    batch = make_batch()
    schedule = (tuple(range(8)),)
    first = chooser.plan_cycle(batch, schedule, cycle_id=12)
    chooser.commit_cycle(first)
    assert chooser.steps_since_switch == 1
    with pytest.raises(ValueError, match="already completed"):
        chooser.commit_cycle(first)
    with pytest.raises(ValueError, match="monotonically"):
        chooser.plan_cycle(batch, schedule, cycle_id=12)
    restored = planner()
    restored.load_state_dict(json.loads(json.dumps(chooser.state_dict())))
    assert restored.state_dict() == chooser.state_dict()
    assert restored.plan_cycle(batch, schedule, cycle_id=13) == chooser.plan_cycle(
        batch, schedule, cycle_id=13
    )


def test_forward_collection_deduplicates_cp_and_restores_input_order():
    """Different local widths are zero-padded into the global trajectory order."""
    batch = make_batch(count=4)
    cycle = planner().plan_cycle(batch, (tuple(range(4)),), 0, cp_size=2)
    plan = cycle.forward_plan
    gathered = [None] * 4
    for index, rows in enumerate(plan.shards):
        width = 20 + index
        gathered[index * 2] = torch.tensor(rows, dtype=torch.float32)[:, None].expand(
            -1, width
        )
        # Nonleaders are ignored even if a caller accidentally materializes them.
        gathered[index * 2 + 1] = torch.full((len(rows), width), 999.0)
    output = AdaptiveTreeRuntime._merge_forward_outputs(gathered, plan, batch)
    assert output.shape == batch["input_ids"].shape
    for index, rows in enumerate(plan.shards):
        for row in rows:
            torch.testing.assert_close(
                output[row, : 20 + index],
                torch.full((20 + index,), float(row)),
                rtol=0,
                atol=0,
            )
            assert not output[row, 20 + index :].count_nonzero()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replan_interval": 0},
        {"replan_interval": True},
        {"workload_change_threshold": -0.1},
        {"workload_change_threshold": float("nan")},
    ],
)
def test_cycle_config_rejects_invalid_replanning_controls(kwargs):
    """Reject policy errors before constructing distributed process groups."""
    with pytest.raises(ValueError, match="replan interval"):
        AdaptiveTreeConfig(**kwargs)


def _collective_cycle_validation_worker(rank: int, init_method: str) -> None:
    """Exercise actual CPU collectives without constructing a Megatron model."""
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = AdaptiveTreeRuntime.__new__(AdaptiveTreeRuntime)
        runtime.active = runtime.closed = runtime.failed = False
        runtime.rank, runtime.world_size, runtime.source = rank, 2, 0
        runtime.cpu_group = dist.group.WORLD
        runtime.pending_cycle = runtime.last_finished_cycle_id = None
        runtime.planner = AdaptiveTreePlanner(
            2, AdaptiveTreeConfig(cp_sizes=(1, 2), local_token_budget=128)
        )
        batch = make_batch() if rank == 0 else None
        schedule = (tuple(range(8)),) if rank == 0 else None
        with pytest.raises(ValueError, match="differ across ranks"):
            runtime.plan_cycle(batch, schedule, cycle_id=rank)
        with pytest.raises(ValueError, match="failed on source"):
            runtime.plan_cycle({} if rank == 0 else None, schedule, cycle_id=0)
        # Both errors were recoverable before any GPU or payload scatter began.
        cycle = runtime.plan_cycle(batch, schedule, cycle_id=0)
        assert cycle.cycle_id == 0
        with pytest.raises(ValueError, match="pending PPO cycle"):
            runtime.plan_cycle(batch, schedule, cycle_id=1)
        with pytest.raises(RuntimeError, match="pending"):
            runtime.state_dict()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_cycle_collective_source_and_epoch_errors_fail_on_all_workers(tmp_path):
    """A rejected plan cannot strand another worker in a later collective."""
    mp.spawn(
        _collective_cycle_validation_worker,
        args=((tmp_path / "cycle-group").as_uri(),),
        nprocs=2,
        join=True,
    )
