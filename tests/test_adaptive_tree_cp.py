# SPDX-License-Identifier: Apache-2.0
"""Planner tests and bounded real-GPU switching/Adam correctness tests."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from areal.models.tree_attn.adaptive import (
    AdaptiveTreeConfig,
    AdaptiveTreePlanner,
    prefix_partition,
)
from areal.tools.tree_workload import make_batch, unique_work


def batch_of_length(length):
    return make_batch(
        [
            dict(input_ids=[i + 1] + [10] * (length - 2) + [20 + j], response_start=1)
            for i in range(4)
            for j in range(2)
        ]
    )


def test_prefix_partition_preserves_rows_and_matches_contiguous_optimum():
    """Check a small exhaustive oracle, including duplicate/prefix-only paths."""
    seqs = [[1, 2, 3], [1, 2, 4], [1, 2], [1, 2, 3], [8, 1], [9, 2, 3, 4]]
    order = sorted(range(len(seqs)), key=lambda i: seqs[i])

    def work(ids):
        return unique_work([dict(input_ids=seqs[i]) for i in ids])[0]

    for count in range(1, len(seqs) + 1):
        actual = prefix_partition(seqs, count)
        assert sorted(i for shard in actual for i in shard) == list(range(len(seqs)))
        optimum = min(
            max(work(order[a:b]) for a, b in zip((0, *cuts), (*cuts, len(seqs))))
            for cuts in itertools.combinations(range(1, len(seqs)), count - 1)
        )
        assert max(map(work, actual)) == optimum


def test_adaptive_growth_and_shrink_changes_cp_without_monotonic_assumption():
    """Capacity changes force CP growth; a later short batch returns to DP."""
    planner = AdaptiveTreePlanner(
        4,
        AdaptiveTreeConfig(
            local_token_budget=128, max_tree_tokens=512, min_dwell_steps=0
        ),
    )
    decisions = []
    for length in (32, 180, 320, 32):
        plan = planner.plan(batch_of_length(length))
        decisions.append(plan.cp_size)
        planner.commit(plan)
    assert decisions == [1, 2, 4, 1]


def test_dwell_does_not_keep_infeasible_cp_and_state_roundtrips():
    """Safety overrides hysteresis, while feasible shrink respects dwell."""
    config = AdaptiveTreeConfig(
        local_token_budget=128, max_tree_tokens=512, min_dwell_steps=3
    )
    planner = AdaptiveTreePlanner(4, config)
    planner.commit(planner.plan(batch_of_length(32)))
    long = planner.plan(batch_of_length(320))
    assert long.cp_size == 4
    planner.commit(long)
    assert planner.plan(batch_of_length(32)).cp_size == 4
    restored = AdaptiveTreePlanner(4, config)
    restored.load_state_dict(planner.state_dict())
    assert restored.plan(batch_of_length(32)) == planner.plan(batch_of_length(32))
    for _ in range(2):
        restored.commit(restored.plan(batch_of_length(32)))
    assert restored.plan(batch_of_length(32)).cp_size == 1


def test_attention_accounting_and_padding_use_actual_packer():
    """Shared nodes count once, and semantic pairs are separate from padding."""
    batch = make_batch(
        [
            dict(input_ids=[1, 2, 3], response_start=1),
            dict(input_ids=[1, 2, 4], response_start=1),
        ]
    )
    plan = AdaptiveTreePlanner(
        1, AdaptiveTreeConfig(cp_sizes=(1,), local_token_budget=128)
    ).plan(batch)
    assert plan.packed_tokens == 4
    assert plan.allowed_attention_pairs == 9
    assert plan.token_cap == 128 and plan.microbatches == 1


def test_infeasible_and_explicit_override_are_rejected():
    """Do not truncate long trajectories or silently select another forced CP."""
    planner = AdaptiveTreePlanner(
        4, AdaptiveTreeConfig(local_token_budget=128, max_tree_tokens=512)
    )
    with pytest.raises(ValueError, match="No feasible"):
        planner.plan(batch_of_length(600))
    with pytest.raises(ValueError, match="infeasible"):
        planner.plan(batch_of_length(320), cp_size=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cp_sizes=(2, 1)),
        dict(cp_sizes=(3,)),
        dict(local_token_budget=10),
        dict(min_relative_gain=1),
        dict(cp_cost_multipliers=(float("nan"),)),
    ],
)
def test_invalid_config_fails_before_collectives(kwargs):
    """Reject unsupported candidates and invalid cost parameters early."""
    with pytest.raises(ValueError):
        AdaptiveTreeConfig(**kwargs)


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "precision,checkpoint,distributed_optimizer",
    [
        ("fp32", False, False),
        ("bf16", False, False),
        ("bf16", True, False),
        ("fp32", False, True),
        ("bf16", True, True),
    ],
)
def test_adaptive_tree_four_gpu_adam(precision, checkpoint, distributed_optimizer):
    """Real MCore Adam survives CP1->2->4->1 under both loss normalization modes."""
    if torch.cuda.device_count() < 4:
        pytest.skip("Adaptive tree integration requires four CUDA GPUs")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="1", AREAL_USE_TRITON_TREE_ATTN="0")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(root), env.get("PYTHONPATH")])
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "tests/torchrun/run_adaptive_tree_cp.py",
        "--precision",
        precision,
    ]
    if checkpoint:
        command.append("--checkpoint")
    if distributed_optimizer:
        command.append("--distributed-optimizer")
    result = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True, timeout=1200
    )
    assert result.returncode == 0, result.stdout + result.stderr
