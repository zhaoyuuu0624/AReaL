# SPDX-License-Identifier: Apache-2.0

"""Unit tests and bounded torchrun entry points for packed-tree Ulysses."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from areal.models.tree_attn.cp_functional import TreePredictionPlan
from areal.models.tree_attn.tree import TrieNode
from areal.models.tree_attn.ulysses import TreeCPLayout


def test_tree_layout_slice_preserves_positions():
    """The second branch keeps its tree depth rather than a local arange."""
    layout = TreeCPLayout(8, 2, 1)
    actual = layout.slice(torch.tensor([[0, 1, 2, 3, 2, 3, 0, 0]]), -1)
    torch.testing.assert_close(actual, torch.tensor([[2, 3, 0, 0]]), rtol=0, atol=0)


@pytest.mark.parametrize("args", [(7, 2, 0), (8, 0, 0), (8, 2, 2), (0, 2, 0)])
def test_tree_layout_rejects_invalid_shapes(args):
    """Reject invalid layouts before entering any collective."""
    with pytest.raises(ValueError):
        TreeCPLayout(*args)


def test_tree_prediction_plan_keeps_branch_targets_and_occurrences():
    """Reuse computations but preserve both branches and terminal slots."""
    root = TrieNode(tree_id=0)
    root.nodes = [
        TrieNode(0, 0, 1, [1, 2], [0, 1]),
        TrieNode(0, 2, 3, [3, 4], [0]),
        TrieNode(0, 4, 5, [5, 6], [1]),
    ]
    plan = TreePredictionPlan.from_trie(root)
    assert (1, 2) in plan.edges and (1, 4) in plan.edges
    assert (3, 0) in plan.edges and (5, 0) in plan.edges
    assert len(plan.edges) == 7
    assert plan.sequences[0][1][0] == plan.sequences[1][1][0]
    assert [len(slots) for _, slots in plan.sequences] == [4, 4]


def test_tree_prediction_plan_empty_tree():
    """Transport-only microbatches must not invent supervised predictions."""
    plan = TreePredictionPlan.from_trie(TrieNode(tree_id=0))
    assert plan.edges == () and plan.sequences == ()


def test_tree_layout_rejects_mismatched_tensor():
    """Do not silently reshard an already-local tensor a second time."""
    with pytest.raises(ValueError, match="global padded length"):
        TreeCPLayout(8, 2, 0).slice(torch.zeros(4), 0)


@pytest.mark.parametrize(
    "overrides,exception",
    [
        ({"tensor_model_parallel_size": 2}, NotImplementedError),
        ({"pipeline_model_parallel_size": 2}, NotImplementedError),
        ({"num_query_groups": 1}, ValueError),
        ({"attention_dropout": 0.1}, NotImplementedError),
    ],
)
def test_tree_cp_rejects_unsupported_attention_before_collectives(overrides, exception):
    """Invalid combinations fail before looking up any process group."""
    from types import SimpleNamespace

    module = pytest.importorskip("areal.models.tree_attn.module_megatron")
    fields = dict(
        context_parallel_size=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        num_attention_heads=8,
        num_query_groups=4,
        attention_dropout=0.0,
    )
    fields.update(overrides)
    with pytest.raises(exception):
        module.PytorchFlexAttention(SimpleNamespace(**fields), 1, None, "self")


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.parametrize("backend", ["sdpa", "flex", "mcore", "engine"])
def test_tree_ulysses_two_gpu_forward_backward_update(backend):
    """Run real collectives; compare against independent unshared sequences."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Tree CP integration test requires two CUDA GPUs")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/torchrun/run_tree_cp.py",
            "--backend",
            backend,
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
