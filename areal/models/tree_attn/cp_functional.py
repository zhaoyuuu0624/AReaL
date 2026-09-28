# SPDX-License-Identifier: Apache-2.0

"""Tree prediction ownership and scalar-only reconstruction across CP ranks."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from areal.models.tree_attn.ulysses import TreeCPLayout
from areal.utils.functional.vocab_parallel import gather_logprobs_entropy

if TYPE_CHECKING:
    from areal.models.tree_attn.tree import TrieNode


@dataclass(frozen=True)
class TreePredictionPlan:
    # Distinct (prediction position, target position) computations. Occurrence
    # indices below preserve duplicates and per-trajectory loss/advantage weights.
    edges: tuple[tuple[int, int], ...]
    sequences: tuple[tuple[int, tuple[int, ...]], ...]

    @classmethod
    def from_trie(cls, trie: "TrieNode") -> "TreePredictionPlan":
        edges: list[tuple[int, int]] = []
        edge_ids: dict[tuple[int, int], int] = {}
        sequences = []
        for sid in trie.all_sequence_ids:
            positions = [
                p
                for start, end in trie.get_sequence_tree_indices(sid)
                for p in range(start, end + 1)
            ]
            slots = []
            # Match functional.py's existing terminal slot (leaf -> token 0).
            # It is excluded by the original trajectory's terminal loss mask.
            for pred, target in zip(positions, positions[1:] + [0]):
                edge = (pred, target)
                if edge not in edge_ids:
                    edge_ids[edge] = len(edges)
                    edges.append(edge)
                slots.append(edge_ids[edge])
            sequences.append((sid, tuple(slots)))
        return cls(tuple(edges), tuple(sequences))


def gather_tree_cp_scalars(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    plan: TreePredictionPlan,
    cp_group: dist.ProcessGroup,
    *,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[
    dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor, torch.Tensor
]:
    """Reconstruct logprob/entropy/min/max, never full-vocabulary logits.

    Every CP rank gets the same scalar outputs and evaluates the full local-DP
    loss. The differentiable SUM also sums replica cotangents in backward. This
    matches MegatronEngine's existing CP normalization contract; do NOT divide by
    CP here. Ranks owning no predictions still join forward/backward collectives.
    """
    if tp_group is not None and dist.get_world_size(tp_group) != 1:
        raise NotImplementedError(
            "Tree CP scalar reconstruction currently requires TP=1"
        )
    input_ids = input_ids.reshape(-1)
    layout = TreeCPLayout(
        input_ids.numel(), dist.get_world_size(cp_group), dist.get_rank(cp_group)
    )
    if logits.ndim != 2 or logits.shape[0] != layout.local_tokens:
        raise ValueError("Expected CP-local padded-tree logits [N/CP, vocab/TP]")
    owned = [
        (slot, pred - layout.start, target)
        for slot, (pred, target) in enumerate(plan.edges)
        if layout.start <= pred < layout.start + layout.local_tokens
    ]
    device = logits.device
    # A finite, connected zero ensures an empty owner executes attention BWD.
    zero = logits.reshape(-1)[:1].float().sum() * 0.0
    scalars = (
        torch.zeros((2, len(plan.edges)), dtype=torch.float32, device=device) + zero
    )
    stats = torch.zeros((2, len(plan.edges)), dtype=torch.float32, device=device)
    # Chunk *before* indexing vocabulary logits, so branching does not create
    # an unbounded [number_of_edges, vocab] duplicate tensor.
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(owned), chunk_size):
        part = owned[start : start + chunk_size]
        slots, preds, targets = zip(*part)
        slot_idx = torch.tensor(slots, dtype=torch.long, device=device)
        pred_idx = torch.tensor(preds, dtype=torch.long, device=device)
        target_idx = torch.tensor(targets, dtype=torch.long, device=device)
        selected = logits.index_select(0, pred_idx)
        lp, ent = gather_logprobs_entropy(
            selected,
            input_ids.index_select(0, target_idx),
            temperature=temperature,
            chunk_size=chunk_size,
            tp_group=tp_group,
        )
        scalars = scalars.index_copy(1, slot_idx, torch.stack((lp, ent)))
        stats[:, slot_idx] = torch.stack(
            (selected.detach().amin(-1), selected.detach().amax(-1))
        ).float()
    if plan.edges:
        scalars = dist_nn.all_reduce(scalars, op=dist.ReduceOp.SUM, group=cp_group)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=cp_group)
    logprobs, entropy = {}, {}
    min_parts, max_parts = [], []
    for sid, slots in plan.sequences:
        idx = torch.tensor(slots, dtype=torch.long, device=device)
        logprobs[sid] = scalars[0].index_select(0, idx)
        entropy[sid] = scalars[1].index_select(0, idx)
        min_parts.append(stats[0].index_select(0, idx))
        max_parts.append(stats[1].index_select(0, idx))
    empty = stats.new_empty(0)
    return (
        logprobs,
        entropy,
        torch.cat(min_parts) if min_parts else empty,
        torch.cat(max_parts) if max_parts else empty,
    )
