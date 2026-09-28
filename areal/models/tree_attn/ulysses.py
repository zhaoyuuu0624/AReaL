# SPDX-License-Identifier: Apache-2.0

"""Uniform Ulysses CP for a packed tree (not a DFS/DTA executor).

The process group is explicit. Token shards are contiguous in CP-group rank
order; attention runs on the whole tree and a subset of heads. The inverse
all-to-all restores token ownership. Neither operation reduces parameter grads.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from areal.models.fsdp.ulysses import all_to_all_tensor


@dataclass(frozen=True)
class TreeCPLayout:
    global_tokens: int
    cp_size: int
    cp_rank: int

    def __post_init__(self) -> None:
        if self.cp_size < 1 or not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("Invalid tree CP size or group-local rank")
        if self.global_tokens < 1 or self.global_tokens % self.cp_size:
            raise ValueError("Padded tree length must be positive and divisible by CP")

    @property
    def local_tokens(self) -> int:
        return self.global_tokens // self.cp_size

    @property
    def start(self) -> int:
        return self.cp_rank * self.local_tokens

    def slice(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if tensor.shape[dim] != self.global_tokens:
            raise ValueError("Tree tensor does not match the global padded length")
        return tensor.narrow(dim, self.start, self.local_tokens).contiguous()


def sequence_to_head(x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """[S/c, B, H, D] -> [S, B, H/c, D], with autograd."""
    size = dist.get_world_size(cp_group)
    if x.ndim != 4 or x.shape[2] % size:
        raise ValueError("Ulysses requires SBHD and heads divisible by CP")
    if size == 1:
        return x
    return all_to_all_tensor(x, scatter_dim=2, gather_dim=0, group=cp_group)


def head_to_sequence(x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """Inverse of sequence_to_head; no gradient scaling is applied."""
    size = dist.get_world_size(cp_group)
    if x.ndim != 4 or x.shape[0] % size:
        raise ValueError("Ulysses requires SBHD and sequence divisible by CP")
    if size == 1:
        return x
    return all_to_all_tensor(x, scatter_dim=0, gather_dim=2, group=cp_group)


def tree_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run a local BHSD tree-attention callable using SBHD token shards.

    The callable must use the *global* tree mask and return BHSD. GQA uses the
    actual post-TP head counts; KV-head replication is deliberately unsupported.
    """
    size = dist.get_world_size(cp_group)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Tree Ulysses Q/K/V must use SBHD layout")
    if key.shape != value.shape or query.shape[:2] != key.shape[:2]:
        raise ValueError("Inconsistent tree Ulysses Q/K/V shapes")
    if query.shape[1] != 1 or query.shape[3] != key.shape[3]:
        raise ValueError("Tree Ulysses requires B=1 and matching Q/K head dimensions")
    if key.shape[2] < 1 or query.shape[2] % key.shape[2]:
        raise ValueError("Query heads must be a multiple of KV heads")
    if query.shape[2] % size or key.shape[2] % size:
        raise ValueError("Both local Q and KV head counts must be divisible by CP")
    q = sequence_to_head(query, cp_group).permute(1, 2, 0, 3)
    k = sequence_to_head(key, cp_group).permute(1, 2, 0, 3)
    v = sequence_to_head(value, cp_group).permute(1, 2, 0, 3)
    output = attention(q, k, v)
    if output.shape != q.shape:
        raise ValueError("Tree attention must return the query's BHSD shape")
    return head_to_sequence(output.permute(2, 0, 1, 3).contiguous(), cp_group)
