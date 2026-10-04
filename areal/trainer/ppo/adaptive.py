# SPDX-License-Identifier: Apache-2.0
"""Experimental, two-phase global-batch bridge for adaptive Megatron GRPO.

Only the source owns unique rollout/advantage data. Model execution is collective;
normalization uses a source-only CPU group. This deliberately retains centralized
CPU transport, while keeping optimizer minibatch membership independent of CP.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from areal.utils import stats_tracker
from areal.utils.data import concat_batch, split_batch

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronPPOActor

ROW_ID = "_adaptive_row_id"


def stamp_trajectory_ids(data: list[dict[str, Any]]) -> None:
    """Attach stable row IDs before any topology-dependent dispatch (RTensor safe)."""
    offset = 0
    for item in data:
        if ROW_ID in item:
            raise ValueError(f"Rollout already contains reserved field {ROW_ID}")
        mask = item["attention_mask"]
        shape = mask.shape if isinstance(mask, torch.Tensor) else mask.data.shape
        rows = shape[0]
        item[ROW_ID] = torch.arange(offset, offset + rows, dtype=torch.int64)
        offset += rows


def global_minibatch_indices(rows: int, count: int) -> tuple[tuple[int, ...], ...]:
    """Stable contiguous, balanced optimizer batches; never depend on DP/CP."""
    if rows < 1 or count < 1 or count > rows:
        raise ValueError(
            "PPO minibatches require 1 <= ppo_n_minibatches <= global rows"
        )
    size, remainder = divmod(rows, count)
    result, offset = [], 0
    for i in range(count):
        end = offset + size + (i < remainder)
        result.append(tuple(range(offset, end)))
        offset = end
    return tuple(result)


def _identity(batch: dict[str, torch.Tensor]) -> str:
    # RTensor publication may remove transport padding. Bind valid tokens and
    # row order, not a particular padded tensor width.
    digest = hashlib.sha256()
    digest.update(batch[ROW_ID].contiguous().numpy().tobytes())
    lengths = batch["attention_mask"].sum(-1, dtype=torch.int64)
    digest.update(lengths.numpy().tobytes())
    for row, length in zip(batch["input_ids"], lengths.tolist(), strict=True):
        digest.update(row[:length].to(torch.int64).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _group_signature(meta) -> tuple:
    return (tuple(meta.traj_group_sizes), tuple(meta.rollout_groups or ()))


def validate_adaptive_actor_config(config) -> None:
    """Reject unsupported semantics before initializing collective execution."""
    if config.adaptive_tree is None:
        return
    config.adaptive_tree.to_runtime_config()
    if config._version != "v1" or not config.backend.startswith("megatron:"):
        raise ValueError("adaptive_tree requires the v1 Megatron controller")
    if config.ppo_n_minibatches < 1:
        raise ValueError("adaptive_tree requires positive ppo_n_minibatches")
    if config.m2_threshold is not None:
        raise ValueError("adaptive_tree does not support packing-dependent M2PO masks")
    if not config.enable_tree_training or config.offload or config.is_critic:
        raise ValueError("adaptive_tree requires a tree actor without offload")
    if config.weight_update_mode == "awex":
        raise ValueError("adaptive_tree does not support AWEX weight residency")


class AdaptivePPOBridge:
    def __init__(self, engine: MegatronPPOActor):
        self.engine = engine
        self.actor = engine.actor
        validate_adaptive_actor_config(engine.config)
        self.runtime = engine.configure_adaptive_tree_parallelism(
            engine.config.adaptive_tree.to_runtime_config()
        )
        # All workers create it, only source participates in its normalizations.
        self.normalization_group = dist.new_group(
            ranks=[self.runtime.source], backend="gloo"
        )
        self.pending: tuple[str, dict, Any] | None = None
        self.cycle = None
        self.identity: str | None = None
        self.group_signature = None
        self.logp_done = False
        self.advantages_done = False
        self.last_cycle_id = -1
        self.closed = False

    def _check(self) -> None:
        if self.closed or self.runtime.failed:
            raise RuntimeError(
                "Adaptive PPO bridge is closed or failed; recover checkpoint"
            )

    def stage(self, data: list[dict[str, Any]], handle: str) -> dict[str, Any]:
        """Source-only RPC: localize and validate before any rank enters collectives."""
        self._check()
        if self.runtime.rank != 0 or self.pending is not None:
            raise RuntimeError("Only source may stage one outstanding adaptive input")
        if not data or not isinstance(handle, str) or not handle:
            raise ValueError("Adaptive staging needs nonempty data and handle")
        batch, meta = concat_batch(data)
        rows = batch["input_ids"].shape[0]
        for key, value in batch.items():
            if value is None:
                continue
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim == 0
                or value.shape[0] != rows
            ):
                raise ValueError(f"Adaptive field {key!r} must be a row-aligned tensor")
            batch[key] = value.detach().to(device="cpu", copy=True)
        batch = {key: value for key, value in batch.items() if value is not None}
        if ROW_ID not in batch or not torch.equal(
            batch[ROW_ID], torch.arange(rows, dtype=torch.int64)
        ):
            raise ValueError(
                "Adaptive trajectory IDs were missing, reordered or duplicated"
            )
        mask = batch["attention_mask"].bool()
        expected = torch.arange(mask.shape[1])[None] < mask.sum(-1, keepdim=True)
        if not torch.equal(mask, expected):
            raise ValueError(
                "Adaptive trajectories must be contiguous and right padded"
            )
        identity = _identity(batch)
        if self.cycle is not None and identity != self.identity:
            raise ValueError("Adaptive batch identity changed during the PPO cycle")
        if self.cycle is not None and _group_signature(meta) != self.group_signature:
            raise ValueError(
                "Adaptive rollout group metadata changed during the PPO cycle"
            )
        self.pending = (handle, batch, meta)
        return {"handle": handle, "rows": rows}

    def release(self, handle: str) -> None:
        if self.pending is not None and self.pending[0] == handle:
            self.pending = None

    def _take(self, handle: str):
        self._check()
        if self.runtime.rank != 0:
            return None, None
        if self.pending is None or self.pending[0] != handle:
            raise ValueError("Unknown, expired or already consumed adaptive handle")
        _, batch, meta = self.pending
        self.pending = None
        return batch, meta

    def _collective_take(
        self, handle: str, *, require_cycle: bool = True, forward: bool = False
    ):
        """Exchange readiness before model collectives; failures are rank-consistent."""
        batch, meta, error = None, None, None
        try:
            if require_cycle and self.cycle is None:
                raise RuntimeError("begin_adaptive_cycle must precede actor execution")
            if forward and (self.logp_done or self.advantages_done):
                raise RuntimeError(
                    "Adaptive logprob may run only once, before advantages"
                )
            batch, meta = self._take(handle)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        checks = [None] * self.runtime.world_size
        dist.all_gather_object(
            checks,
            (handle, error, None if self.cycle is None else self.cycle.cycle_id),
            group=self.runtime.cpu_group,
        )
        if any(item != checks[0] for item in checks) or any(item[1] for item in checks):
            raise RuntimeError(
                f"Adaptive workers disagree or input is unavailable: {checks}"
            )
        return batch, meta

    def begin(self, handle: str, cycle_id: int, cp_size: int | None = None) -> dict:
        if self.cycle is not None:
            raise RuntimeError("Previous adaptive PPO cycle has not completed")
        batch, meta = self._collective_take(handle, require_cycle=False)
        indices, identity, error = None, None, None
        if self.runtime.rank == 0:
            try:
                if cycle_id <= self.last_cycle_id:
                    raise ValueError("Adaptive cycle IDs must increase")
                indices = global_minibatch_indices(
                    batch["input_ids"].shape[0], self.engine.config.ppo_n_minibatches
                )
                identity = _identity(batch)
            except Exception as exc:
                error = str(exc)
        envelope = [(indices, identity, error)]
        dist.broadcast_object_list(
            envelope, src=self.runtime.source, group=self.runtime.cpu_group
        )
        indices, identity, error = envelope[0]
        if error:
            raise ValueError(error)
        self.cycle = self.runtime.plan_cycle(batch, indices, cycle_id, cp_size=cp_size)
        self.identity = identity
        self.group_signature = _group_signature(meta) if meta is not None else None
        self.logp_done = False
        self.advantages_done = False
        return {
            "cycle_id": cycle_id,
            "cp_size": self.cycle.forward_plan.cp_size,
            "dp_size": self.cycle.forward_plan.dp_size,
        }

    @torch.no_grad()
    def compute_logp(self, handle: str):
        batch, meta = self._collective_take(handle, forward=True)
        self.engine.eval()
        # Metadata is kept outside the model's row-aligned tensor contract.
        inputs = (
            None
            if batch is None
            else {key: batch[key] for key in ("input_ids", "attention_mask")}
        )
        output = self.runtime.forward_batch(inputs, self.cycle.forward_plan)
        self.logp_done = True
        return split_batch(output, meta) if self.runtime.rank == 0 else None

    @torch.no_grad()
    def compute_advantages(self, handle: str, **kwargs):
        """Source only: unique samples, original rollout groups, singleton reduction."""
        if self.cycle is None:
            raise RuntimeError("begin_adaptive_cycle must precede advantages")
        if self.advantages_done:
            raise RuntimeError("Advantages already computed for this cycle")
        if self.engine.config.should_compute_prox_logp() and not self.logp_done:
            raise RuntimeError("This PPO configuration requires proximal logprob first")
        batch, meta = self._take(handle)
        if self.runtime.rank != 0:
            return None
        result = self.actor._compute_advantages(
            batch, meta, normalization_group=self.normalization_group, **kwargs
        )
        self.advantages_done = True
        return split_batch(result, meta)

    def update(self, handle: str) -> dict:
        batch, meta = self._collective_take(handle)
        self.engine.train()
        error = None
        if self.runtime.rank == 0:
            try:
                if not self.advantages_done:
                    raise RuntimeError("Adaptive update requires completed advantages")
                required = (
                    "advantages",
                    "loss_mask",
                    "logprobs",
                    "rewards",
                    "tot_rewards",
                    "kl_rewards",
                )
                for key in required:
                    if key not in batch:
                        raise ValueError(f"PPO update is missing {key}")
                with stats_tracker.scope("ppo_actor"):
                    self.actor._record_batch_stats(batch, meta)
                for key in (
                    ROW_ID,
                    "rewards",
                    "tot_rewards",
                    "kl_rewards",
                    "is_truncated",
                    "token_rewards",
                ):
                    batch.pop(key, None)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        envelope = [error]
        dist.broadcast_object_list(
            envelope, src=self.runtime.source, group=self.runtime.cpu_group
        )
        if envelope[0] is not None:
            raise ValueError(envelope[0])
        successful = True
        try:
            with stats_tracker.scope("ppo_actor/update"):
                for rows, plan in zip(
                    self.cycle.minibatch_indices,
                    self.cycle.optimizer_plans,
                    strict=True,
                ):
                    local = None
                    if batch is not None:
                        index = torch.tensor(rows, dtype=torch.long)
                        local = {k: v.index_select(0, index) for k, v in batch.items()}
                    result = self.runtime.train_batch_with_plan(
                        local,
                        plan,
                        loss_fn=self.actor._make_loss_fn(
                            self.engine.get_version(),
                            record_stats=self.runtime.rank % plan.cp_size == 0,
                        ),
                        loss_weight_fn=lambda item: item["loss_mask"].count_nonzero(),
                    )
                    successful = successful and bool(result["update_successful"])
                    stats_tracker.scalar(**result)
            self.runtime.end_cycle(self.cycle, update_successful=successful)
        except BaseException:
            # A partially completed optimizer sequence cannot be replayed safely.
            self.runtime.failed = True
            raise
        self.last_cycle_id = self.cycle.cycle_id
        self.cycle = None
        self.identity = None
        self.group_signature = None
        if self.runtime.rank == 0:
            with stats_tracker.scope("adaptive_tree"):
                stats_tracker.scalar(**self.runtime.last_cycle_stats)
        return {"update_successful": successful, "cycle_id": self.last_cycle_id}

    def state_dict(self) -> dict:
        if self.cycle is not None or self.pending is not None:
            raise RuntimeError(
                "Adaptive PPO checkpoints require a complete cycle boundary"
            )
        return {
            "version": 1,
            "last_cycle_id": self.last_cycle_id,
            "ppo_n_minibatches": self.engine.config.ppo_n_minibatches,
            "runtime": self.runtime.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._check()
        if self.cycle is not None or self.pending is not None:
            raise RuntimeError("Cannot restore adaptive state with outstanding work")
        if (
            state["version"] != 1
            or state["ppo_n_minibatches"] != self.engine.config.ppo_n_minibatches
        ):
            raise ValueError(
                "Adaptive checkpoint minibatch schedule/configuration differs"
            )
        self.runtime.load_state_dict(state["runtime"])
        self.last_cycle_id = int(state["last_cycle_id"])

    def close(self) -> None:
        self.pending = None
        if not self.closed:
            if self.runtime.rank == 0:
                dist.destroy_process_group(self.normalization_group)
            self.closed = True
