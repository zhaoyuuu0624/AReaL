# SPDX-License-Identifier: Apache-2.0
"""AReaL-owned tree execution over Megatron's prebuilt hybrid CP groups.

This explicit collective API accepts one global CPU batch on rank zero. It does
not change the static RPC controller's allocation, MCore parallel state, DDP,
or optimizer. Do not invoke it through the ordinary per-DP batch dispatcher.
"""

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from areal.models.tree_attn.adaptive import (
    AdaptiveTreeConfig,
    AdaptiveTreePlanner,
    TreeCyclePlan,
    TreeParallelPlan,
    extract_sequences,
)

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine


class AdaptiveTreeRuntime:
    def __init__(self, engine: "MegatronEngine", config: AdaptiveTreeConfig) -> None:
        from megatron.core import parallel_state as mpu

        from areal.engine.megatron_utils.tree_context_parallel import (
            supports_tree_rotary_positions,
        )
        from areal.models.tree_attn.constants import BLOCK_SIZE, USE_TRITON_TREE_ATTN
        from areal.models.tree_attn.module_megatron import PytorchFlexAttention

        self.engine = engine
        self.config = config
        self.world_group = mpu.get_data_parallel_group(with_context_parallel=True)
        self.cpu_group = engine.cpu_group
        self.ranks = dist.get_process_group_ranks(self.world_group)
        self.rank = dist.get_rank(self.world_group)
        self.world_size = len(self.ranks)
        self.source = self.ranks[0]
        self.active = False
        self.closed = False
        self.failed = False
        self.last_plan: TreeParallelPlan | None = None
        self.pending_cycle: TreeCyclePlan | None = None
        self.last_finished_cycle_id: int | None = None
        self.last_cycle_stats: dict[str, float] = {}
        self._cycle_updates = 0
        self._cycle_successful = True
        self.attentions = []
        error = None
        try:
            self.planner = AdaptiveTreePlanner(self.world_size, config)
            if USE_TRITON_TREE_ATTN or BLOCK_SIZE != 128:
                raise ValueError(
                    "Adaptive trees require FlexAttention with block size 128"
                )
            tf = engine.tf_config
            if (
                not engine.enable_tree_training
                or len(engine.model) != 1
                or not supports_tree_rotary_positions(engine.model[0])
                or mpu.get_tensor_model_parallel_world_size() != 1
                or mpu.get_pipeline_model_parallel_world_size() != 1
                or mpu.get_context_parallel_world_size() != 1
                or mpu.get_expert_model_parallel_world_size() != 1
                or tf.context_parallel_size != 1
                or tf.hybrid_context_parallel
            ):
                raise ValueError(
                    "Adaptive trees require standard GPT/RoPE, TP=PP=EP=CP=1 base initialization and hybrid_context_parallel=False"
                )
            if (
                engine.config.is_critic
                or engine.config.use_lora
                or engine.is_vision_model
                or engine.use_padded_seq
                or engine.use_model_packed_seq
                or engine.enable_fp8
                or engine.mcore_config.enable_mtp_training
                or tf.multi_latent_attention
                or tf.num_moe_experts
                or getattr(tf, "mtp_num_layers", None)
                or tf.attention_dropout != 0
                or tf.hidden_dropout != 0
                or not engine.config.pad_to_maximum
                or engine.config.offload
                or engine.dtype not in (torch.float32, torch.bfloat16)
            ):
                raise ValueError(
                    "Adaptive trees require padded dense text actors, FP32/BF16, zero dropout, no offload/LoRA/MTP"
                )
            ddp = engine.model[0]
            if (
                ddp.ddp_config.overlap_grad_reduce
                or ddp.ddp_config.overlap_param_gather
                or engine.mcore_config.overlap_param_gather_with_optimizer_step
            ):
                raise ValueError(
                    "Adaptive trees require non-overlapped gradient reduction and parameter gathering"
                )
            if dist.get_process_group_ranks(self.cpu_group) != self.ranks:
                raise ValueError("CPU control group must match the fixed DP+CP domain")
            if self.ranks != list(range(dist.get_world_size())):
                raise ValueError(
                    "First adaptive version requires the full training world"
                )
            self.attentions = [
                m
                for m in engine.model[0].modules()
                if isinstance(m, PytorchFlexAttention)
            ]
            if len(self.attentions) != tf.num_layers:
                raise ValueError("Every layer must use AReaL PytorchFlexAttention")
            if any(
                tf.num_attention_heads % cp or tf.num_query_groups % cp
                for cp in config.cp_sizes
            ):
                raise ValueError("Every CP candidate must divide both Q and KV heads")
        except (ValueError, AttributeError) as exc:
            error = str(exc)
        # Fail collectively before any rank starts subgroup creation.
        checks = [None] * self.world_size
        dist.all_gather_object(checks, (asdict(config), error), group=self.cpu_group)
        if any(item[1] is not None for item in checks):
            raise ValueError(
                f"Invalid adaptive tree runtime: {[item[1] for item in checks]}"
            )
        if any(item[0] != checks[0][0] for item in checks):
            raise ValueError("Adaptive tree configuration differs across ranks")
        # This is the group-construction primitive used by MCore Dynamic CP.
        # Keep ownership local; do not mutate _HYBRID_DP_CP_GROUPS or mpu getters.
        factory = getattr(mpu, "create_hybrid_dp_cp_groups", None)
        if factory is None:
            raise RuntimeError("Installed MCore lacks create_hybrid_dp_cp_groups")
        self.groups = factory(dist.get_rank(), self.ranks, None)
        self.owned_groups = list(self.groups.values())
        # MCore's helper deliberately omits these two endpoints.
        self.groups[1] = mpu.get_context_parallel_group()
        self.groups[self.world_size] = self.world_group
        for cp in config.cp_sizes:
            group = self.groups[cp]
            if (
                dist.get_world_size(group) != cp
                or dist.get_rank(group) != self.rank % cp
            ):
                raise RuntimeError("Unexpected MCore hybrid CP group layout")

    @contextmanager
    def execution(self, plan: TreeParallelPlan):
        if self.active or self.closed or self.failed:
            raise RuntimeError("Adaptive tree runtime is active, closed, or failed")
        engine = self.engine
        old_groups = [module.cp_group for module in self.attentions]
        self.active = True
        engine._active_tree_group = self.groups[plan.cp_size]
        engine._active_tree_cap = plan.token_cap
        try:
            for module in self.attentions:
                module.cp_group = self.groups[plan.cp_size]
            yield
        except BaseException:
            # Restoring Python bindings cannot roll back partially executed
            # optimizer/collective operations. Restart from a checkpoint instead.
            self.failed = True
            raise
        finally:
            # The context encloses the full backward/recompute and optimizer step.
            for module, previous in zip(self.attentions, old_groups):
                module.cp_group = previous
            engine._active_tree_group = None
            engine._active_tree_cap = None
            self.active = False

    @staticmethod
    def _validate_batch(batch: dict[str, torch.Tensor]) -> None:
        extract_sequences(batch)
        count = batch["input_ids"].shape[0]
        for key, value in batch.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.ndim == 0
                or value.shape[0] != count
            ):
                raise ValueError(
                    f"Adaptive global batch field {key!r} must be a row-aligned CPU tensor"
                )
        mask = batch["attention_mask"].bool()
        expected = torch.arange(mask.shape[1], device="cpu")[None] < mask.sum(
            -1, keepdim=True
        )
        if not torch.equal(mask, expected):
            raise ValueError(
                "Adaptive batches require contiguous right-padded trajectories"
            )

    def _prepare_payloads(
        self, batch: dict[str, torch.Tensor], plan: TreeParallelPlan
    ) -> list[dict[str, torch.Tensor]]:
        self._validate_batch(batch)
        if (
            plan.cp_size not in self.config.cp_sizes
            or plan.dp_size * plan.cp_size != self.world_size
            or plan.token_cap
            != min(
                self.config.local_token_budget * plan.cp_size,
                self.config.max_tree_tokens,
            )
            or any(not rows for rows in plan.shards)
            or sorted(row for rows in plan.shards for row in rows)
            != list(range(batch["input_ids"].shape[0]))
            or int(batch["attention_mask"].sum(-1).max()) > plan.token_cap
        ):
            raise ValueError("Prepared tree plan does not match the batch or runtime")
        shards = []
        for rows in plan.shards:
            indices = torch.tensor(rows, dtype=torch.long, device="cpu")
            shards.append(
                {key: value.index_select(0, indices) for key, value in batch.items()}
            )
        return [shards[rank // plan.cp_size] for rank in range(self.world_size)]

    def _prepare(
        self, batch: dict[str, torch.Tensor], cp_size: int | None
    ) -> tuple[TreeParallelPlan, list[dict[str, torch.Tensor]]]:
        self._validate_batch(batch)
        plan = self.planner.plan(batch, cp_size)
        return plan, self._prepare_payloads(batch, plan)

    def _collective_check(self, value: object, error: str | None = None) -> None:
        """Agree on control flow before a rank can enter a GPU collective."""
        if self.active or self.closed or self.failed:
            error = error or "Adaptive tree runtime is active, closed, or failed"
        checks = [None] * self.world_size
        dist.all_gather_object(checks, (value, error), group=self.cpu_group)
        if any(item[1] is not None for item in checks):
            raise ValueError(f"Invalid collective adaptive operation: {checks}")
        if any(item[0] != checks[0][0] for item in checks):
            raise ValueError(
                "Adaptive execution plans or cycle IDs differ across ranks"
            )

    def plan_cycle(
        self,
        global_batch: dict[str, torch.Tensor] | None,
        minibatch_indices: list[tuple[int, ...]] | tuple[tuple[int, ...], ...] | None,
        cycle_id: int,
        cp_size: int | None = None,
    ) -> TreeCyclePlan:
        """Collectively prepare a PPO cycle; only rank zero supplies input data.

        The full and optimizer batches share one CP. All source-side errors are
        broadcast before scatter, and policy state advances only in end_cycle.
        """
        error = None
        if self.pending_cycle is not None:
            error = "Finish the pending PPO cycle before planning another"
        elif (
            not isinstance(cycle_id, int) or isinstance(cycle_id, bool) or cycle_id < 0
        ):
            error = "Cycle ID must be a nonnegative integer"
        elif (
            self.last_finished_cycle_id is not None
            and cycle_id <= self.last_finished_cycle_id
        ):
            error = "Cannot reuse a completed PPO cycle ID"
        self._collective_check(
            (
                "plan_cycle",
                cycle_id,
                self.last_finished_cycle_id,
                self.planner.state_dict(),
            ),
            error,
        )
        start = time.perf_counter()
        envelope = [None]
        if self.rank == 0:
            try:
                self._validate_batch(global_batch)
                plan = self.planner.plan_cycle(
                    global_batch, minibatch_indices, cycle_id, cp_size
                )
                envelope[0] = (plan, None)
            except Exception as exc:
                envelope[0] = (None, f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(envelope, src=self.source, group=self.cpu_group)
        plan, error = envelope[0]
        if error is not None:
            raise ValueError(f"Adaptive cycle planning failed on source: {error}")
        self.pending_cycle = plan
        self._cycle_updates = 0
        self._cycle_successful = True
        self.last_cycle_stats = {
            "adaptive_cycle_id": plan.cycle_id,
            "adaptive_cp_size": plan.cp_size,
            "adaptive_dp_size": plan.forward_plan.dp_size,
            "adaptive_replanned": float(plan.replanned),
            "adaptive_planning_seconds": time.perf_counter() - start,
            "adaptive_estimated_cycle_cost": plan.estimated_cost,
            "adaptive_switched": float(
                self.planner.current_cp is not None
                and self.planner.current_cp != plan.cp_size
            ),
        }
        return plan

    def _distribute_prepared(
        self,
        global_batch: dict[str, torch.Tensor] | None,
        plan: TreeParallelPlan,
    ) -> dict[str, torch.Tensor]:
        envelope, payloads = [None], None
        if self.rank == 0:
            try:
                payloads = self._prepare_payloads(global_batch, plan)
                envelope[0] = None
            except Exception as exc:
                envelope[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(envelope, src=self.source, group=self.cpu_group)
        if envelope[0] is not None:
            raise ValueError(
                f"Adaptive batch distribution failed on source: {envelope[0]}"
            )
        local = [None]
        dist.scatter_object_list(local, payloads, src=self.source, group=self.cpu_group)
        return local[0]

    @torch.no_grad()
    def forward_batch(
        self,
        global_batch: dict[str, torch.Tensor] | None,
        plan: TreeParallelPlan,
    ) -> torch.Tensor | None:
        """Return original-order, zero-padded CPU logprobs only on rank zero."""
        error = None
        if self.pending_cycle is None or plan != self.pending_cycle.forward_plan:
            error = "Forward requires the pending PPO cycle's full-batch plan"
        elif self._cycle_updates:
            error = "PPO logprobs must be computed before optimizer updates"
        self._collective_check(("forward", plan), error)
        local = self._distribute_prepared(global_batch, plan)
        with self.execution(plan):
            result = self.engine.forward_batch(local)
            torch.cuda.synchronize(self.engine.device)
            # Every CP rank reconstructs scalars; send a single copy per group.
            result = result.detach().cpu() if self.rank % plan.cp_size == 0 else None
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(result, gathered, dst=self.source, group=self.cpu_group)
        envelope, output = [None], None
        if self.rank == 0:
            try:
                output = self._merge_forward_outputs(gathered, plan, global_batch)
            except Exception as exc:
                envelope[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(envelope, src=self.source, group=self.cpu_group)
        if envelope[0] is not None:
            raise ValueError(
                f"Adaptive forward result collection failed: {envelope[0]}"
            )
        self.last_plan = plan
        return output

    @staticmethod
    def _merge_forward_outputs(
        gathered: list[torch.Tensor | None],
        plan: TreeParallelPlan,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        leaders = [gathered[index * plan.cp_size] for index in range(plan.dp_size)]
        if any(
            not isinstance(result, torch.Tensor)
            or result.ndim != 2
            or result.shape[0] != len(rows)
            or result.shape[1] > batch["input_ids"].shape[1]
            for rows, result in zip(plan.shards, leaders)
        ):
            raise ValueError("CP leaders returned invalid logprob row shapes")
        output = torch.zeros(
            batch["input_ids"].shape, dtype=leaders[0].dtype, device="cpu"
        )
        for rows, result in zip(plan.shards, leaders):
            output[torch.tensor(rows, dtype=torch.long), : result.shape[1]] = result
        return output

    def train_batch_with_plan(
        self,
        global_batch: dict[str, torch.Tensor] | None,
        plan: TreeParallelPlan,
        loss_fn: Callable,
        loss_weight_fn: Callable,
    ) -> dict[str, float]:
        """Execute the next planned optimizer batch, without policy replanning."""
        error = None
        if (
            self.pending_cycle is None
            or self._cycle_updates >= len(self.pending_cycle.optimizer_plans)
            or plan != self.pending_cycle.optimizer_plans[self._cycle_updates]
        ):
            error = "Train requires the next optimizer plan of the pending PPO cycle"
        self._collective_check(("train", self._cycle_updates, plan), error)
        local = self._distribute_prepared(global_batch, plan)
        with self.execution(plan):
            stats = self.engine.train_batch(local, loss_fn, loss_weight_fn)
            torch.cuda.synchronize(self.engine.device)
        self._cycle_updates += 1
        self._cycle_successful = self._cycle_successful and bool(
            stats["update_successful"]
        )
        self.last_plan = plan
        stats.update(
            adaptive_cp_size=plan.cp_size,
            adaptive_dp_size=plan.dp_size,
            adaptive_token_cap=plan.token_cap,
            adaptive_estimated_cost=plan.estimated_cost,
            adaptive_packed_tokens=plan.packed_tokens,
            adaptive_allowed_attention_pairs=plan.allowed_attention_pairs,
        )
        return stats

    def end_cycle(self, plan: TreeCyclePlan, update_successful: bool = True) -> None:
        """Finish once; never count forward or individual PPO minibatches as dwell."""
        error = None
        if plan != self.pending_cycle or self._cycle_updates != len(
            plan.optimizer_plans
        ):
            error = "Cannot finish an absent, mismatched, or incomplete PPO cycle"
        successful = bool(update_successful) and self._cycle_successful
        self._collective_check(("end_cycle", plan.cycle_id, successful), error)
        if successful:
            self.planner.commit_cycle(plan)
        self.last_finished_cycle_id = plan.cycle_id
        self.pending_cycle = None
        self.last_cycle_stats["adaptive_cycle_successful"] = float(successful)

    def train_batch(
        self,
        global_batch: dict[str, torch.Tensor] | None,
        loss_fn: Callable,
        loss_weight_fn: Callable,
        cp_size: int | None = None,
    ) -> dict[str, float]:
        """All ranks enter; only source rank supplies a global batch/CP override.

        CPU scatter is a correctness-first transport. Its overhead is included in
        reported time. Production rollout-controller integration is separate.
        """
        if self.active or self.closed or self.failed or self.pending_cycle is not None:
            raise RuntimeError("Adaptive tree runtime is active, closed, or failed")
        torch.cuda.synchronize(self.engine.device)
        start = time.perf_counter()
        envelope, payloads = [None], None
        if self.rank == 0:
            try:
                plan, payloads = self._prepare(global_batch, cp_size)
                envelope[0] = (plan, None)
            except Exception as exc:
                envelope[0] = (None, f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(envelope, src=self.source, group=self.cpu_group)
        plan, error = envelope[0]
        if error is not None:
            raise ValueError(f"Adaptive tree planning failed on source: {error}")
        local = [None]
        dist.scatter_object_list(local, payloads, src=self.source, group=self.cpu_group)
        with self.execution(plan):
            stats = self.engine.train_batch(local[0], loss_fn, loss_weight_fn)
            # Surface asynchronous CUDA failures before restoring the bindings;
            # execution() must poison the runtime for these failures too.
            torch.cuda.synchronize(self.engine.device)
        seconds = torch.tensor(
            time.perf_counter() - start, dtype=torch.float64, device="cpu"
        )
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX, group=self.cpu_group)
        switched = (
            self.planner.current_cp is not None
            and self.planner.current_cp != plan.cp_size
        )
        if stats["update_successful"]:
            self.planner.commit(plan)
        self.last_plan = plan
        stats.update(
            adaptive_cp_size=plan.cp_size,
            adaptive_dp_size=plan.dp_size,
            adaptive_token_cap=plan.token_cap,
            adaptive_switched=float(switched),
            adaptive_step_seconds=float(seconds),
            adaptive_estimated_cost=plan.estimated_cost,
            adaptive_packed_tokens=plan.packed_tokens,
            adaptive_allowed_attention_pairs=plan.allowed_attention_pairs,
        )
        return stats

    def state_dict(self) -> dict:
        if self.active or self.failed or self.pending_cycle is not None:
            raise RuntimeError(
                "Cannot checkpoint active, pending, or failed adaptive control state"
            )
        return {
            "version": 2,
            "world_size": self.world_size,
            "config": asdict(self.config),
            "planner": self.planner.state_dict(),
            "last_finished_cycle_id": self.last_finished_cycle_id,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.active or self.closed or self.failed or self.pending_cycle is not None:
            raise RuntimeError("Cannot restore an active or closed runtime")
        # JSON checkpoints turn tuples into lists; normalize known tuple fields.
        saved_config = dict(state["config"])
        for key in ("cp_sizes", "cp_cost_multipliers"):
            if key in saved_config:
                saved_config[key] = tuple(saved_config[key])
        if state["version"] == 1:
            saved_config.setdefault("replan_interval", 8)
            saved_config.setdefault("workload_change_threshold", 0.25)
        if (
            state["version"] not in (1, 2)
            or state["world_size"] != self.world_size
            or saved_config != asdict(self.config)
        ):
            raise ValueError(
                "Adaptive state requires the same world and policy configuration"
            )
        restored_planner = AdaptiveTreePlanner(self.world_size, self.config)
        restored_planner.load_state_dict(state["planner"])
        finished = state.get("last_finished_cycle_id", restored_planner.last_cycle_id)
        if finished is not None and (
            not isinstance(finished, int)
            or isinstance(finished, bool)
            or finished < 0
            or (
                restored_planner.last_cycle_id is not None
                and finished < restored_planner.last_cycle_id
            )
        ):
            raise ValueError("Invalid saved last-finished cycle ID")
        self.planner.load_state_dict(restored_planner.state_dict())
        self.last_finished_cycle_id = finished

    def close(self) -> None:
        """Collectively close before destroying the base MCore process groups."""
        if self.active:
            raise RuntimeError("Cannot close adaptive groups during a step")
        if not self.closed:
            for group in self.owned_groups:
                dist.destroy_process_group(group)
            self.closed = True
