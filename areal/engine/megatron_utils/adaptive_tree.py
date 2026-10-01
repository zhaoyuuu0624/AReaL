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

    def _prepare(
        self, batch: dict[str, torch.Tensor], cp_size: int | None
    ) -> tuple[TreeParallelPlan, list[dict[str, torch.Tensor]]]:
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
        plan = self.planner.plan(batch, cp_size)
        shards = []
        for rows in plan.shards:
            indices = torch.tensor(rows, dtype=torch.long, device="cpu")
            shards.append(
                {key: value.index_select(0, indices) for key, value in batch.items()}
            )
        return plan, [shards[rank // plan.cp_size] for rank in range(self.world_size)]

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
        if self.active or self.closed or self.failed:
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
        if self.active or self.failed:
            raise RuntimeError(
                "Cannot checkpoint active or failed adaptive control state"
            )
        return {
            "version": 1,
            "world_size": self.world_size,
            "config": asdict(self.config),
            "planner": self.planner.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if self.active or self.closed or self.failed:
            raise RuntimeError("Cannot restore an active or closed runtime")
        if (
            state["version"] != 1
            or state["world_size"] != self.world_size
            or state["config"] != asdict(self.config)
        ):
            raise ValueError(
                "Adaptive state requires the same world and policy configuration"
            )
        self.planner.load_state_dict(state["planner"])

    def close(self) -> None:
        """Collectively close before destroying the base MCore process groups."""
        if self.active:
            raise RuntimeError("Cannot close adaptive groups during a step")
        if not self.closed:
            for group in self.owned_groups:
                dist.destroy_process_group(group)
            self.closed = True
