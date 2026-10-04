# SPDX-License-Identifier: Apache-2.0
"""CPU planning for uniform, step-boundary tree DP x CP.

The default cost is an explicit heuristic in work units, not predicted seconds
or a GPU-memory guarantee. Calibrated per-CP multipliers can be supplied by an
external auto-parallel planner. No distributed state is touched here.
"""

import math
from dataclasses import dataclass

import torch

from areal.models.tree_attn.tree import _greedy_build_tries


@dataclass(frozen=True)
class AdaptiveTreeConfig:
    """Candidate policy; token limits include padding but exclude model state.

    ``local_token_budget`` bounds padded input tokens per CP rank. The global
    cap is min(local_token_budget * CP, max_tree_tokens); the latter also bounds
    the replicated dense mask. Neither limit is a complete memory estimator.
    CP values must divide both Q and KV heads and the fixed power-of-two world.
    """

    cp_sizes: tuple[int, ...] = (1, 2, 4)
    local_token_budget: int = 1024
    max_tree_tokens: int = 4096
    min_dwell_steps: int = 3
    min_relative_gain: float = 0.1
    attention_pair_cost: float = 1 / 1024
    communication_token_cost: float = 0.1
    microbatch_cost: float = 128.0
    cp_cost_multipliers: tuple[float, ...] = ()
    replan_interval: int = 8
    workload_change_threshold: float = 0.25

    def __post_init__(self) -> None:
        if not self.cp_sizes or tuple(sorted(set(self.cp_sizes))) != self.cp_sizes:
            raise ValueError("cp_sizes must be sorted, unique and nonempty")
        if any(c < 1 or c & (c - 1) for c in self.cp_sizes):
            raise ValueError("CP sizes must be positive powers of two")
        if any(
            v < 128 or v % 128 for v in (self.local_token_budget, self.max_tree_tokens)
        ):
            raise ValueError("Token budgets must be positive multiples of 128")
        if self.min_dwell_steps < 0 or not 0 <= self.min_relative_gain < 1:
            raise ValueError("Invalid dwell steps or relative gain")
        if (
            not isinstance(self.replan_interval, int)
            or isinstance(self.replan_interval, bool)
            or self.replan_interval < 1
            or not math.isfinite(self.workload_change_threshold)
            or self.workload_change_threshold < 0
        ):
            raise ValueError("Invalid replan interval or workload change threshold")
        costs = (
            self.attention_pair_cost,
            self.communication_token_cost,
            self.microbatch_cost,
        )
        if any(not math.isfinite(v) or v < 0 for v in costs):
            raise ValueError("Cost coefficients must be finite and nonnegative")
        if self.cp_cost_multipliers and (
            len(self.cp_cost_multipliers) != len(self.cp_sizes)
            or any(not math.isfinite(v) or v <= 0 for v in self.cp_cost_multipliers)
        ):
            raise ValueError("Provide one finite positive cost multiplier per CP size")


@dataclass(frozen=True)
class TreeParallelPlan:
    cp_size: int
    token_cap: int
    shards: tuple[tuple[int, ...], ...]
    microbatches: int
    packed_tokens: int
    allowed_attention_pairs: int
    estimated_cost: float

    @property
    def dp_size(self) -> int:
        return len(self.shards)


@dataclass(frozen=True)
class TreeCyclePlan:
    """One immutable topology for a PPO cycle and its global optimizer batches.

    ``minibatch_indices`` address the full input batch, while each optimizer
    plan's shards address its corresponding minibatch. These identities must
    remain fixed across CP candidates: changing topology cannot change SGD.
    """

    cycle_id: int
    forward_plan: TreeParallelPlan
    minibatch_indices: tuple[tuple[int, ...], ...]
    optimizer_plans: tuple[TreeParallelPlan, ...]
    workload_signature: tuple[float, ...]
    replanned: bool

    @property
    def cp_size(self) -> int:
        return self.forward_plan.cp_size

    @property
    def estimated_cost(self) -> float:
        return self.forward_plan.estimated_cost + sum(
            plan.estimated_cost for plan in self.optimizer_plans
        )


def _workload_signature(sequences: list[list[int]]) -> tuple[float, ...]:
    """Cheap tree-sensitive drift signal; no candidate packing or dense mask."""
    total = sum(map(len, sequences))
    unique = 0
    previous: list[int] = []
    for sequence in sorted(sequences):
        shared = 0
        for left, right in zip(previous, sequence):
            if left != right:
                break
            shared += 1
        unique += len(sequence) - shared
        previous = sequence
    return (
        float(len(sequences)),
        float(max(map(len, sequences))),
        total / len(sequences),
        float(unique),
    )


def validate_minibatch_indices(
    minibatch_indices: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    batch_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Require exactly one global PPO pass, without drops or repeated rows."""
    batches = tuple(tuple(rows) for rows in minibatch_indices)
    if not batches or any(not rows for rows in batches):
        raise ValueError("Optimizer minibatches must be nonempty")
    flat = [row for rows in batches for row in rows]
    if any(not isinstance(row, int) or isinstance(row, bool) for row in flat):
        raise ValueError("Optimizer minibatch row indices must be integers")
    if sorted(flat) != list(range(batch_size)):
        raise ValueError(
            "Optimizer minibatches must cover each global row exactly once"
        )
    return batches


def extract_sequences(batch: dict[str, torch.Tensor]) -> list[list[int]]:
    """Require CPU padded trajectories; preserve their original row identities."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if ids.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("Adaptive planning requires CPU input tensors")
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] == 0:
        raise ValueError("Expected nonempty matching [batch, sequence] input and mask")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must contain integer token IDs")
    sequences = [row[valid.bool()].tolist() for row, valid in zip(ids, mask)]
    if any(len(seq) < 2 for seq in sequences):
        raise ValueError("Each trajectory must contain at least two valid tokens")
    return sequences


def prefix_partition(
    sequences: list[list[int]], count: int
) -> tuple[tuple[int, ...], ...]:
    """Minimize maximum unique-token work over lexicographically contiguous shards.

    Binary search a feasible work budget, then split intervals to exactly count
    nonempty shards. This preserves prefix locality; it is not a joint optimal
    attention/communication-aware partitioner.
    """
    if not 1 <= count <= len(sequences):
        raise ValueError("Need at least one trajectory per logical DP shard")
    order = sorted(range(len(sequences)), key=lambda i: sequences[i])
    increments = []
    previous: list[int] = []
    for i in order:
        seq = sequences[i]
        shared = 0
        for a, b in zip(previous, seq):
            if a != b:
                break
            shared += 1
        increments.append(len(seq) - shared)
        previous = seq

    def partition_at(budget: int) -> list[list[int]]:
        shards: list[list[int]] = [[]]
        work = 0
        for pos, i in enumerate(order):
            extra = increments[pos] if shards[-1] else len(sequences[i])
            if shards[-1] and work + extra > budget:
                shards.append([])
                work = 0
                extra = len(sequences[i])
            shards[-1].append(i)
            work += extra
        return shards

    low, high = max(map(len, sequences)), sum(increments)
    while low < high:
        mid = (low + high) // 2
        if len(partition_at(mid)) <= count:
            high = mid
        else:
            low = mid + 1
    shards = partition_at(low)
    while len(shards) < count:
        index = max(range(len(shards)), key=lambda i: len(shards[i]))
        shard = shards[index]
        middle = len(shard) // 2
        shards[index : index + 1] = [shard[:middle], shard[middle:]]
    return tuple(tuple(shard) for shard in shards)


def make_plan(
    batch: dict[str, torch.Tensor],
    sequences: list[list[int]],
    world_size: int,
    cp_size: int,
    config: AdaptiveTreeConfig,
) -> TreeParallelPlan:
    """Account with AReaL's actual greedy packer, without constructing masks."""
    if cp_size not in config.cp_sizes or world_size % cp_size:
        raise ValueError("CP must be an enabled divisor of world size")
    cap = min(config.local_token_budget * cp_size, config.max_tree_tokens)
    if cap % cp_size or max(map(len, sequences)) > cap:
        raise ValueError("A trajectory exceeds this candidate's aligned token cap")
    shards = prefix_partition(sequences, world_size // cp_size)
    counts, tokens, pairs = [], [], []
    for shard in shards:
        index = torch.tensor(shard, dtype=torch.long, device="cpu")
        data = {
            key: batch[key].index_select(0, index)
            for key in ("input_ids", "attention_mask")
        }
        tries, lengths = _greedy_build_tries(data, cap)
        counts.append(len(tries))
        tokens.append(sum(lengths))
        pairs.append(
            sum(
                node.num_tokens * sum(a.num_tokens for a in node.ancestors)
                + node.num_tokens * (node.num_tokens + 1) // 2
                for trie in tries
                for node in trie.nodes
            )
        )
    microbatches = max(counts)
    # Every rank runs the same padded count, including dummy microbatches.
    padded = microbatches * cap
    compute = (padded + config.attention_pair_cost * max(pairs)) / cp_size
    communication = config.communication_token_cost * padded * (cp_size - 1) / cp_size
    cost = compute + communication + config.microbatch_cost * microbatches
    if config.cp_cost_multipliers:
        cost *= config.cp_cost_multipliers[config.cp_sizes.index(cp_size)]
    return TreeParallelPlan(
        cp_size, cap, shards, microbatches, sum(tokens), sum(pairs), cost
    )


class AdaptiveTreePlanner:
    """Select a uniform layout from the current global batch, with hysteresis."""

    def __init__(self, world_size: int, config: AdaptiveTreeConfig) -> None:
        if world_size < 1 or world_size & (world_size - 1):
            raise ValueError("Adaptive tree CP currently requires a power-of-two world")
        if any(world_size % cp for cp in config.cp_sizes):
            raise ValueError("Every CP candidate must divide world size")
        self.world_size = world_size
        self.config = config
        self.current_cp: int | None = None
        self.steps_since_switch = 0
        self.cycles_since_replan = 0
        self.last_workload: tuple[float, ...] | None = None
        self.last_cycle_id: int | None = None

    def plan(
        self, batch: dict[str, torch.Tensor], cp_size: int | None = None
    ) -> TreeParallelPlan:
        sequences = extract_sequences(batch)
        candidates = []
        for cp in self.config.cp_sizes:
            cap = min(self.config.local_token_budget * cp, self.config.max_tree_tokens)
            if (
                max(map(len, sequences)) <= cap
                and cap % cp == 0
                and len(sequences) >= self.world_size // cp
            ):
                candidates.append(
                    make_plan(batch, sequences, self.world_size, cp, self.config)
                )
        if not candidates:
            raise ValueError("No feasible tree DP x CP layout within the token budgets")
        by_cp = {p.cp_size: p for p in candidates}
        if cp_size is not None:
            if cp_size not in by_cp:
                raise ValueError(
                    "Requested CP is disabled or infeasible for this batch"
                )
            return by_cp[cp_size]
        best = min(candidates, key=lambda p: (p.estimated_cost, p.cp_size))
        current = by_cp.get(self.current_cp)
        # Infeasible old plans are never retained by dwell time or hysteresis.
        if current is not None and (
            self.steps_since_switch < self.config.min_dwell_steps
            or best.estimated_cost
            >= current.estimated_cost * (1 - self.config.min_relative_gain)
        ):
            return current
        return best

    def plan_cycle(
        self,
        batch: dict[str, torch.Tensor],
        minibatch_indices: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
        cycle_id: int,
        cp_size: int | None = None,
    ) -> TreeCyclePlan:
        """Plan a full PPO cycle without advancing any committed policy state.

        Reused topology still gets fresh row partitions and packing estimates.
        Infeasibility always forces a search, regardless of interval or dwell.
        Costs sum the forward and optimizer batch estimates in work units.
        """
        if (
            not isinstance(cycle_id, int)
            or isinstance(cycle_id, bool)
            or cycle_id < 0
            or (self.last_cycle_id is not None and cycle_id <= self.last_cycle_id)
        ):
            raise ValueError("Cycle ID must be a new monotonically increasing integer")
        sequences = extract_sequences(batch)
        indices = validate_minibatch_indices(minibatch_indices, len(sequences))
        signature = _workload_signature(sequences)
        max_length = max(map(len, sequences))
        min_rows = min(map(len, indices))
        feasible = [
            cp
            for cp in self.config.cp_sizes
            if min_rows >= self.world_size // cp
            and max_length
            <= min(self.config.local_token_budget * cp, self.config.max_tree_tokens)
            and min(self.config.local_token_budget * cp, self.config.max_tree_tokens)
            % cp
            == 0
        ]
        if not feasible:
            raise ValueError(
                "No feasible tree DP x CP layout for every optimizer minibatch"
            )
        if cp_size is not None and cp_size not in feasible:
            raise ValueError(
                "Requested CP is disabled or infeasible for this PPO cycle"
            )
        drifted = self.last_workload is None or any(
            abs(new - old) / max(abs(old), 1.0) >= self.config.workload_change_threshold
            for new, old in zip(signature, self.last_workload or ())
        )
        replan = (
            cp_size is not None
            or self.current_cp not in feasible
            or drifted
            or self.cycles_since_replan >= self.config.replan_interval - 1
        )
        candidates = (
            [cp_size]
            if cp_size is not None
            else feasible
            if replan
            else [self.current_cp]
        )
        minibatches = [
            {
                key: batch[key].index_select(0, torch.tensor(rows, dtype=torch.long))
                for key in ("input_ids", "attention_mask")
            }
            for rows in indices
        ]
        plans = []
        for cp in candidates:
            forward = make_plan(batch, sequences, self.world_size, cp, self.config)
            optimizer = tuple(
                make_plan(
                    data,
                    [sequences[row] for row in rows],
                    self.world_size,
                    cp,
                    self.config,
                )
                for data, rows in zip(minibatches, indices)
            )
            plans.append(
                TreeCyclePlan(cycle_id, forward, indices, optimizer, signature, replan)
            )
        best = min(plans, key=lambda plan: (plan.estimated_cost, plan.cp_size))
        current = next(
            (plan for plan in plans if plan.cp_size == self.current_cp), None
        )
        if (
            cp_size is None
            and current is not None
            and (
                self.steps_since_switch < self.config.min_dwell_steps
                or best.estimated_cost
                >= current.estimated_cost * (1 - self.config.min_relative_gain)
            )
        ):
            return current
        return best

    def commit_cycle(self, plan: TreeCyclePlan) -> None:
        """Advance dwell/search counters once per successfully completed cycle."""
        if self.last_cycle_id is not None and plan.cycle_id <= self.last_cycle_id:
            raise ValueError("Cannot commit an already completed or stale cycle")
        self.commit(plan.forward_plan)
        self.last_cycle_id = plan.cycle_id
        if plan.replanned:
            self.cycles_since_replan = 0
            self.last_workload = plan.workload_signature
        else:
            self.cycles_since_replan += 1

    def commit(self, plan: TreeParallelPlan) -> None:
        """Advance control state only after a successful optimizer update."""
        if plan.cp_size != self.current_cp:
            self.current_cp = plan.cp_size
            self.steps_since_switch = 0
        self.steps_since_switch += 1

    def state_dict(self) -> dict:
        return {
            "current_cp": self.current_cp,
            "steps_since_switch": self.steps_since_switch,
            "cycles_since_replan": self.cycles_since_replan,
            "last_workload": self.last_workload,
            "last_cycle_id": self.last_cycle_id,
        }

    def load_state_dict(self, state: dict) -> None:
        cp, steps = state["current_cp"], state["steps_since_switch"]
        if cp is not None and cp not in self.config.cp_sizes:
            raise ValueError("Saved CP is outside the configured candidates")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid saved dwell counter")
        cycles = state.get("cycles_since_replan", 0)
        last_cycle = state.get("last_cycle_id")
        workload = state.get("last_workload")
        if (
            not isinstance(cycles, int)
            or isinstance(cycles, bool)
            or cycles < 0
            or (
                last_cycle is not None
                and (
                    not isinstance(last_cycle, int)
                    or isinstance(last_cycle, bool)
                    or last_cycle < 0
                )
            )
            or (
                workload is not None
                and (
                    len(workload) != 4
                    or any(not math.isfinite(value) or value <= 0 for value in workload)
                )
            )
        ):
            raise ValueError("Invalid saved cycle policy state")
        self.current_cp, self.steps_since_switch = cp, steps
        self.cycles_since_replan = cycles
        self.last_cycle_id = last_cycle
        self.last_workload = None if workload is None else tuple(workload)
