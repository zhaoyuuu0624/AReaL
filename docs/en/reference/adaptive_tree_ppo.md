# Adaptive tree DP x CP in PPOTrainer

`actor.adaptive_tree` opts the v1 single-controller Megatron actor into the tree runtime
from [adaptive tree parallelism](adaptive_tree_parallelism.md). The normal static path
remains the default. This first integration supports dense, critic-free PPO/GRPO with
tree training; it is experimental.

## Enable in an existing training configuration

Merge this fragment into a working Megatron GRPO configuration. Keep model paths,
rollout placement and cluster resources appropriate to your existing deployment. The
example assumes four **actor** ranks; rollout uses separately allocated resources. This
is not an eight-GPU requirement for the standalone oracle below.

```yaml
actor:
  backend: megatron:d4p1t1c1
  _version: v1
  enable_tree_training: true
  disable_dropout: true
  pad_to_maximum: true
  offload: false
  ppo_n_minibatches: 2
  recompute_logprob: true
  mb_spec:
    max_tokens_per_mb: 4096
  megatron:
    ddp:
      overlap_grad_reduce: false
      overlap_param_gather: false
    overlap_param_gather_with_optimizer_step: false
  adaptive_tree:
    cp_sizes: [1, 2, 4]
    local_token_budget: 1024
    max_tree_tokens: 4096
    replan_interval: 8
    workload_change_threshold: 0.25
    min_dwell_steps: 3
    min_relative_gain: 0.10
```

Set `AREAL_USE_TRITON_TREE_ATTN=0` and `AREAL_FLEX_ATTENTION_BLOCK_SIZE=128` in the
actor worker environment before imports. Every CP candidate must divide **both Q and KV
head counts**. In particular, a model with only two KV heads cannot use CP4. The fixed
actor world must be a power of two. Keep Megatron's `hybrid_context_parallel` disabled:
this uses its group-construction primitive, not the THD data scheduler.

The runtime retains its dense GPT/RoPE, FP32/BF16, zero-dropout, no-TP/PP/EP,
no-MoE/FP8/MTP/MLA/VLM/LoRA/offload/communication-overlap restrictions. BF16 numerical
coverage uses bias-free projections and RMSNorm; the earlier validation notes explain
the tighter Adam-equivalence limitations of biased models.

The trainer rejects critic, teacher/MOPD, SPMD, v2 controllers, M2PO masking and
actor/rollout colocation or AWEX. M2PO currently chooses masks within a packed
microbatch; changing packing would change training targets. A static reference engine is
allowed, with `mb_spec.max_tokens_per_mb >= max_tree_tokens`; its model and memory
capacity must also support that length. Adaptive CP applies to the actor, not the
reference. Token caps do not predict total GPU memory use.

## Execution and algorithm contract

1. After rollout filtering and auxiliary scoring, `PPOTrainer` calls
   `begin_adaptive_cycle(rollout_batch, cycle_id=global_step)` before actor logprob.
1. The controller stamps stable row IDs and sends the complete input to the source
   worker using a CPU-staged RPC with automatic broadcasting disabled. Only after
   materialization succeeds does it call all actor ranks with an opaque handle; no
   RTensor fetch occurs inside these collective requests.
1. The source defines balanced, contiguous **global optimizer minibatches** in original
   row order. Each real row participates exactly once per cycle. The number of updates
   is exactly `ppo_n_minibatches`, which must not exceed rows. This schedule is
   independent of DP/CP. It can differ from the legacy static controller's per-shard
   schedule: use the same global schedule in experiments.
1. One CP is chosen for the full forward batch and every optimizer minibatch. Candidates
   with an empty logical DP shard or insufficient token cap are excluded. Forward
   results are deduplicated and restored to original row order.
1. Advantages are computed once on the source from unique global samples, preserving
   `TrajBatchMeta` and `RolloutGroup` metadata. All reward/advantage normalizations use
   a source-only Gloo group. Thus sample counts, unbiased standard deviations and
   leave-one-out baselines do not include CP copies.
1. Each global optimizer minibatch is tree-partitioned and executed collectively with
   the original PPO loss options. DDP reduction and optimizer ownership stay in the
   original full actor world. Batch diagnostics are recorded once on the source; loss
   diagnostics are recorded once per logical CP group.
1. Only a complete successful cycle advances the policy's dwell counter. LR and weight
   versions advance through the original trainer code. No optimizer or parameter
   migration is performed. Execution bindings are restored after each forward/update; a
   partial execution exception is fail-stop, not a rollback.

Prepared inputs are immutable snapshots with single-use handles. Valid tokens, row order
and rollout grouping cannot change within a cycle. Transport padding may be trimmed by
RTensor publication. Repeated logprob/advantage phases and out-of-order optimizer plans
are rejected. Existing all-worker RTensor cleanup remains responsible for source storage
and every consumer's fetch buffers.

## Planning cost and remaining transport cost

The initial cycle searches all feasible CP candidates. Later cycles search when
`replan_interval` expires, the workload signature changes sufficiently, or the old
configuration becomes infeasible. The signature includes trajectory count, maximum/mean
length and unique prefix-token work. Dwell and relative-gain gates limit switching
between feasible configurations; infeasibility bypasses them.

Skipping search still builds fresh plans for the current batch at the selected CP. It
never reuses an earlier batch's row mapping. The objective is currently a heuristic in
work units, including a forward pass and optimizer batches, not a calibrated time or
memory model. External tree-aware modeling can supply cost coefficients and per-CP
multipliers.

This implementation retains **source materialization, CPU object scatter and source
output collection**. It does not implement CP leaders fetching their own RTensor shards.
These costs are distinct from switching prebuilt CP groups and must be included when
measuring end-to-end overhead. `adaptive_tree/*` statistics report search, planning
time, selected topology and cycle completion; existing trainer timings include the
surrounding RPC stages.

## Checkpoint recovery

The existing recovery manifest stores adaptive runtime state in `extra_state.json`,
inside the same checkpoint generation as model/optimizer publication. State is captured
only when the existing checkpoint frequency fires, and only at a completed cycle
boundary. It includes policy configuration, world size, committed CP, dwell/search
counters, completed cycle ID and the global minibatch count. Deterministic row
scheduling needs no additional shuffle RNG.

Recovery initializes the fixed base groups and candidate groups first, loads the
existing model/optimizer checkpoint, then installs the saved control state on all
workers before the next cycle. Configuration/world/minibatch mismatches fail. An
adaptive run cannot silently resume a legacy manifest without adaptive state. Old
manifests continue to work for ordinary static training. Prepared handles and partial
PPO cycles are deliberately not checkpointed.

## Bounded validation

From a complete project environment:

```bash
AREAL_USE_TRITON_TREE_ATTN=0 python -m pytest -q \
  tests/test_adaptive_tree_cycle.py tests/test_adaptive_ppo_bridge.py \
  tests/test_adaptive_ppo_guards.py -k 'not four_gpu'

AREAL_USE_TRITON_TREE_ATTN=0 AREAL_FLEX_ATTENTION_BLOCK_SIZE=128 \
  python -m pytest -q tests/test_adaptive_ppo_bridge.py -k four_gpu
```

The four-GPU oracle uses a small real Megatron model, real GRPO loss, distributed Adam,
heterogeneous synthetic rollout groups, proximal logprob recomputation and two global
optimizer minibatches per cycle. It compares CP1→2→4→1 with fixed CP1 under the same
global schedule, checks counts, and saves/restores DCP state before replaying a cycle.
BF16 also enables gradient checkpointing. This validates the training path; it is not a
long-running pretrained RL experiment or a live rollout server benchmark. Controller
ordering and preparation failures have CPU tests.
