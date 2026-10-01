# Experimental adaptive tree DP x CP

This AReaL extension changes the **execution** DP/CP layout between complete
optimizer steps. Model parameters, optimizer instances, optimizer shard ownership,
the gradient reduction domain, and Megatron's base parallel state remain fixed.
It extends the packed-tree Ulysses path, not the DFS/DTA executor.

## Supported contract

- Initialize the normal Megatron engine with TP=PP=EP=CP=1 and DP equal to the
  fixed training world. The world size must be a power of two.
- Dense text GPT with standard RoPE, AReaL `PytorchFlexAttention` in every layer,
  FP32/BF16, zero attention and hidden dropout, `pad_to_maximum=True`.
- CP candidates must divide both Q and KV head counts. KV-head replication is
  not implemented. The implementation requires FlexAttention/block size 128.
- MCore AdamW can retain replicated or distributed optimizer state. Disable
  gradient reduction/parameter gather overlap and gather-with-optimizer-step.
- No TP/PP/EP expansion, FP8, MoE, MTP, MLA, VLM, LoRA, engine offload, concurrent
  engine calls, or per-microbatch heterogeneous CP in this first version.
- Each logical DP shard must contain at least one complete trajectory. Candidates
  with too many DP shards are excluded. A trajectory is never truncated.

The ordinary `train_batch`, RPC controller allocation, and RLTrainer dispatcher
remain static. The new API is an **explicit all-worker global-batch collective**.
Do not send it only to existing DP heads or pass it an already sharded batch.
Integrating the normal rollout/controller path requires a global plan boundary
before actor logprob computation/update and is not supplied by a YAML flag here.

## Reused Megatron Dynamic CP interface

The runtime calls MCore's `parallel_state.create_hybrid_dp_cp_groups` to prebuild
the intermediate power-of-two CP groups, and borrows its existing singleton CP
and full DP+CP groups for the endpoints. It checks group sizes/rank ordering.
No changes to the installed Megatron package or `_HYBRID_DP_CP_GROUPS` are needed.

Keep `hybrid_context_parallel=False`. MCore's full hybrid scheduler consumes a
different packed-sequence/data-iterator contract. AReaL's tree path needs its
global tree mask, original trajectory loss metadata, and depth-indexed RoPE.
It therefore reuses the group primitive while supplying its own tree planner,
CPU batch distribution and execution binding. It does not enable the blog's
THD scheduler or pass THD `PackedSeqParams` to tree attention.

Reference:
[NVIDIA Dynamic CP blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/).

## Calling the engine

After normal engine initialization, all ranks call:

```python
from areal.models.tree_attn.adaptive import AdaptiveTreeConfig

runtime = engine.configure_adaptive_tree_parallelism(
    AdaptiveTreeConfig(
        cp_sizes=(1, 2, 4),
        local_token_budget=2048,
        max_tree_tokens=8192,
        min_dwell_steps=3,
        min_relative_gain=0.10,
    )
)

# All ranks enter this call. Only rank zero supplies the complete global batch.
# Every field must be a CPU tensor with the same leading batch dimension.
# input_ids/attention_mask describe contiguous right-padded trajectories.
# Preserve loss masks, advantages, old/reference logprobs, and other row-aligned
# loss metadata. The caller computes any global/group-relative preprocessing.
stats = engine.train_adaptive_tree_batch(
    global_batch if rank == 0 else None,
    loss_fn=loss_fn,
    loss_weight_fn=loss_weight_fn,
)
```

`cp_size=2` optionally overrides automatic selection on the source rank, for
numerical validation/profiling. Infeasible overrides are rejected collectively.
Omitting it selects from the current global workload. The learning-rate scheduler
and weight version advance according to the caller's ordinary engine contract;
topology selection does not introduce an extra optimizer/LR/version step.

Normal AReaL engine initialization currently requires the distributed optimizer
for its DCP checkpointer. Replicated-optimizer coverage uses the small replay/test
engine factory. The pretrained initialization path itself is not exercised by
the random-model tests in this extension.

## Initial adaptive policy

For each CP candidate, the planner sets a padded tree token cap to
`min(local_token_budget * CP, max_tree_tokens)`. It excludes candidates that cannot
hold the longest trajectory. Trajectories are sorted lexicographically and
partitioned into nonempty contiguous logical-DP shards, minimizing maximum unique
token count within that restricted partition space. This preserves prefix locality.

The planner then runs AReaL's actual greedy packer without constructing masks. Its
cost includes padded/dummy microbatch token work, semantic ancestor-attention pairs,
a CP communication proxy, and microbatch overhead. It selects the minimum-cost
candidate, retaining the old feasible configuration until both the dwell time and
relative-gain threshold are satisfied. An infeasible old configuration bypasses
these gates. CP can decrease as well as increase.

These coefficients are **heuristics in work units**, not fitted latency or memory
models. `cp_cost_multipliers` accepts one externally calibrated multiplier per CP
candidate. The runtime reports measured time, but does not silently fit this model
from warmup/JIT-contaminated samples. An existing auto-parallel planner can replace
or calibrate this decision layer independently of the execution API.

`local_token_budget` is not a total GPU memory bound. The global dense mask remains
replicated, scaling quadratically with the padded tree cap. `max_tree_tokens`
limits this term explicitly; choose both limits for the actual model/hardware.
Semantic attention-pair counts are not measured kernel FLOPs.

## Loss and optimizer invariants

Let W be the fixed world, C the active CP, M the synchronized microbatch count,
and G the global original loss weight. Packing is synchronized over the fixed
world; CP partners receive identical trajectories and loss metadata.

- Manual normalization: the existing weight sum is C*G. Retain the existing
  **base** DP multiplier W*M. The base CP1 schedule divides by M, the differentiable
  CP scalar SUM accumulates C consumer gradients, and the fixed DDP reducer averages
  over W. Do not substitute logical DP=W/C or add another CP division.
- Per-token normalization: split each original integer microbatch weight across
  active CP ranks using quotient/remainder. The full-world token sum is G; MCore
  uses SUM gradients and finalizes with this original token denominator.
- Parameter/Adam shard layouts and reduction groups never change. Distributed
  optimizer mode uses its fixed reduce-scatter and parameter all-gather.
- Bind attention CP groups and input/loss CP helpers for the entire forward,
  backward/recomputation and optimizer step. Restore them afterward. Never mutate
  the shared `TransformerConfig.context_parallel_size` or global `mpu` getters.

Source-side planning/input errors are broadcast before scatter so every rank
raises coherently. An execution error marks the runtime failed; restoring Python
bindings is not a rollback. Restart the distributed job from a valid checkpoint
instead of trying another CP after a partially executed collective.

## Control state and lifecycle

Save `runtime.state_dict()` alongside the ordinary model/optimizer checkpoint;
restore it on all ranks with `runtime.load_state_dict(...)` at a quiescent boundary.
It contains the active policy configuration, last committed CP, and dwell counter.
Use a serializer that preserves tuple values (for example `torch.save`). World and
policy configuration must match on restore. The extension does not automatically
add this state to AReaL's DCP checkpoint manager.

`engine.destroy()` closes the owned hybrid groups. Standalone test/replay code
must call `runtime.close()` before tearing down the base process groups. Ordinary
CP singleton/full-world groups are borrowed and are not destroyed by this method.

## Four-GPU replay

From the repository root with the existing environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m areal.tools.benchmark_adaptive_tree_workload \
  --synthetic-smoke --distributed-optimizer --checkpoint \
  --output /tmp/adaptive-tree-smoke.json

# Replay previously generated, checksummed workload batches in temporal order:
# Replace the three file paths with the actual trace files and choose budgets.
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m areal.tools.benchmark_adaptive_tree_workload \
  --workloads short.json.gz medium.json.gz long.json.gz \
  --local-token-budget 2048 --max-tree-tokens 8192 \
  --min-dwell-steps 3 --distributed-optimizer \
  --output /tmp/adaptive-tree-trace.json
```

The smoke trace is synthetic growth followed by shrinkage. The runner uses a small
random GPT and actual MCore AdamW, includes CPU planning/scatter in timing, and
does not exclude compilation. It is not a speedup benchmark or full RL run.

## Verification

```bash
.venv/bin/python -m pytest tests/test_adaptive_tree_cp.py -m 'not slow' -q
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_adaptive_tree_cp.py -vv -x
```

The numerical oracle explicitly requests CP1->2->4->CP1 with one persistent
model and optimizer, comparing to fixed-CP1 execution of the identical global
trajectories. Both normalization modes run. It checks loss, gradients, model
parameters, Adam master parameters/moments/step counts, and topology restoration.
Distributed Adam tests compare only locally owned, reduced gradient slices and
local optimizer states, plus all gathered model parameters. BF16 fixtures use
bias-free projections and RMSNorm; FP32 retains biases and LayerNorm.
Activation-recompute coverage checks
that backward actually re-enters the layer.

BF16 has different accumulation orders across packing/CP layouts. Tests use
explicit normwise bounds (6% for individual gradient/moment tensors, 1% for
parameters, plus a 2e-6 absolute floor), and stricter FP32 elementwise bounds.
The initial BF16 fixtures with zero-initialized QKV or LayerNorm biases failed
the parameter bound under different packing: Adam amplifies near-zero bias
gradient differences. Those cases are not certified by the bias-free tests.
The replay's default model has biases and LayerNorm; its finite-loss smoke check
does not establish the numerical equivalence tested by the bias-free oracle.

These fixtures bypass pretrained loading and rollout infrastructure. They do not
establish model convergence, checkpoint restart correctness, multi-node scaling,
full RLTrainer integration, or performance improvement. See the
[validation record](adaptive_tree_parallelism_validation.md) for the commands
and outcomes actually observed.
