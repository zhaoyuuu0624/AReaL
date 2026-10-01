# Adaptive tree DP x CP validation

## Provenance

- Date: 2026-10-01; host: gpu02; four NVIDIA RTX 5090 GPUs.
- Repository baseline: `f2d9d7d74a8de97951732716c20150bb550a9457`.
- Branch: `feat/tree-ulysses-cp`; this extension is an uncommitted working-tree change.
- Existing environment: Python 3.12, PyTorch 2.10.0, Megatron Core 0.19.0,
  mbridge 0.15.1. No dependency or installed Megatron source changes.
- Scope: dense packed-tree Ulysses, fixed TP=PP=EP=1 and fixed optimizer/DDP
  ownership, with uniform execution CP changes at optimizer-step boundaries.

Usage and implementation constraints are in
[adaptive_tree_parallelism.md](adaptive_tree_parallelism.md).

## New correctness suite

```bash
.venv/bin/python -m pytest tests/test_adaptive_tree_cp.py -q -x --tb=short
```

Observed: **15 passed in 123.01 seconds**, with no skips.

Ten CPU cases cover prefix-aware partitioning against an exhaustive contiguous
partition oracle, real-packer attention accounting, growth/shrinkage, dwell
control, infeasible overrides, state round trips, and invalid policies.

Five real four-GPU cases use a two-layer, hidden-size-128 random GPT and MCore
AdamW. Each runs both manual and per-token loss normalization:

| Precision | Full activation recomputation | Distributed optimizer | Result |
| --- | --- | --- | --- |
| FP32 | Off | Off | Passed |
| BF16 | Off | Off | Passed |
| BF16 | On | Off | Passed |
| FP32 | Off | On | Passed |
| BF16 | On | On | Passed |

Every case requests CP1 -> CP2 -> CP4 -> CP1 explicitly and keeps the model,
optimizer, and Adam history alive across those four steps. The reference uses
fixed CP1 and a different partition/packing of the same global trajectories.
Checks compare loss, gradients, model parameters, FP32 master parameters, Adam
moments and counters. Distributed-optimizer gradient comparisons cover the
locally owned reduced slices; gathered model parameters are compared in full.
Fixtures include duplicate and prefix-only trajectories, unequal loss weights,
zero-supervision rows, and entropy in the loss.

Additional checks reject different rank configurations collectively, recover
from invalid source batches, verify actual backward recomputation, restore CP
bindings, round-trip control state, and prohibit reuse after an injected
execution failure. The failure injection validates Python lifecycle handling;
it is not a distributed fault-recovery test.

FP32 uses elementwise rtol=3e-3 and atol=3e-5 for gradients/moments/loss,
and atol=3e-6 for parameters. BF16 uses per-tensor L2 bounds of 6% for
gradients/moments/loss and 1% for parameters, with a 2e-6 absolute floor.
All compared tensors must be finite.

**BF16 qualification:** the successful oracle uses bias-free linear projections
and RMSNorm. Initial BF16 fixtures with QKV biases or LayerNorm biases failed
parameter equivalence under different packing: near-zero gradient differences
were amplified by Adam. The successful FP32 fixture retains these biases.
Tolerances were not widened to hide the BF16 failures; equivalence for BF16
models with those biases remains unverified.

An initial distributed-optimizer test failed because the test read shard maps
directly from `ChainedOptimizer`. The corrected oracle unwraps the sole dense
optimizer before comparing shard ownership and ranges.

## Automatic workload replay

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m areal.tools.benchmark_adaptive_tree_workload \
  --synthetic-smoke --distributed-optimizer --checkpoint \
  --output /tmp/areal-adaptive-smoke-20261001.json
```

Observed: all four optimizer updates succeeded with finite losses. No CP override
was supplied. Default token budgets were 128 per rank and 512 per padded tree;
the smoke runner uses zero dwell steps to demonstrate both growth and shrinkage.

| Phase | Maximum trajectory tokens | Selected DP x CP | Padded tree cap | Synchronized microbatches |
| --- | --- | --- | --- | --- |
| Short | 64 | 4 x 1 | 128 | 2 |
| Growing | 180 | 1 x 4 | 512 | 6 |
| Long | 320 | 1 x 4 | 512 | 16 |
| Short again | 64 | 4 x 1 | 128 | 2 |

The heuristic chooses CP4 directly in the middle phase based on actual packing
and padded microbatch costs. CP2 is covered separately by the switching oracle.
This replay uses BF16, full recomputation and distributed AdamW. Its default
model has linear biases and LayerNorm; finite loss is a smoke check, not the
numerical-equivalence claim established for the bias-free BF16 fixture.

The JSON records versions, model configuration, control state, selected layout,
loss, step time and peak allocated GPU memory. Timings include planning, CPU
scatter, optimizer work and any compilation. This run provides **no steady-state
speedup or RL convergence claim**.

## Static-path regression and formatting

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_tree_cp.py tests/test_tree_workload.py \
  -q -x --tb=short
```

Observed: **50 passed in 460.94 seconds**, with no skips. This includes all
31 existing tree-CP cases and 19 workload/packing cases. Static CP2/CP4,
DP2 x CP2, multi-step updates, BF16 and full recomputation remain covered.

Ruff lint and format checks passed for all nine changed Python files.
`git diff --check` passed. The environment has no `mdformat` executable/module;
that documentation-formatting check was not run. No commit or push was made.

## Remaining validation boundaries

- Full RLTrainer/RPC/controller integration is not part of this engine-level API.
- The cost coefficients are heuristics, not a fitted latency/memory model;
  representative rollout traces and hardware calibration are still needed.
- Pretrained model initialization, end-to-end checkpoint save/restart,
  multi-node collectives, large-model convergence and performance are untested.
- Dense tree masks remain replicated; increasing CP does not eliminate their
  quadratic memory requirement.
- Only the documented dense FP32/BF16, zero-dropout configuration is covered;
  TP/PP/MoE/FP8 and heterogeneous CP within a step are outside this version.
