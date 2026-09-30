# Tree Ulysses CP validation record

## Scope and provenance

- Base AReaL commit: `2fad2d0e308fe631e70e971b97188ad5c5cc03cb`.
- Remote checkout at the four-GPU validation run:
  `b810c1eb206cf88d9ec9d1bd5583e468b21330f4`, with the test extensions uncommitted.
- Remote development branch: `feat/tree-ulysses-cp`.
- Target: main packed-tree training, not DTA; static uniform CP with TP=PP=1.
- GPU host: gpu02, four RTX 5090 GPUs with 32 GB each; driver 580.126.20.
- Isolated Python 3.12 environment in the repository's `.venv`.
- Megatron Core 0.19.0; mbridge commit
  `310e8fb35ccf4fcd4419d32973e563a6d43ee5fb`.
- Megatron Bridge 0.6.0 uses the custom wheel URL pinned in `pyproject.toml`.
- Runtime: PyTorch 2.10.0 with CUDA 12.8, Transformer Engine 2.12.0.
- Installed cuDNN at validation time: `nvidia-cudnn-cu12==9.10.2.21`
  (queried from the environment; different from the repository uv override).

No system driver/toolkit changes, model checkpoint downloads, commits, or pushes
are part of this change. CUDA compilation uses the existing CUDA 12.8 toolkit.

## Initial FP32 validation on 2026-09-28

- Python syntax compilation: passed on gpu02 for the updated implementation.
- Ruff lint/format and `git diff --check`: passed for the updated implementation;
  the lint and whitespace checks were also repeated on gpu02.
- Extended suite: **22 passed in 199.74 seconds**, no skips or failures, on gpu02.
  This is correctness-test runtime, not a training performance measurement.

| Tests | Configuration | Result |
| --- | --- | --- |
| 12 unit cases | Layout, prediction plan, unsupported combinations | Passed |
| 4 two-GPU cases | DP=1, CP=2; SDPA, FlexAttention, MCore, engine | Passed |
| 4 four-GPU cases | DP=1, CP=4; same four levels | Passed |
| 1 four-GPU multistep case | DP=1, CP=4; 10 steps in each normalization mode | Passed |
| 1 four-GPU multistep case | DP=2, CP=2; 10 steps in each normalization mode | Passed |

All model tests in this initial run use TP=PP=1, FP32, zero dropout, and no activation checkpointing.
The multistep tests keep the model and optimizer alive across steps. Each step
changes sequence contents and lengths using a private deterministic data RNG
(base seed 42, step offset 101, DP-rank offset 10007). The fixture contains a
two-level branching tree, a prefix-only sample, a duplicate on alternating
steps, and an unrelated root. This is a bounded randomized fixture, not exhaustive
coverage of arbitrary tree topologies or independent model initializations.

DP replicas have different data, loss coefficients, and supervised-token counts;
the counts are asserted unequal and non-divisible by CP. The reference on every
rank runs the original sequences of the **entire global batch** independently,
normalizes by one copy of the global supervised-token count, and uses serial SGD.
It does not use gradient collectives. Actual engine gradients are never manually
rescaled or reduced by the test. A DP-only reduction of a detached diagnostic
loss is used solely for the loss assertion.

Every step compares loss, all parameter gradients, and post-update parameters.
Both `calculate_per_token_loss=False` and `True` are exercised. Existing
thresholds are retained: engine loss rtol/atol 3e-4/3e-5, gradients 2e-3/2e-5,
updated parameters 2e-4/2e-6. Raw Ulysses layout/inverse/gradient tests require
exact equality. Passing these checks establishes numerical agreement for the
tested fixtures, not a proof for all inputs.

## BF16 and activation-checkpoint test extension

The initial targeted run on gpu02 completed with **9 passed, 22 deselected
in 308.34 seconds**. No selected test was skipped or failed.

The final full regression on 2026-09-28 completed with **31 passed in 455.74
seconds**, with no skips or failures. Remote JUnit report:
`/tmp/areal-tree-cp-precision.15Lyqd/results.xml` (temporary storage).
The tested script SHA-256 values are:

- `tests/test_tree_cp.py`: `f3bfd3033f7c4601917d655d44134ce61b82b5353cd37838618122bf69f1c88b`
- `tests/torchrun/run_tree_cp.py`: `18b16285d7fec9ecd978bf198945550e23ea437d20c2a2125107db356443c5ae`

Only tests and documentation changed for this extension; no training-backend
implementation or dependency changes were needed. Final Ruff checks and
`git diff --check` also passed.

Nine additional pytest cases cross three configurations (DP=1/CP=2,
DP=1/CP=4, DP=2/CP=2) with three modes (BF16 without checkpointing,
FP32 with checkpointing, BF16 with checkpointing). Each case uses three
changing tree batches and both manual and per-token loss normalization.
The checkpoint policy is specifically `recompute_granularity=full`,
`recompute_method=uniform`, `recompute_num_layers=1`, matching AReaL's defaults.
Selective and block policies are not covered.

BF16 is genuine mixed-precision execution: MCore `Float16Module` converts
model parameters, the first transformer layer's output dtype is asserted BF16,
DDP accumulates gradients in FP32, and MCore's BF16 optimizer maintains FP32
master parameters. The independent serial reference also uses BF16 compute,
but accumulates each sequence's gradient into separate FP32 buffers and updates
FP32 master parameters with plain SGD before copying back to BF16. It does not
share the engine's gradient normalization or optimizer implementation.

The following checks supplement the original FP32 elementwise assertions:

- BF16 vs independent BF16 sequences: loss rtol/atol 3e-3/3e-5; per-parameter
  gradient L2 error <= 5% of its reference norm + 1e-6, and whole-model gradient
  L2 error <= 3% + the same absolute floor. These normwise bounds account for
  different BF16 accumulation orders under prefix reuse; they were specified
  before execution, not widened after a failure.
- All gradients and model parameters must be finite. Each actual FP32 master
  update must equal `-0.01 * main_grad` (rtol/atol 2e-4/1e-7), and every BF16
  model parameter must equal its rounded FP32 master exactly. Reference master
  weights are additionally compared with rtol/atol 5e-3/2e-4.
- A separately initialized FP32 control starts from the same pre-quantization
  weights and processes the same batches. BF16 vs FP32 loss rtol/atol is
  5e-3/3e-5, and the full gradient-vector relative L2 bound is 8%. This is a
  numerical sanity bound for the small fixture, not a convergence guarantee.
- Checkpoint vs eager uses **the same dtype and DP/CP layout**, starting from
  the same seed and replaying all batches. Loss, all gradients, FP32 masters,
  and model parameters use rtol/atol **1e-5/1e-6**, including in BF16 mode;
  checkpoint comparisons do not inherit the looser mixed-precision bounds.
- A hook on the first transformer layer counts grad-enabled and no-grad calls.
  Full checkpointing must execute one no-grad original forward and one
  grad-enabled recomputation per microbatch; eager execution must have only
  one grad-enabled call. The temporary tree RoPE override must be removed
  after `train_batch`, so backward cannot depend on a stale forward override.

The precision tests initialize small models locally and exercise the real engine
training path, but still bypass pretrained loading and normal bridge initialization.
They do not establish checkpoint memory savings or real-model convergence.

## Reproducible test commands

Run from the repository root, using its dedicated environment:

```bash
.venv/bin/python -m pytest tests/test_tree_cp.py -m 'not slow' -q
# Original two-GPU regression only:
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest tests/test_tree_cp.py -k two_gpu -vv -x
# New four-GPU cases only (choose idle devices):
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m pytest tests/test_tree_cp.py -k four_gpu -vv -x
# BF16/checkpoint matrix only (nine cases, some using two of the four visible GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m pytest tests/test_tree_cp.py -k precision_checkpoint -vv -x
# Full suite, now including the precision/checkpoint cases:
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_tree_cp.py -vv -x -rA
```

The pytest wrapper exports the repository in subprocess `PYTHONPATH`, selects
two or four torchrun workers, and skips explicitly if too few GPUs are visible.
It uses 600-second subprocess timeouts for single-step tests and 1200 seconds
for multistep tests. A skip is not a successful numerical validation.

For detailed per-step error output, run the underlying script directly:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/torchrun/run_tree_cp.py --backend engine --cp-size 2 --random-trees --steps 10 --seed 42
```

The engine tests construct a small random GPT model and inject it into the real
engine; no pretrained model is downloaded. Actual tree packing, forward routing,
loss callbacks, MCore scheduling, DDP reduction, and MCore FP32 SGD are exercised.
The BF16 extension also exercises MCore's mixed-precision SGD wrapper.
Pretrained loading, normal engine initialization, and rollout infrastructure
are outside this fixture.

## Installation notes

Initial downloads and TE compilation were blocked; those historical failures
are superseded by the successful suite above. AReaL's uv overrides exclude the
TE metapackage and pin cuDNN to 9.19.0.56. The TE installation required explicit
packages with `uv --no-config`, and the installed cuDNN/NCCL include directories
on the compiler search path. This validation did not rebuild system CUDA/drivers
or modify unrelated Conda environments. Refer to the conversation for the exact
installation diagnostics; this report records the tested runtime, not a claim
that all package metadata constraints of the full AReaL stack are satisfied.

## Limits of the validation

The successful tests above do not certify end-to-end asynchronous RL,
pretrained weight conversion, arbitrary BF16 workloads, other activation checkpoint policies, DP>2,
TP/PP combinations, Adam/AdamW, optimizer-state sharding, or adaptive DP/CP switching.
The dense global mask is still replicated. No throughput or memory-scaling
claim is made until those quantities are measured.
