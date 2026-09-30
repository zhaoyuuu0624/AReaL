# Experimental packed-tree Ulysses CP

This branch adds **uniform, static CP** to main's packed-tree Megatron path,
not to the DTA executor. It does not implement adaptive DP/CP switching.

## Initial scope

- Megatron `mbridge`, standard MCore GPTModel/RoPE, dense text actor.
- TP=PP=1. Both Q and KV head counts must be divisible by CP.
- B=1 per packed tree; `pad_to_maximum: true`; the token capacity must satisfy
  the tree block and CP alignment constraints.
- FlexAttention; global tree visibility, contiguous CP token shards.
- No FP8, MLA, MoE, MTP, vision, or hybrid SSM models. TP/PP combinations are
  deliberately rejected until their numerical and schedule tests are added.
- Set effective attention dropout to zero; it is not implemented by this
  FlexAttention path. Keep other dropout disabled for numerical validation.
  No performance claim is
  made: the dense global tree mask is still replicated on every CP rank.

Example training-engine settings (not a complete rollout/trainer recipe):

```yaml
backend: megatron:d1p1t1c2
enable_tree_training: true
pad_to_maximum: true
disable_dropout: true
gradient_checkpointing: false
mb_spec:
  max_tokens_per_mb: 256
megatron:
  bridge_type: mbridge
```

## Data and gradient contract

Every CP rank must receive the **same packed tree**, tree depths, and original
trajectory loss metadata. Only model input tokens/positions are sliced. An
all-to-all changes `[N/CP, B, H, D]` to `[N, B, H/CP, D]`; each head sees the
global tree mask. The inverse communication restores local token ownership.

Loss reconstruction never gathers the full vocabulary logits. The owner of
each predictor token computes its log-probabilities and entropy. A
differentiable scalar SUM reconstructs the original trajectory order on each
CP rank. Shared physical predictions may appear in multiple loss occurrences
with different weights. The existing masked terminal slot is retained.

The scalar SUM's backward sums gradients from all replicated consumers. This
is the same contract as AReaL's ordinary CP differentiable gather. Retain
`MegatronEngine.train_batch`'s DP+CP denominator (manual mode) or its existing
per-token count partition. Do not add a second CP division. Parameter
gradients still need the backend's DP+CP reducer; attention all-to-all is not
a replacement for that reduction.

## Positional correctness

MCore 0.19's standard GPT RoPE does not use explicit tree-depth position IDs,
and its ordinary CP positional table uses a zigzag partition. The tree
forward adapter replaces the returned rotary tensor of this model instance's
preprocessor with depth-indexed, pre-CP frequencies. The selected tensor is
passed into checkpointed transformer blocks; no mutable per-microbatch state
is used by backward. The temporary method override is restored on exceptions.

This correction also applies to supported **CP=1, TP=PP=1** tree models. Therefore
their outputs can intentionally differ from the uncorrected upstream tree
path. Validate against independent original sequences, not only against the
old packed-tree result. Other CP=1 architectures and existing TP/PP combinations
retain their old route; this patch does not certify their positional behavior.

Padding/dummy queries are allowed to attend to themselves once in the forward
adapter. They retain zero supervised weight. The safe dense mask is shared
across layers, not copied once per attention layer.

## Tests

From the repository root with the dedicated environment active:

```bash
python -m pytest tests/test_tree_cp.py -m 'not slow' -q
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/test_tree_cp.py -k two_gpu -vv -x
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest tests/test_tree_cp.py -k four_gpu -vv -x
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest tests/test_tree_cp.py -k precision_checkpoint -vv -x
```

The torchrun wrapper uses a 600-second subprocess timeout (1200 for multistep).
It exports the repository path for child imports. Tests construct
small models locally; no pretrained checkpoint is downloaded. The SDPA and
FlexAttention variants compare Ulysses output, shared-prefix gradients, and
one SGD update against independent unshared sequences. They cover MHA/GQA,
cross-shard branches, prefix-only trajectories, an empty prediction owner,
and finite zero-loss backward. The MCore variant additionally exercises the
actual GPT/RoPE/core-attention adapter. The engine variant calls actual
`train_batch` with random GPT weights, tree packing, MCore DDP, the scheduler,
and an MCore FP32 SGD optimizer; it covers manual and per-token normalization
over two packed microbatches.

The four-GPU extension repeats the four levels at CP=4. Two additional engine
cases run 10 changing tree batches at DP=1/CP=4 and DP=2/CP=2, each in both
normalization modes. They include nested sharing, duplicate sequences,
prefix-only samples, and independent roots. DP replicas use unequal token
counts and different weights. Each rank's serial reference processes the full
global batch without gradient collectives; the engine's gradients are never
manually corrected. Loss, all parameter gradients, and post-SGD parameters are
checked after every step. See `tree_ulysses_cp_validation.md` for the actual
run result and tolerances.

Nine precision/checkpoint cases additionally cover DP=1/CP=2, DP=1/CP=4,
and DP=2/CP=2 in BF16, FP32 with checkpointing, and BF16 with checkpointing.
Each runs three changing batches in both normalization modes. BF16 uses the
actual MCore `Float16Module` and mixed-precision SGD with FP32 master weights;
the reference accumulates independent-sequence gradients in FP32. A separate
FP32 control checks mixed-precision numerical drift. Checkpointing uses full,
uniform recomputation with one layer per checkpoint and is compared against
the same-layout eager run. Layer-call counters assert that recomputation really
executes in backward, and that temporary RoPE overrides do not leak between steps.
This validates neither selective/block recomputation nor checkpoint memory savings.

These tests do **not** by themselves
certify the full RL trainer, all optimizer policies, or all checkpoint modes.

Before using this branch for RL experiments, also validate the actual
MegatronEngine loss/reducer/optimizer combination, forward-only evaluation,
multiple microbatches, and the chosen activation checkpoint policy. Inspect
the companion validation report for which commands were actually run.
