# Offline tree workload replay for a single-node GPU allocation

This bundle supplies **synthetic workloads**, not collected RL trajectories. It uses
AReaL main's greedy tree packing and the experimental tree-aware Ulysses path.
No rollout server, model download, pretrained checkpoint, or external dataset is needed.
The environment must already contain the matching AReaL/Megatron dependencies.

## Scope

- Fixed global batch of original token-ID sequences, unchanged across DP x CP.
- TP=PP=1. Each separate process launch uses one uniform, static CP size.
- Eight Q heads and eight KV heads permit CP=1/2/4/8. No GQA replication trick.
- Random small GPT, LayerNorm, RoPE, zero dropout, BF16, actual MCore AdamW.
- Default performance model: 4 layers, hidden size 512, vocabulary 256.
  Smoke model: 2 layers, hidden size 128. Neither is Qwen3 or a pretrained LLM.
- Response-only next-token NLL. No fabricated PPO old logprobs or advantages.
- Optional full/uniform activation checkpointing, one layer per checkpoint.
- No PP/TP validation, online DP x CP switching, auto-parallel search, real RL
  convergence, or real-model throughput claims are implied by these runs.

## Workloads

Each file is one global batch with eight distinct root prefixes. Token values and
branch structure are fixed by a seed and protected by a content SHA256. Repeated
steps intentionally replay the same batch; this is not a workload-growth trace.

| Scenario | Structure per root | Sequences in global batch |
| --- | --- | --- |
| smoke | prefix 32, two distinct trunks of 8, suffix 24 | 16 |
| short_low_reuse | prefix 128, 2 suffixes of 1,920 | 16 |
| long_high_reuse | prefix 6,144, 8 suffixes of 2,048 | 64 |
| nested | prefix 2,048, two trunks of 2,048, 4 suffixes of 2,048 per trunk | 64 |
| mixed_lengths | full lengths 2K/8K/16K, half shared prefix, 4 branches | 32 |
| imbalanced | prefix 4,096, 1 to 8 branches, suffix length 512 to 2,304 | 36 |

`tree` assignment keeps each original root's sequences on one DP rank (round robin
by tree ID). `sequence` assigns individual trajectories round robin, potentially
duplicating prefixes across DP ranks. These are two **explicit heuristic baselines**,
not the same fixed partition and not an optimized tree partitioner. Within each
DP rank, the unchanged main packer can split a tree further under the token cap.
Changing DP therefore changes packing; compare policies separately and inspect the
reported duplication. No original trajectory is split or silently truncated.

## Run after extracting the bundle

Run from the extracted directory containing `areal/` and use the Python executable
from the RJob image/environment. The commands below do not install anything.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/test_tree_workload.py -q

# First: eight GPUs, eight small runs (four CP sizes x two assignment policies).
python -m areal.tools.run_tree_workload_matrix \
  --data workloads --output results-smoke --gpus 8 --mode smoke

# Only after smoke succeeds: forty larger runs, sequentially, no GPU oversubscription.
python -m areal.tools.run_tree_workload_matrix \
  --data workloads --output results-full --gpus 8 --mode full

# Optional separate checkpoint comparison. Use a fresh output directory.
python -m areal.tools.run_tree_workload_matrix \
  --data workloads --output results-smoke-checkpoint --gpus 8 --mode smoke --checkpoint
```

For a shorter initial job add `--policies tree` or `--cp-values 1 2`.
The matrix runner defaults to a one-hour timeout **per configuration** and stops on
the first failure; it does not submit RJobs or infer scheduler-specific variables.
It assumes all GPUs are on one node, with exclusive GPU allocation, and sufficient
host RAM for the current dense-mask packing implementation. Larger scenarios can
need substantial host memory; use an allocation with at least 64 GiB host RAM as an
initial request and monitor the actual peak. This is not a proven bound.

To regenerate identical data in a fresh directory:

```bash
python -m areal.tools.tree_workload --output workloads-new --seed 42
```

## Measurement and correctness boundaries

Each JSON contains workload checksum, original/supervised/unique tokens, actual
packed tokens, duplicated prefix tokens, dummy microbatches, padding, semantic
allowed QK-pair counts, per-rank memory/time/loss, GPU and package versions.
Semantic pair counts are **not measured kernel FLOPs**: block sparsity, padding,
and implementation overhead affect actual work.

Time covers `engine.train_batch`: CPU packing, host/device transfers, forward,
backward, gradient communication, optimizer, and the final CUDA synchronization.
Start barriers and metric collection are excluded. Report the slowest rank's time.
Warmup steps update weights identically in count across configurations and are
recorded but excluded from the mean. Three warmup steps do not guarantee all JIT
compilation is amortized; inspect individual step timings and rerun with a larger
`--warmup` using the individual command saved in `*.command.json` if needed.
Peak reserved memory can include compilation caches. Dense masks are replicated
across CP ranks; CP does not shard model or AdamW state in this implementation.

There is no serial gradient oracle in this performance runner. Finite loss/weights,
token invariants, and successful execution are smoke checks, **not a proof of
numerical equivalence**. Keep the separate `tests/test_tree_cp.py` correctness suite.
Eight-GPU and H200 results must be recorded after running on that target; four-GPU
RTX 5090 validation cannot establish H200/CP8 correctness or performance.

## Packaging and environment

The archive includes source, tests, these instructions, generated data and a file
checksum manifest. It excludes `.venv`, Git history, CUDA caches, user credentials
and model weights. `ENVIRONMENT.json` describes the gpu02 package versions as a
reference, not a portable environment or a dependency lock for H200. Use an H200
compatible image; do not copy SM120-only compiled extensions from RTX 5090.

An actual RJob submission still needs the user's image, single-node eight-GPU
resource specification, upload/mount path, and entrypoint syntax. The commands
above are the entrypoint payload, not a cluster-specific submission template.
