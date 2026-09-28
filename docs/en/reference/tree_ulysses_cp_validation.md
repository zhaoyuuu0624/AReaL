# Tree Ulysses CP validation record

## Scope and provenance

- Base AReaL commit: `2fad2d0e308fe631e70e971b97188ad5c5cc03cb`.
- Remote development branch: `feat/tree-ulysses-cp`.
- Target: main packed-tree training, not DTA; static uniform CP with TP=PP=1.
- GPU host: gpu02, four RTX 5090 GPUs with 32 GB each; driver 580.126.20.
- Isolated Python 3.12 environment in the repository's `.venv`.
- Megatron Core 0.19.0; mbridge commit
  `310e8fb35ccf4fcd4419d32973e563a6d43ee5fb`.
- Megatron Bridge 0.6.0 uses the custom wheel URL pinned in `pyproject.toml`.
- Intended runtime: PyTorch 2.10.0 with CUDA 12.8, Transformer Engine 2.12.0.
  Package installation and runtime verification are still in progress.

No system driver/toolkit changes, model checkpoint downloads, commits, or pushes
are part of this change. CUDA compilation uses the existing CUDA 12.8 toolkit.

## Current verification status

- Python syntax compilation: passed on gpu02 for the updated implementation.
- Ruff lint/format and `git diff --check`: passed for the updated implementation;
  the lint and whitespace checks were also repeated on gpu02.
- Unit tests and GPU numerical tests: **not yet verified**. Do not interpret
  the existence of tests as evidence that they passed.

The current blocker is environment installation. Public-network direct access
from gpu02 times out. A temporary proxy allowed several dependencies to
install, but PyTorch/cuDNN downloads repeatedly slowed down or failed with TLS
and proxy connection errors. A mirror was tried with SHA-256 values obtained
from official PyPI and `--require-hashes`; the installation is still pending.
The task did not disable TLS or hash verification. Download status is logged
on gpu02 at `/tmp/areal-tree-cp-large-wheels.log`.

Read-only diagnostic imports with existing, non-target Conda environments
were unsuccessful (PyTorch 2.8 lacks AReaL's `DefaultStager`; the 2.13
environment is incompatible with the newly pinned torchvision). Those attempts
are **not** passing tests and did not modify the existing environments. The
Transformer Engine PyTorch extension must be built after the target PyTorch
runtime is ready.

## Reproducible test commands

Run from the repository root, using its dedicated environment:

```bash
.venv/bin/python -m pytest tests/test_tree_cp.py -m 'not slow' -q
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest tests/test_tree_cp.py -m slow -v
```

The four distributed variants are `sdpa`, `flex`, `mcore`, and `engine`.
Each uses real NCCL collectives on two GPUs. The final variant bypasses
pretrained loading only: actual AReaL tree packing, forward routing, loss
callbacks, MCore scheduling, DDP reduction, and FP32 SGD are exercised against
independent original sequences. It includes two packed microbatches and an odd
supervised-token count to exercise per-token CP remainders.

## Limits of the validation

Even successful tests above would not certify end-to-end asynchronous RL,
pretrained weight conversion, BF16, all activation checkpoint policies, DP>1,
TP/PP combinations, optimizer-state sharding, or adaptive DP/CP switching.
The dense global mask is still replicated. No throughput or memory-scaling
claim is made until those quantities are measured.
