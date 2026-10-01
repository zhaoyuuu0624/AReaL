# SPDX-License-Identifier: Apache-2.0
"""Replay a changing global workload through the adaptive tree engine API.

Random small GPT + real MCore AdamW; no rollout service or pretrained downloads.
All timings include planning/scatter and may include JIT compilation. This is
an execution smoke/replay tool, not an end-to-end RL performance claim.
"""

import argparse
import importlib.metadata
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from areal.models.tree_attn.adaptive import AdaptiveTreeConfig
from areal.tools.benchmark_tree_workload import build_engine
from areal.tools.tree_workload import generate, load_workload, make_batch


def smoke_trace() -> list[dict]:
    """Four synthetic phases; the final phase verifies reversible adaptation."""
    trace = []
    for extension in (0, 116, 256, 0):
        payload = generate("smoke")
        for record in payload["records"]:
            record["input_ids"] += [32 + record["sequence_id"]] * extension
        # Modified synthetic data is not the checksummed on-disk workload.
        payload.pop("content_sha256")
        payload.pop("stats")
        payload["scenario"] = f"synthetic_growth_extension_{extension}"
        trace.append(payload)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic-smoke", action="store_true")
    source.add_argument("--workloads", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cp-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--local-token-budget", type=int, default=128)
    parser.add_argument("--max-tree-tokens", type=int, default=512)
    parser.add_argument("--min-dwell-steps", type=int, default=0)
    parser.add_argument("--min-relative-gain", type=float, default=0.1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--distributed-optimizer", action="store_true")
    parser.add_argument("--per-token-loss", action="store_true")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=7)
    args = parser.parse_args()
    if args.hidden < 16 or args.hidden % 16 or args.layers < 1 or args.repeat < 1:
        parser.error(
            "hidden must be a positive multiple of 16; layers/repeat must be positive"
        )
    if (
        os.environ.get("AREAL_USE_TRITON_TREE_ATTN", "0") != "0"
        or os.environ.get("AREAL_FLEX_ATTENTION_BLOCK_SIZE", "128") != "128"
    ):
        parser.error("Adaptive replay requires FlexAttention with block size 128")
    config = AdaptiveTreeConfig(
        tuple(args.cp_sizes),
        args.local_token_budget,
        args.max_tree_tokens,
        args.min_dwell_steps,
        args.min_relative_gain,
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(context_parallel_size=1)
    control = dist.new_group(backend="gloo")
    runtime = None
    try:
        trace, envelope = None, [None]
        if dist.get_rank() == 0:
            try:
                if args.output.exists():
                    raise FileExistsError(args.output)
                trace = (
                    smoke_trace()
                    if args.synthetic_smoke
                    else [load_workload(p) for p in args.workloads]
                )
                vocab_sizes = {p["vocab_size"] for p in trace}
                if len(vocab_sizes) != 1:
                    raise ValueError("All workload phases must use the same vocabulary")
                envelope[0] = (len(trace), vocab_sizes.pop(), None)
            except Exception as exc:
                envelope[0] = (None, None, str(exc))
        dist.broadcast_object_list(envelope, src=0, group=control)
        count, vocab_size, error = envelope[0]
        if error is not None:
            raise ValueError(error)
        args.cp, args.cap = 1, args.max_tree_tokens
        engine, model = build_engine(args, {"vocab_size": vocab_size}, device)
        runtime = engine.configure_adaptive_tree_parallelism(config)
        rows = []
        for step in range(count * args.repeat):
            batch = (
                make_batch(trace[step % count]["records"])
                if dist.get_rank() == 0
                else None
            )
            numerator = torch.zeros((), dtype=torch.float32, device=device)
            weight = torch.zeros((), dtype=torch.int64, device=device)

            def loss_fn(logprobs, entropy, inputs, **kwargs):
                value = (-logprobs * inputs["loss_mask"]).sum()
                numerator.add_(value.detach())
                weight.add_(inputs["loss_mask"].sum())
                return value / inputs["loss_mask"].sum().clamp_min(1)

            torch.cuda.reset_peak_memory_stats(device)
            stats = engine.train_adaptive_tree_batch(
                batch, loss_fn, lambda x: x["loss_mask"].sum()
            )
            dist.all_reduce(numerator, group=runtime.world_group)
            dist.all_reduce(weight, group=runtime.world_group)
            if not torch.isfinite(numerator):
                raise FloatingPointError("Non-finite replay loss")
            memory = torch.tensor(
                torch.cuda.max_memory_allocated(device),
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(memory, op=dist.ReduceOp.MAX, group=runtime.world_group)
            if dist.get_rank() == 0:
                rows.append(
                    dict(
                        step=step,
                        scenario=trace[step % count]["scenario"],
                        loss=float(numerator / weight.clamp_min(1)),
                        global_supervised_tokens=int(weight)
                        // int(stats["adaptive_cp_size"]),
                        peak_allocated_bytes=int(memory),
                        **stats,
                    )
                )
                engine.logger.info(
                    "Adaptive replay step=%s DP=%s CP=%s seconds=%.4f",
                    step,
                    stats["adaptive_dp_size"],
                    stats["adaptive_cp_size"],
                    stats["adaptive_step_seconds"],
                )
        if dist.get_rank() == 0:
            result = dict(
                kind="synthetic_random_gpt_adaptive_tree_replay_not_full_rl",
                timing_note="Includes CPU planning/scatter, train_batch, optimizer and synchronization; compilation is not excluded. Do not infer steady-state speedup from this smoke trace.",
                model=dict(
                    layers=args.layers,
                    hidden=args.hidden,
                    precision=args.precision,
                    checkpoint=args.checkpoint,
                    per_token_loss=args.per_token_loss,
                    distributed_optimizer=args.distributed_optimizer,
                    add_bias_linear=True,
                    normalization="LayerNorm",
                ),
                versions={
                    name: importlib.metadata.version(name)
                    for name in ("torch", "megatron-core", "mbridge")
                },
                gpu=torch.cuda.get_device_name(device),
                world_size=dist.get_world_size(),
                control_state=runtime.state_dict(),
                steps=rows,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if runtime is not None:
            runtime.close()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
