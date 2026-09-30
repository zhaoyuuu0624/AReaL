# SPDX-License-Identifier: Apache-2.0
"""Run static DP x CP jobs sequentially within an existing single-node allocation."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("tree", "sequence"),
        default=["tree", "sequence"],
    )
    parser.add_argument("--cp-values", nargs="+", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds per configuration; fail fast on error/timeout",
    )
    args = parser.parse_args()
    import torch
    import transformer_engine.pytorch  # noqa: F401

    if torch.cuda.device_count() < args.gpus:
        parser.error("Fewer visible GPUs than requested")
    if not torch.cuda.is_bf16_supported():
        parser.error("BF16 GPU support required")
    cps = args.cp_values or [c for c in (1, 2, 4, 8) if c <= args.gpus]
    if any(args.gpus % c for c in cps):
        parser.error("Every CP size must divide GPU count")
    scenarios = (
        {"smoke": 128}
        if args.mode == "smoke"
        else {
            "short_low_reuse": 4096,
            "long_high_reuse": 16384,
            "nested": 16384,
            "mixed_lengths": 16384,
            "imbalanced": 16384,
        }
    )
    for name in scenarios:
        if not (args.data / f"{name}.json.gz").is_file():
            parser.error(f"Missing workload {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["AREAL_USE_TRITON_TREE_ATTN"] = "0"
    env["AREAL_FLEX_ATTENTION_BLOCK_SIZE"] = "128"
    summaries = []
    for name, cap in scenarios.items():
        for policy in args.policies:
            for cp in cps:
                stem = f"{name}_{policy}_dp{args.gpus // cp}_cp{cp}"
                result = args.output / f"{stem}.json"
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={args.gpus}",
                    "--module",
                    "areal.tools.benchmark_tree_workload",
                    "--workload",
                    str((args.data / f"{name}.json.gz").resolve()),
                    "--output",
                    str(result.resolve()),
                    "--cp",
                    str(cp),
                    "--cap",
                    str(cap),
                    "--policy",
                    policy,
                ]
                if args.mode == "smoke":
                    command += [
                        "--hidden",
                        "128",
                        "--layers",
                        "2",
                        "--warmup",
                        "1",
                        "--steps",
                        "2",
                    ]
                if args.checkpoint:
                    command.append("--checkpoint")
                (args.output / f"{stem}.command.json").write_text(
                    json.dumps(command, indent=2) + "\n"
                )
                with (args.output / f"{stem}.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = process.wait(timeout=args.timeout)
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                    except BaseException:
                        # Stop only this launch's process group, including its workers.
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                row = json.loads(result.read_text())
                summaries.append(
                    dict(
                        run=stem,
                        workload_sha256=row["packing"]["workload_sha256"],
                        mean_step_seconds=row["mean_step_seconds"],
                        duplicated_prefix_tokens=row["packing"][
                            "duplicated_prefix_tokens"
                        ],
                        max_peak_allocated_bytes=max(
                            r["peak_allocated_bytes"]
                            for s in row["steps"]
                            if not s["warmup"]
                            for r in s["ranks"]
                        ),
                    )
                )
                (args.output / "summary.json").write_text(
                    json.dumps(summaries, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
