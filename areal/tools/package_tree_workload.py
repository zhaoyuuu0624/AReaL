# SPDX-License-Identifier: Apache-2.0
"""Build a source/data-only replay archive from an explicit file allowlist."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists():
        raise FileExistsError(args.output)
    # Tracked package sources only; never recurse into .venv/.git/user outputs.
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "areal"], cwd=root)
        .decode()
        .split("\0")
    )
    names = {n for n in tracked if n and (root / n).is_file()}
    names.update(
        {
            "areal/tools/tree_workload.py",
            "areal/tools/benchmark_tree_workload.py",
            "areal/tools/run_tree_workload_matrix.py",
            "areal/tools/package_tree_workload.py",
            "tests/test_tree_workload.py",
            "tests/test_tree_cp.py",
            "tests/torchrun/run_tree_cp.py",
            "tests/__init__.py",
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "docs/en/reference/tree_workload_replay.md",
            "docs/en/reference/tree_ulysses_cp.md",
            "docs/en/reference/tree_ulysses_cp_validation.md",
        }
    )
    files = {}
    for name in sorted(names):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Refusing missing or symlinked source: {name}")
        files[name] = path.read_bytes()
    for name in (
        "smoke",
        "short_low_reuse",
        "long_high_reuse",
        "nested",
        "mixed_lengths",
        "imbalanced",
    ):
        path = args.data / f"{name}.json.gz"
        from areal.tools.tree_workload import load_workload

        load_workload(path)
        files[f"workloads/{path.name}"] = path.read_bytes()
    files["workloads/manifest.json"] = (args.data / "manifest.json").read_bytes()
    files["README_REPLAY.md"] = files["docs/en/reference/tree_workload_replay.md"]
    packages = {}
    for name in (
        "torch",
        "megatron-core",
        "transformer-engine",
        "transformer-engine-cu12",
        "transformer-engine-torch",
        "mbridge",
        "megatron-bridge",
        "triton",
        "numpy",
        "transformers",
        "nvidia-cudnn-cu12",
        "nvidia-nccl-cu12",
        "pytest",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = dict(
        python=platform.python_version(),
        packages=packages,
        source_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        source_note="Archive includes current modified source contents; file hashes identify the actual artifact, not HEAD alone. No virtualenv, dependency wheels, models or GPU caches are included.",
    )
    files["ENVIRONMENT.json"] = (json.dumps(environment, indent=2) + "\n").encode()
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(files.items())
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids replacing any previously prepared user bundle.
    with (
        args.output.open("xb") as target,
        tarfile.open(fileobj=target, mode="w:gz") as archive,
    ):
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(f"tree-cp-workload/{name}")
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))


if __name__ == "__main__":
    main()
