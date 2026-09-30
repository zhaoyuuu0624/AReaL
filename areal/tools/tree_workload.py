# SPDX-License-Identifier: Apache-2.0
"""Portable synthetic token-ID workloads; deliberately not real RL rollouts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def unique_work(records: list[dict]) -> tuple[int, int]:
    """Return trie nodes and allowed causal QK pairs (not hardware FLOPs)."""
    root: dict = {}
    nodes = pairs = 0
    for record in records:
        cursor = root
        for depth, token in enumerate(record["input_ids"], 1):
            if token not in cursor:
                cursor[token] = {}
                nodes += 1
                pairs += depth
            cursor = cursor[token]
    return nodes, pairs


def generate(name: str, seed: int = 42, vocab_size: int = 256) -> dict:
    if vocab_size < 64:
        raise ValueError("vocab_size must be at least 64")
    rng = random.Random(seed)

    def segment(length: int, first: int) -> list[int]:
        return [first] + [rng.randrange(32, vocab_size) for _ in range(length - 1)]

    records = []
    for tree in range(8):
        if name == "smoke":
            prefix, trunk, suffix, branches = 32, 8, 24, 2
        elif name == "short_low_reuse":
            prefix, trunk, suffix, branches = 128, 0, 1920, 2
        elif name == "long_high_reuse":
            prefix, trunk, suffix, branches = 6144, 0, 2048, 8
        elif name == "nested":
            prefix, trunk, suffix, branches = 2048, 2048, 2048, 8
        elif name == "mixed_lengths":
            length = (2048, 8192, 16384)[tree % 3]
            prefix, trunk, suffix, branches = length // 2, 0, length // 2, 4
        elif name == "imbalanced":
            prefix, trunk, suffix, branches = 4096, 0, 512 + tree * 256, tree + 1
        else:
            raise ValueError(f"Unknown scenario: {name}")
        shared = segment(prefix, tree + 1)
        trunks = [segment(trunk, 16 + i) for i in range(2)] if trunk else []
        for branch in range(branches):
            path = shared + (trunks[branch // ((branches + 1) // 2)] if trunk else [])
            tokens = path + segment(suffix, branch + 1)
            records.append(
                dict(
                    sequence_id=len(records),
                    tree_id=tree,
                    input_ids=tokens,
                    response_start=prefix,
                )
            )
    nodes, pairs = unique_work(records)
    payload = dict(
        format_version=1,
        source="synthetic_random_token_ids_not_rl_rollouts",
        scenario=name,
        seed=seed,
        vocab_size=vocab_size,
        records=records,
        stats=dict(
            original_tokens=sum(len(r["input_ids"]) for r in records),
            supervised_tokens=sum(
                len(r["input_ids"]) - r["response_start"] for r in records
            ),
            unique_tree_tokens=nodes,
            unique_allowed_attention_pairs=pairs,
            max_sequence_length=max(len(r["input_ids"]) for r in records),
        ),
    )
    payload["content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def load_workload(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        payload = json.load(stream)
    expected = payload.pop("content_sha256")
    if hashlib.sha256(canonical_bytes(payload)).hexdigest() != expected:
        raise ValueError("Workload content checksum mismatch")
    payload["content_sha256"] = expected
    if payload["format_version"] != 1 or not payload["records"]:
        raise ValueError("Unsupported or empty workload")
    for r in payload["records"]:
        if not 1 <= r["response_start"] < len(r["input_ids"]):
            raise ValueError("Invalid response_start")
        if any(not 0 <= t < payload["vocab_size"] for t in r["input_ids"]):
            raise ValueError("Token outside vocabulary")
    return payload


def partition(records: list[dict], dp_size: int, policy: str) -> list[list[dict]]:
    if dp_size < 1 or policy not in {"tree", "sequence"}:
        raise ValueError("Invalid DP size or assignment policy")
    groups: list[list[dict]] = [[] for _ in range(dp_size)]
    for r in records:
        key = r["tree_id"] if policy == "tree" else r["sequence_id"]
        groups[key % dp_size].append(r)
    if any(not group for group in groups):
        raise ValueError("Empty DP shard; reduce DP or generate a larger global batch")
    return groups


def make_batch(records: list[dict]) -> dict:
    import torch

    shape = (len(records), max(len(r["input_ids"]) for r in records))
    ids = torch.zeros(shape, dtype=torch.long)
    attention = torch.zeros(shape, dtype=torch.bool)
    loss = torch.zeros(shape, dtype=torch.bool)
    for i, record in enumerate(records):
        n = len(record["input_ids"])
        ids[i, :n] = torch.tensor(record["input_ids"], dtype=torch.long)
        attention[i, :n] = True
        # AReaL logprobs at position t predict input_ids[t+1].
        loss[i, record["response_start"] - 1 : n - 1] = True
    return dict(input_ids=ids, attention_mask=attention, loss_mask=loss)


def packing_report(payload: dict, dp_size: int, cap: int, policy: str) -> dict:
    """Use main's actual greedy packer, without materializing quadratic masks."""
    from areal.models.tree_attn.tree import _greedy_build_tries

    if cap <= 0 or cap % 128:
        raise ValueError("Packing cap must be a positive multiple of 128")
    shards = partition(payload["records"], dp_size, policy)
    counts = []
    per_rank = []
    for shard in shards:
        batch = make_batch(shard)
        tries, lengths = _greedy_build_tries(batch, cap)
        pairs = 0
        for trie in tries:
            subset = [shard[i] for i in trie.all_sequence_ids]
            nodes, allowed_pairs = unique_work(subset)
            if nodes != lengths[trie.tree_id]:
                raise AssertionError(
                    "Independent trie accounting disagrees with packer"
                )
            pairs += allowed_pairs
        counts.append(len(tries))
        per_rank.append(
            dict(
                real_microbatches=len(tries),
                packed_tokens=sum(lengths),
                allowed_attention_pairs=pairs,
            )
        )
    max_count = max(counts)
    packed = sum(r["packed_tokens"] for r in per_rank)
    for r in per_rank:
        r["dummy_microbatches"] = max_count - r["real_microbatches"]
        r["padded_tokens_including_dummy"] = max_count * cap
    return dict(
        dp_size=dp_size,
        policy=policy,
        packing_cap=cap,
        workload_sha256=payload["content_sha256"],
        global_batch_stats=payload["stats"],
        per_dp_rank=per_rank,
        executed_unique_tokens_before_padding=packed,
        duplicated_prefix_tokens=packed - payload["stats"]["unique_tree_tokens"],
        padded_tokens_including_dummy=max_count * cap * dp_size,
        note="Counts exclude CP replication; allowed attention pairs are semantic, not measured kernel FLOPs.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in (
        "smoke",
        "short_low_reuse",
        "long_high_reuse",
        "nested",
        "mixed_lengths",
        "imbalanced",
    ):
        payload = generate(name, args.seed)
        path = args.output / f"{name}.json.gz"
        if path.exists():
            raise FileExistsError(path)
        path.write_bytes(gzip.compress(canonical_bytes(payload), mtime=0))
        manifest.append({k: v for k, v in payload.items() if k != "records"})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
