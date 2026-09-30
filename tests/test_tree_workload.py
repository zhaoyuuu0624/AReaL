# SPDX-License-Identifier: Apache-2.0

import gzip

import pytest

from areal.tools.tree_workload import (
    canonical_bytes,
    generate,
    load_workload,
    make_batch,
    packing_report,
    partition,
    unique_work,
)


def test_tree_workload_deterministic_and_integrity(tmp_path):
    """Portable workload content is reproducible and protected by a checksum."""
    data = generate("smoke")
    assert data == generate("smoke")
    assert data["content_sha256"] != generate("smoke", seed=43)["content_sha256"]
    path = tmp_path / "smoke.json.gz"
    path.write_bytes(gzip.compress(canonical_bytes(data), mtime=0))
    assert load_workload(path) == data
    data["records"][0]["input_ids"][0] += 1
    path.write_bytes(gzip.compress(canonical_bytes(data), mtime=0))
    with pytest.raises(ValueError, match="checksum"):
        load_workload(path)


@pytest.mark.parametrize("dp", [1, 2, 4, 8])
@pytest.mark.parametrize("policy", ["tree", "sequence"])
def test_tree_partition_preserves_global_batch(dp, policy):
    """No dropped or duplicated training trajectories across DP ranks."""
    data = generate("smoke")
    shards = partition(data["records"], dp, policy)
    assert sorted(r["sequence_id"] for s in shards for r in s) == list(range(16))
    assert all(shards)


def test_tree_long_workload_has_exact_prefix_reuse():
    """Long scenario has 8 independent roots, each with 8 suffix branches."""
    data = generate("long_high_reuse")
    assert data["stats"]["unique_tree_tokens"] == 8 * (6144 + 8 * 2048)
    assert data["stats"]["original_tokens"] == 8 * 8 * 8192
    assert data["stats"]["supervised_tokens"] == 8 * 8 * 2048


def test_tree_packing_report_matches_main_packer_and_loss_alignment():
    """Compare accounting with real padded microbatches, including token masks."""
    from areal.api.cli_args import MicroBatchSpec
    from areal.models.tree_attn.tree import build_packed_tree_batch

    data = generate("smoke")
    report = packing_report(data, 1, 128, "tree")
    batch = make_batch(data["records"])
    actual = build_packed_tree_batch(batch, MicroBatchSpec(max_tokens_per_mb=128))
    assert sum(actual.group_lens) == report["executed_unique_tokens_before_padding"]
    assert len(actual.mbs) == report["per_dp_rank"][0]["real_microbatches"]
    assert batch["loss_mask"].sum().item() == data["stats"]["supervised_tokens"]
    assert batch["loss_mask"][0, 31] and not batch["loss_mask"][0, 30]
    assert not batch["loss_mask"][0, 63]
    split = packing_report(data, 8, 128, "sequence")
    together = packing_report(data, 8, 128, "tree")
    assert split["duplicated_prefix_tokens"] > together["duplicated_prefix_tokens"]


def test_unique_attention_pairs_count_shared_prefix_once():
    """Two length-three paths with a two-token prefix have four unique nodes."""
    records = [dict(input_ids=[1, 2, 3]), dict(input_ids=[1, 2, 4])]
    assert unique_work(records) == (4, 1 + 2 + 3 + 3)


@pytest.mark.parametrize(
    "name,count,max_length",
    [
        ("smoke", 16, 64),
        ("short_low_reuse", 16, 2048),
        ("long_high_reuse", 64, 8192),
        ("nested", 64, 6144),
        ("mixed_lengths", 32, 16384),
        ("imbalanced", 36, 6400),
    ],
)
def test_workload_scenarios_have_declared_shapes(name, count, max_length):
    """Every published scenario has the declared shape and eight distinct roots."""
    data = generate(name)
    assert len(data["records"]) == count
    assert data["stats"]["max_sequence_length"] == max_length
    assert len({r["input_ids"][0] for r in data["records"]}) == 8
    assert data["stats"]["supervised_tokens"] > 0


def test_tree_packing_cap_can_duplicate_prefix_within_one_dp_rank():
    """Splitting eight branches into two packed trees repeats the 6K prefix."""
    data = generate("long_high_reuse")
    report = packing_report(data, 1, 16384, "tree")
    assert report["duplicated_prefix_tokens"] == 8 * 6144
    assert report["per_dp_rank"][0]["real_microbatches"] == 16
