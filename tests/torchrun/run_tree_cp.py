# SPDX-License-Identifier: Apache-2.0

"""Small, deterministic two-GPU tests; no downloads or pretrained model needed."""

import argparse
import copy
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from areal.models.tree_attn.cp_functional import (
    TreePredictionPlan,
    gather_tree_cp_scalars,
)
from areal.models.tree_attn.tree import TrieNode
from areal.models.tree_attn.ulysses import (
    TreeCPLayout,
    head_to_sequence,
    sequence_to_head,
    tree_ulysses_attention,
)


def make_tree(short: bool, device: torch.device):
    """Include a prefix-only trajectory and branches crossing the CP boundary."""
    a, b, c = (2, 2, 2) if short else (100, 80, 60)
    n = 256
    root = TrieNode(0)
    root.nodes = [
        TrieNode(0, 0, a - 1, list(range(a)), [0, 1, 2]),
        TrieNode(0, a, a + b - 1, list(range(b)), [0]),
        TrieNode(0, a + b, a + b + c - 1, list(range(c)), [1]),
    ]
    paths = [
        list(range(a + b)),
        list(range(a)) + list(range(a + b, a + b + c)),
        list(range(a)),
    ]
    ids = (torch.arange(n, device=device) % 63 + 1).unsqueeze(0)
    positions = torch.zeros((1, n), dtype=torch.long, device=device)
    mask = torch.eye(n, dtype=torch.bool, device=device)
    for path in paths:
        idx = torch.tensor(path, device=device)
        positions[0, idx] = torch.arange(len(path), device=device)
        for j, q in enumerate(path):
            mask[q, idx[: j + 1]] = True
    return root, ids, positions, mask, paths


def apply_rope(x, positions):
    dim = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    freqs = positions.reshape(-1, 1).float() * inv
    angles = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    left, right = x.chunk(2, dim=-1)
    return x * angles.cos() + torch.cat((-right, left), dim=-1) * angles.sin()


class TinyTreeModel(nn.Module):
    def __init__(self, kv_heads: int):
        super().__init__()
        self.kv_heads = kv_heads
        self.head_dim = 16
        self.embedding = nn.Embedding(64, 32)
        self.q = nn.Linear(32, 8 * self.head_dim, bias=False)
        self.k = nn.Linear(32, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(32, kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(8 * self.head_dim, 32, bias=False)
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.head = nn.Linear(32, 64, bias=False)

    def forward(self, ids, positions, mask, cp_group=None, flex=False):
        x = self.embedding(ids).transpose(0, 1)
        q = apply_rope(self.q(x).reshape(-1, 1, 8, self.head_dim), positions)
        k = apply_rope(
            self.k(x).reshape(-1, 1, self.kv_heads, self.head_dim), positions
        )
        v = self.v(x).reshape(-1, 1, self.kv_heads, self.head_dim)
        if flex:
            from areal.models.tree_attn.module_fsdp import (
                _flex_attention,
                create_block_mask_from_dense,
            )

            block = create_block_mask_from_dense(mask, mask.shape[0], mask.device)

            def attend(q, k, v):
                return _flex_attention(
                    q, k, v, block_mask=block, enable_gqa=q.shape[1] != k.shape[1]
                )
        else:

            def attend(q, k, v):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, enable_gqa=q.shape[1] != k.shape[1]
                )

        if cp_group is not None:
            y = tree_ulysses_attention(q, k, v, attend, cp_group)
        else:
            y = attend(
                q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3)
            ).permute(2, 0, 1, 3)
        x = x + self.proj(y.reshape(x.shape[0], 1, -1))
        x = x + self.mlp(x)
        return self.head(x).squeeze(1)


def sequence_loss(logits, ids, sid):
    # Distinct trajectory weights exercise shared-prefix gradient accumulation.
    logp = F.log_softmax(logits.float(), -1)
    lp = logp[:-1].gather(1, ids[1:, None]).squeeze(1)
    ent = -(logp[:-1].exp() * logp[:-1]).sum(-1)
    weights = torch.linspace(0.3, 1.3, lp.numel(), device=lp.device) * (sid + 1)
    return ((-lp + 0.02 * ent) * weights).sum()


def compare_models(reference, parallel, name, rtol=3e-4, atol=3e-5):
    errors = []
    for (n, p), (n2, q) in zip(
        reference.named_parameters(), parallel.named_parameters()
    ):
        assert n == n2
        # Replicated full loss used loss/CP below; Ulysses parameter contributions
        # are summed across CP (attention A2A itself never does this reduction).
        dist.all_reduce(q.grad, group=dist.group.WORLD)
        torch.testing.assert_close(
            q.grad, p.grad, rtol=rtol, atol=atol, msg=f"{name}: {n}"
        )
        errors.append((q.grad - p.grad).abs().max().item())
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    torch.optim.SGD(parallel.parameters(), lr=0.01).step()
    for p, q in zip(reference.parameters(), parallel.parameters()):
        torch.testing.assert_close(q, p, rtol=rtol, atol=atol)
    if dist.get_rank() == 0:
        print(f"PASS {name}: max_gradient_abs_error={max(errors):.6g}", flush=True)


def test_layout(device):
    rank, size = dist.get_rank(), dist.get_world_size()
    x = (
        torch.arange(4 * 1 * 8 * 2, device=device, dtype=torch.float64).reshape(
            4, 1, 8, 2
        )
        + rank * 1000
    ).requires_grad_()
    y = sequence_to_head(x, dist.group.WORLD)
    expected = torch.cat(
        [
            (
                torch.arange(x.numel(), device=device, dtype=x.dtype).reshape(x.shape)
                + r * 1000
            )[:, :, rank * (8 // size) : (rank + 1) * (8 // size)]
            for r in range(size)
        ]
    )
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    z = head_to_sequence(y, dist.group.WORLD)
    torch.testing.assert_close(z, x, rtol=0, atol=0)
    upstream = torch.randn_like(x)
    (z * upstream).sum().backward()
    torch.testing.assert_close(x.grad, upstream, rtol=0, atol=0)


def test_tiny(device, short, kv_heads, flex):
    torch.manual_seed(42)
    root, ids, pos, mask, paths = make_tree(short, device)
    plan = TreePredictionPlan.from_trie(root)
    reference = TinyTreeModel(kv_heads).to(device)
    parallel = copy.deepcopy(reference)
    ref_loss = torch.zeros((), device=device)
    expected = {}
    for sid, path in enumerate(paths):
        idx = torch.tensor(path, device=device)
        seq = ids[0, idx]
        causal = torch.ones(
            (len(path), len(path)), dtype=torch.bool, device=device
        ).tril()
        logits = reference(
            seq[None], torch.arange(len(path), device=device)[None], causal
        )
        expected[sid] = (
            F.log_softmax(logits.float(), -1)[:-1].gather(1, seq[1:, None]).squeeze(1)
        )
        ref_loss = ref_loss + sequence_loss(logits, seq, sid)
    ref_loss.backward()
    layout = TreeCPLayout(256, dist.get_world_size(), dist.get_rank())
    logits = parallel(
        layout.slice(ids, -1), layout.slice(pos, -1), mask, dist.group.WORLD, flex
    )
    lp, ent, _, _ = gather_tree_cp_scalars(
        logits, ids, plan, dist.group.WORLD, chunk_size=37
    )
    loss = torch.zeros((), device=device)
    for sid, path in enumerate(paths):
        torch.testing.assert_close(lp[sid][:-1], expected[sid], rtol=3e-4, atol=3e-5)
        weights = torch.linspace(0.3, 1.3, len(path) - 1, device=device) * (sid + 1)
        loss = loss + ((-lp[sid][:-1] + 0.02 * ent[sid][:-1]) * weights).sum()
    torch.testing.assert_close(loss, ref_loss, rtol=3e-4, atol=3e-5)
    (loss / dist.get_world_size()).backward()
    compare_models(
        reference, parallel, f"tiny short={short} kv_heads={kv_heads} flex={flex}"
    )
    # Dummy/no-loss backward must remain finite and execute all A2As.
    parallel.zero_grad(set_to_none=True)
    dummy_mask = torch.eye(256, dtype=torch.bool, device=device)
    dummy_logits = parallel(
        layout.slice(ids, -1), layout.slice(pos, -1), dummy_mask, dist.group.WORLD, flex
    )
    (dummy_logits.float().mean() * 0).backward()
    for p in parallel.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_mcore(device, engine_mode=False, per_token_loss=False):
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from areal.engine.megatron_utils.tree_context_parallel import (
        tree_context_parallel_forward,
    )
    from areal.models.tree_attn.module_megatron import PytorchFlexAttention

    mpu.initialize_model_parallel(context_parallel_size=dist.get_world_size())
    try:
        torch.manual_seed(7)
        model_parallel_cuda_manual_seed(7)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=4,
            ffn_hidden_size=256,
            kv_channels=16,
            context_parallel_size=dist.get_world_size(),
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cpu_initialization=True,
            normalization="LayerNorm",
            masked_softmax_fusion=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            gradient_accumulation_fusion=False,
            calculate_per_token_loss=per_token_loss,
        )
        spec = get_gpt_layer_local_spec()
        spec.submodules.self_attention.submodules.core_attention = PytorchFlexAttention
        from megatron.core.transformer.enums import AttnMaskType

        spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.arbitrary
        model = GPTModel(
            config, spec, 64, 256, position_embedding_type="rope", parallel_output=False
        ).to(device)
        # Reference uses the same MCore module with a singleton CP group.
        singleton_groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
        ref_config = copy.deepcopy(config)
        ref_config.context_parallel_size = 1
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = copy.copy(pg)
        ref_pg.cp = singleton_groups[dist.get_rank()]
        ref_spec = get_gpt_layer_local_spec()
        ref_spec.submodules.self_attention.submodules.core_attention = (
            PytorchFlexAttention
        )
        ref_spec.submodules.self_attention.params["attn_mask_type"] = (
            AttnMaskType.arbitrary
        )
        reference = GPTModel(
            ref_config,
            ref_spec,
            64,
            256,
            position_embedding_type="rope",
            parallel_output=False,
            pg_collection=ref_pg,
        ).to(device)
        reference.load_state_dict(model.state_dict())
        root, ids, pos, mask, paths = make_tree(False, device)
        if engine_mode:
            test_engine_step(model, reference, config, ids, paths, device)
            return
        loss_ref = torch.zeros((), device=device)
        for sid, path in enumerate(paths):
            # Pad reference to FlexAttention's block granularity.
            seq = ids[:, path]
            padded = F.pad(seq, (0, 256 - seq.shape[1]))
            causal = torch.ones((256, 256), dtype=torch.bool, device=device).tril()
            logits_ref = reference(
                padded, torch.arange(256, device=device)[None], attention_mask=causal
            )[0, : seq.shape[1]]
            loss_ref = loss_ref + sequence_loss(logits_ref, seq[0], sid)
        loss_ref.backward()
        logits = tree_context_parallel_forward(
            model, {"input_ids": ids, "position_ids": pos, "attention_mask": mask}
        )
        lp, ent, _, _ = gather_tree_cp_scalars(
            logits,
            ids,
            TreePredictionPlan.from_trie(root),
            mpu.get_context_parallel_group(),
        )
        loss = sum(
            (
                (-lp[sid][:-1] + 0.02 * ent[sid][:-1])
                * torch.linspace(0.3, 1.3, len(path) - 1, device=device)
                * (sid + 1)
            ).sum()
            for sid, path in enumerate(paths)
        )
        torch.testing.assert_close(loss, loss_ref, rtol=3e-4, atol=3e-5)
        (loss / dist.get_world_size()).backward()
        compare_models(
            reference, model, "mcore-unpacked-vs-tree-cp", rtol=2e-3, atol=2e-4
        )
    finally:
        mpu.destroy_model_parallel()


def test_engine_step(model, reference, config, ids, paths, device):
    """Exercise actual packing, loss, MCore schedule, DDP and optimizer.

    Only pretrained loading and rollout infrastructure are bypassed. No manual
    CP division or extra parameter-gradient all-reduce is allowed here.
    """
    from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from areal.api import MegatronParallelStrategy
    from areal.api.cli_args import MicroBatchSpec, TrainEngineConfig
    from areal.engine.megatron_engine import MegatronEngine, _MegatronModelList
    from areal.utils import logging

    sequences = [ids[0, path] for path in paths]
    # Distinct root forces a second microbatch with only one prediction owner.
    # Total supervised count is odd, exercising per-token CP remainders.
    sequences.append(torch.zeros(41, dtype=torch.long, device=device))
    width = max(seq.numel() for seq in sequences)
    shape = (len(sequences), width)
    batch = {
        "input_ids": torch.zeros(shape, dtype=torch.long, device=device),
        "attention_mask": torch.zeros(shape, dtype=torch.bool, device=device),
        "loss_mask": torch.zeros(shape, dtype=torch.bool, device=device),
        "coefficients": torch.zeros(shape, dtype=torch.float32, device=device),
    }
    ref_numerator = torch.zeros((), device=device)
    for sid, seq in enumerate(sequences):
        length = seq.numel()
        batch["input_ids"][sid, :length] = seq
        batch["attention_mask"][sid, :length] = True
        batch["loss_mask"][sid, : length - 1] = True
        batch["coefficients"][sid, : length - 1] = torch.linspace(
            0.3, 1.3, length - 1, device=device
        ) * (sid + 1)
        padded = F.pad(seq[None], (0, 256 - length))
        causal = torch.ones((256, 256), dtype=torch.bool, device=device).tril()
        logits = reference(
            padded, torch.arange(256, device=device)[None], attention_mask=causal
        )[0, :length]
        ref_numerator = ref_numerator + sequence_loss(logits, seq, sid)
    total_weight = batch["loss_mask"].sum()
    assert total_weight % 2 == 1
    (ref_numerator / total_weight).backward()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()

    engine = MegatronEngine(
        TrainEngineConfig(
            dtype="float32",
            enable_tree_training=True,
            pad_to_maximum=True,
            mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
        )
    )
    engine.device = device
    engine.logger = logging.getLogger("TreeCPTest")
    engine.tf_config = config
    engine.use_padded_seq = False
    engine.parallel_strategy = MegatronParallelStrategy(
        context_parallel_size=dist.get_world_size()
    )
    engine._cpu_group = dist.new_group(backend="gloo")
    engine.process_group_initialized = True
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            grad_reduce_in_fp32=True,
            average_in_collective=False,
        ),
        model,
    )
    engine.model = _MegatronModelList([ddp])
    config.finalize_model_grads_func = finalize_model_grads
    engine.optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="sgd",
            lr=0.01,
            min_lr=0.01,
            weight_decay=0.0,
            sgd_momentum=0.0,
            clip_grad=0.0,
            use_distributed_optimizer=False,
        ),
        engine.model,
    )
    engine._set_optimizer_grad_scale_func()

    def loss_fn(logprobs, entropy, inputs, **kwargs):
        numerator = ((-logprobs + 0.02 * entropy) * inputs["coefficients"]).sum()
        return numerator / inputs["loss_mask"].sum()

    stats = engine.train_batch(batch, loss_fn, lambda inputs: inputs["loss_mask"].sum())
    assert stats["num_micro_batches"] == 2
    for (name, p), (other_name, q) in zip(
        reference.named_parameters(), model.named_parameters()
    ):
        assert name == other_name
        torch.testing.assert_close(q.main_grad, p.grad, rtol=2e-3, atol=2e-5, msg=name)
        torch.testing.assert_close(q, p, rtol=2e-4, atol=2e-6, msg=name)
    if dist.get_rank() == 0:
        print(
            f"PASS actual MegatronEngine tree CP: per_token_loss={config.calculate_per_token_loss}, microbatches=2",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["sdpa", "flex", "mcore", "engine"], default="sdpa"
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        test_layout(device)
        if args.backend == "engine":
            for per_token_loss in [False, True]:
                test_mcore(device, engine_mode=True, per_token_loss=per_token_loss)
        elif args.backend == "mcore":
            test_mcore(device)
        else:
            for short in [False, True]:
                for kv_heads in [8, 4]:
                    test_tiny(device, short, kv_heads, args.backend == "flex")
        if dist.get_rank() == 0:
            print(f"PASS tree CP backend={args.backend}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
