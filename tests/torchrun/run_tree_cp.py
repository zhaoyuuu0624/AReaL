# SPDX-License-Identifier: Apache-2.0

"""Small CP and DP x CP tests; no downloads or pretrained model needed."""

import argparse
import copy
import os
import random
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


def test_mcore(
    device,
    engine_mode=False,
    per_token_loss=False,
    cp_size=None,
    steps=1,
    random_trees=False,
    seed=42,
    precision="fp32",
    checkpoint="none",
    capture=False,
):
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from areal.engine.megatron_utils.tree_context_parallel import (
        tree_context_parallel_forward,
    )
    from areal.models.tree_attn.module_megatron import PytorchFlexAttention

    cp_size = cp_size or dist.get_world_size()
    mpu.initialize_model_parallel(context_parallel_size=cp_size)
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
            context_parallel_size=cp_size,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cpu_initialization=True,
            normalization="LayerNorm",
            masked_softmax_fusion=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            gradient_accumulation_fusion=False,
            calculate_per_token_loss=per_token_loss,
            bf16=precision == "bf16",
            pipeline_dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
            recompute_granularity="full" if checkpoint != "none" else None,
            recompute_method=checkpoint if checkpoint != "none" else None,
            recompute_num_layers=1 if checkpoint != "none" else None,
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
        ref_config.recompute_granularity = None
        ref_config.recompute_method = None
        ref_config.recompute_num_layers = None
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
        if precision == "bf16":
            from megatron.core.transformer.module import Float16Module

            # Initialize in FP32, then quantize the same initial weights used by
            # the FP32 control. Exercise the real MCore mixed-precision wrapper.
            model = Float16Module(config, model)
            reference = Float16Module(ref_config, reference)
            config.params_dtype = ref_config.params_dtype = torch.bfloat16
        root, ids, pos, mask, paths = make_tree(False, device)
        if engine_mode:
            return test_engine_step(
                model,
                reference,
                config,
                ids,
                paths,
                device,
                steps=steps,
                random_trees=random_trees,
                seed=seed,
                capture=capture,
            )
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


def make_engine_sequences(dp_rank, step, seed, cp_size, device):
    """A nested tree, a duplicate, a prefix-only sample, and a separate root.

    Use a private CPU RNG: model/dropout RNG streams must not depend on data.
    DP ranks get unequal lengths and content; CP peers get identical sequences.
    """
    rng = random.Random(seed + step * 101 + dp_rank * 10007)

    def tokens(length):
        return torch.tensor(
            [rng.randrange(1, 64) for _ in range(length)], device=device
        )

    prefix = tokens(rng.randrange(70, 86) - 30 * dp_rank)
    trunk = tokens(rng.randrange(18, 26))
    a, b, c = (
        tokens(rng.randrange(lo, hi)) for lo, hi in [(35, 46), (24, 33), (42, 54)]
    )
    # Make the two divergence points deterministic, not accidental token matches.
    trunk[0], c[0], a[0], b[0] = 1, 2, 3, 4
    first = torch.cat((prefix, trunk, a))
    sequences = [first, torch.cat((prefix, trunk, b)), torch.cat((prefix, c))]
    if step % 2 == 0:
        sequences.append(first.clone())
    sequences.append(prefix.clone())
    extra_length = 135 + step % 7 + 9 * dp_rank
    # Exercise a nonzero remainder in the CP token-count partition.
    weight = sum(seq.numel() - 1 for seq in sequences) + extra_length - 1
    if weight % cp_size == 0:
        extra_length += 1
    sequences.append(torch.zeros(extra_length, dtype=torch.long, device=device))
    return sequences


def test_engine_step(
    model,
    reference,
    config,
    ids,
    paths,
    device,
    steps=1,
    random_trees=False,
    seed=42,
    capture=False,
):
    """Exercise actual packing, loss, MCore schedule, DDP and optimizer.

    Only pretrained loading and rollout infrastructure are bypassed. No manual
    CP division or extra parameter-gradient all-reduce is allowed here.
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from areal.api import MegatronParallelStrategy
    from areal.api.cli_args import MicroBatchSpec, TrainEngineConfig
    from areal.engine.megatron_engine import MegatronEngine, _MegatronModelList
    from areal.utils import logging

    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    assert dp_size * cp_size == dist.get_world_size()
    assert dist.get_world_size(mpu.get_context_parallel_group()) == cp_size
    assert dist.get_world_size(mpu.get_data_parallel_group()) == dp_size
    engine = MegatronEngine(
        TrainEngineConfig(
            dtype="bfloat16" if config.bf16 else "float32",
            gradient_checkpointing=config.recompute_granularity is not None,
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
        data_parallel_size=dp_size, context_parallel_size=cp_size
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
            bf16=config.bf16,
        ),
        engine.model,
    )
    engine._set_optimizer_grad_scale_func()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    reference_params = dict(reference.named_parameters())
    reference_masters = (
        {name: p.detach().float().clone() for name, p in reference_params.items()}
        if config.bf16
        else None
    )
    raw_model = model.module if config.bf16 else model
    calls = {"grad": 0, "no_grad": 0}

    def count_layer_call(module, args, output):
        calls["grad" if torch.is_grad_enabled() else "no_grad"] += 1
        hidden = output[0] if isinstance(output, tuple) else output
        assert hidden.dtype == (torch.bfloat16 if config.bf16 else torch.float32)

    hook = raw_model.decoder.layers[0].register_forward_hook(count_layer_call)
    snapshots = []
    causal = torch.ones((256, 256), dtype=torch.bool, device=device).tril()
    for step in range(steps):
        if random_trees:
            global_sequences = [
                make_engine_sequences(r, step, seed, cp_size, device)
                for r in range(dp_size)
            ]
        else:
            assert dp_size == 1
            global_sequences = [
                [ids[0, path] for path in paths]
                + [torch.zeros(41, dtype=torch.long, device=device)]
            ]
        weights = [sum(seq.numel() - 1 for seq in seqs) for seqs in global_sequences]
        if dp_size > 1:
            assert len(set(weights)) == dp_size, "DP token counts must differ"
        assert all(weight % cp_size != 0 for weight in weights)
        global_weight = sum(
            weights
        )  # Each sequence counted once, NOT once per CP rank.
        sequences = global_sequences[dp_rank]
        shape = (len(sequences), max(seq.numel() for seq in sequences))
        batch = {
            "input_ids": torch.zeros(shape, dtype=torch.long, device=device),
            "attention_mask": torch.zeros(shape, dtype=torch.bool, device=device),
            "loss_mask": torch.zeros(shape, dtype=torch.bool, device=device),
            "coefficients": torch.zeros(shape, dtype=torch.float32, device=device),
        }
        for sid, seq in enumerate(sequences):
            length = seq.numel()
            batch["input_ids"][sid, :length] = seq
            batch["attention_mask"][sid, :length] = True
            batch["loss_mask"][sid, : length - 1] = True
            batch["coefficients"][sid, : length - 1] = (
                torch.linspace(0.3, 1.3, length - 1, device=device)
                * (sid + 1)
                * (dp_rank + 1)
            )

        # Independent serial global-batch oracle on EVERY rank, without gradient
        # collectives or CP-specific normalization. Backward per sequence bounds memory.
        reference_optimizer.zero_grad(set_to_none=True)
        reference_grads = (
            {
                name: torch.zeros_like(master)
                for name, master in reference_masters.items()
            }
            if config.bf16
            else None
        )
        reference_loss = torch.zeros((), device=device)
        for r, seqs in enumerate(global_sequences):
            for sid, seq in enumerate(seqs):
                length = seq.numel()
                padded = F.pad(seq[None], (0, 256 - length))
                logits = reference(
                    padded,
                    torch.arange(256, device=device)[None],
                    attention_mask=causal,
                )[0, :length]
                loss = sequence_loss(logits, seq, sid) * (r + 1) / global_weight
                reference_loss += loss.detach()
                loss.backward()
                if config.bf16:
                    # Accumulate separate sequence contributions in FP32, like
                    # the engine's FP32 main_grad buffers, not BF16 .grad additions.
                    for name, p in reference_params.items():
                        assert p.grad is not None
                        reference_grads[name].add_(p.grad.float())
                        p.grad = None
        if config.bf16:
            with torch.no_grad():
                for name, p in reference_params.items():
                    reference_masters[name].add_(reference_grads[name], alpha=-0.01)
                    p.copy_(reference_masters[name])
        else:
            reference_optimizer.step()

        local_numerator = torch.zeros((), device=device)

        def loss_fn(logprobs, entropy, inputs, **kwargs):
            numerator = ((-logprobs + 0.02 * entropy) * inputs["coefficients"]).sum()
            local_numerator.add_(numerator.detach())
            return numerator / inputs["loss_mask"].sum()

        calls.update(grad=0, no_grad=0)
        masters_before = (
            {
                name: p.main_param.detach().clone()
                for name, p in model.named_parameters()
            }
            if config.bf16
            else None
        )
        stats = engine.train_batch(
            batch, loss_fn, lambda inputs: inputs["loss_mask"].sum()
        )
        assert stats["num_micro_batches"] >= 2
        if not random_trees:
            assert stats["num_micro_batches"] == 2
        n_mbs = int(stats["num_micro_batches"])
        assert calls["grad"] == n_mbs, calls
        assert calls["no_grad"] == (n_mbs if config.recompute_granularity else 0), calls
        # The temporary tree-depth RoPE override must not survive the forward,
        # including when saved tensors are consumed by backward recomputation.
        assert "_preprocess" not in raw_model.__dict__
        # This reduction is for the detached diagnostic loss ONLY; never adjust
        # engine gradients manually to make the oracle comparison pass.
        dist.all_reduce(local_numerator, group=mpu.get_data_parallel_group())
        actual_loss = local_numerator / global_weight
        torch.testing.assert_close(
            actual_loss,
            reference_loss,
            rtol=3e-3 if config.bf16 else 3e-4,
            atol=3e-5,
        )
        gradient_errors, parameter_errors = [], []
        actual_gradients, expected_gradients, actual_masters, actual_parameters = (
            [],
            [],
            [],
            [],
        )
        for (name, p), (other_name, q) in zip(
            reference.named_parameters(), model.named_parameters(), strict=True
        ):
            assert name == other_name
            expected_grad = reference_grads[name] if config.bf16 else p.grad
            assert q.main_grad.dtype == torch.float32
            assert torch.isfinite(q.main_grad).all() and torch.isfinite(q).all()
            if config.bf16:
                assert q.dtype == p.dtype == torch.bfloat16
                assert q.main_param.dtype == torch.float32
                assert_relative_error(
                    q.main_grad, expected_grad, 0.05, name, absolute_floor=1e-6
                )
                torch.testing.assert_close(
                    q.main_param - masters_before[name],
                    -0.01 * q.main_grad,
                    rtol=2e-4,
                    atol=1e-7,
                    msg=f"SGD master update: {name}",
                )
                torch.testing.assert_close(
                    q, q.main_param.to(torch.bfloat16), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    q.main_param,
                    reference_masters[name],
                    rtol=5e-3,
                    atol=2e-4,
                    msg=name,
                )
            else:
                torch.testing.assert_close(
                    q.main_grad, expected_grad, rtol=2e-3, atol=2e-5, msg=name
                )
                torch.testing.assert_close(q, p, rtol=2e-4, atol=2e-6, msg=name)
            gradient_errors.append((q.main_grad - expected_grad).abs().max().item())
            parameter_errors.append((q - p).abs().max().item())
            actual_gradients.append(q.main_grad.detach().flatten())
            expected_gradients.append(expected_grad.detach().flatten())
            actual_masters.append(
                (q.main_param if config.bf16 else q).detach().float().flatten()
            )
            actual_parameters.append(q.detach().float().flatten())
        grad_vector = torch.cat(actual_gradients)
        expected_vector = torch.cat(expected_gradients)
        gradient_relative_error = assert_relative_error(
            grad_vector,
            expected_vector,
            0.03 if config.bf16 else 0.002,
            "global gradient",
            absolute_floor=1e-6,
        )
        if capture:
            snapshots.append(
                {
                    "loss": actual_loss.detach().cpu(),
                    "grad": grad_vector.cpu(),
                    "master": torch.cat(actual_masters).cpu(),
                    "parameters": torch.cat(actual_parameters).cpu(),
                }
            )
        if dist.get_rank() == 0:
            print(
                f"PASS actual MegatronEngine tree CP: DP={dp_size} CP={cp_size} "
                f"per_token_loss={config.calculate_per_token_loss} step={step} "
                f"bf16={config.bf16} checkpoint={config.recompute_method} layer_calls={calls} "
                f"data_seed={seed + step * 101} tokens_per_dp={weights} "
                f"microbatches={stats['num_micro_batches']} "
                f"max_gradient_abs_error={max(gradient_errors):.6g} "
                f"max_parameter_abs_error={max(parameter_errors):.6g}",
                flush=True,
            )
            print(
                f"gradient_relative_l2_error={gradient_relative_error:.6g}", flush=True
            )
    hook.remove()
    return snapshots


def assert_relative_error(actual, expected, bound, name, absolute_floor=0.0):
    """Normwise check alongside (not replacing) strict FP32 elementwise checks."""
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = (actual.float() - expected.float()).norm().item()
    reference_norm = expected.float().norm().item()
    assert error <= bound * reference_norm + absolute_floor, (
        f"{name}: L2 error={error}, reference norm={reference_norm}, bound={bound}"
    )
    return error / max(reference_norm, 1e-12)


def test_precision_checkpoint(device, args, cp_size, per_token_loss):
    """Compare checkpointing to eager at identical precision and BF16 to FP32."""
    kwargs = dict(
        engine_mode=True,
        per_token_loss=per_token_loss,
        cp_size=cp_size,
        steps=args.steps,
        random_trees=args.random_trees,
        seed=args.seed,
        capture=True,
    )
    fp32 = None
    eager = None
    if args.precision == "bf16":
        fp32 = test_mcore(device, **kwargs)
    if args.checkpoint != "none":
        eager = test_mcore(device, precision=args.precision, **kwargs)
    actual = test_mcore(
        device, precision=args.precision, checkpoint=args.checkpoint, **kwargs
    )
    for step, result in enumerate(actual):
        if eager is not None:
            # Same layout, dtype, seed, data and optimizer. Checkpointing must
            # not inherit the looser BF16-vs-FP32 equivalence thresholds.
            for key in ("loss", "grad", "master", "parameters"):
                torch.testing.assert_close(
                    result[key],
                    eager[step][key],
                    rtol=1e-5,
                    atol=1e-6,
                    msg=f"checkpoint vs eager: step={step}, {key}",
                )
        if fp32 is not None:
            torch.testing.assert_close(
                result["loss"], fp32[step]["loss"], rtol=5e-3, atol=3e-5
            )
            error = assert_relative_error(
                result["grad"],
                fp32[step]["grad"],
                0.08,
                f"BF16 vs FP32 gradient step={step}",
            )
            if dist.get_rank() == 0:
                print(
                    f"BF16 vs FP32 step={step} gradient_relative_l2_error={error:.6g}",
                    flush=True,
                )
    if dist.get_rank() == 0:
        print(
            f"PASS precision/checkpoint: dtype={args.precision} checkpoint={args.checkpoint} "
            f"CP={cp_size} steps={args.steps} per_token_loss={per_token_loss}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["sdpa", "flex", "mcore", "engine"], default="sdpa"
    )
    parser.add_argument("--cp-size", type=int, choices=[2, 4])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trees", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--checkpoint", choices=["none", "uniform"], default="none")
    args = parser.parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    cp_size = args.cp_size or world_size
    if cp_size not in (2, 4) or world_size % cp_size:
        parser.error("world size must be divisible by CP=2 or CP=4")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if world_size != cp_size and not (args.backend == "engine" and args.random_trees):
        parser.error("DP>1 requires --backend engine --random-trees")
    if world_size // cp_size > 2:
        parser.error("This bounded fixture supports at most two DP replicas")
    if (args.random_trees or args.steps != 1) and args.backend != "engine":
        parser.error("--random-trees and --steps are engine-only options")
    extended = args.precision != "fp32" or args.checkpoint != "none"
    if extended and args.backend != "engine":
        parser.error("Precision/checkpoint controls require --backend engine")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        test_layout(device)
        if args.backend == "engine":
            for per_token_loss in [False, True]:
                if extended:
                    test_precision_checkpoint(device, args, cp_size, per_token_loss)
                    continue
                test_mcore(
                    device,
                    engine_mode=True,
                    per_token_loss=per_token_loss,
                    cp_size=cp_size,
                    steps=args.steps,
                    random_trees=args.random_trees,
                    seed=args.seed,
                )
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
