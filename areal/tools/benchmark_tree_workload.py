# SPDX-License-Identifier: Apache-2.0
"""Offline synthetic Megatron tree-training replay. TP=PP=1; static DP x CP.

Uses a random small GPT and real AReaL packing/train_batch/AdamW. This is not
pretrained-model loading, PPO, end-to-end RL, or a numerical equivalence oracle.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from areal.tools.tree_workload import (
    load_workload,
    make_batch,
    packing_report,
    partition,
)


def build_engine(args, payload, device):
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.module import Float16Module

    from areal.api import MegatronParallelStrategy
    from areal.api.cli_args import MicroBatchSpec, TrainEngineConfig
    from areal.engine.megatron_engine import MegatronEngine, _MegatronModelList
    from areal.models.tree_attn.module_megatron import PytorchFlexAttention
    from areal.utils import logging

    torch.manual_seed(args.model_seed)
    model_parallel_cuda_manual_seed(args.model_seed)
    bf16 = args.precision == "bf16"
    config = TransformerConfig(
        num_layers=args.layers,
        hidden_size=args.hidden,
        num_attention_heads=8,
        num_query_groups=8,
        ffn_hidden_size=4 * args.hidden,
        kv_channels=args.hidden // 8,
        context_parallel_size=args.cp,
        add_bias_linear=getattr(args, "add_bias_linear", True),
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cpu_initialization=True,
        normalization=getattr(args, "normalization", "LayerNorm"),
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        gradient_accumulation_fusion=False,
        calculate_per_token_loss=getattr(args, "per_token_loss", False),
        bf16=bf16,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        recompute_granularity="full" if args.checkpoint else None,
        recompute_method="uniform" if args.checkpoint else None,
        recompute_num_layers=1 if args.checkpoint else None,
    )
    spec = get_gpt_layer_local_spec()
    spec.submodules.self_attention.submodules.core_attention = PytorchFlexAttention
    spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.arbitrary
    model = GPTModel(
        config,
        spec,
        payload["vocab_size"],
        args.cap,
        position_embedding_type="rope",
        parallel_output=False,
    ).to(device)
    if bf16:
        model = Float16Module(config, model)
        config.params_dtype = torch.bfloat16
    engine = MegatronEngine(
        TrainEngineConfig(
            dtype="bfloat16" if bf16 else "float32",
            gradient_checkpointing=args.checkpoint,
            enable_tree_training=True,
            pad_to_maximum=True,
            mb_spec=MicroBatchSpec(max_tokens_per_mb=args.cap),
        )
    )
    engine.device = device
    engine.logger = logging.getLogger("TreeWorkloadBenchmark")
    engine.tf_config = config
    engine.use_padded_seq = False
    engine.parallel_strategy = MegatronParallelStrategy(
        data_parallel_size=mpu.get_data_parallel_world_size(),
        context_parallel_size=args.cp,
    )
    engine._cpu_group = dist.new_group(backend="gloo")
    engine.process_group_initialized = True
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=getattr(args, "distributed_optimizer", False),
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
            optimizer="adam",
            lr=1e-4,
            min_lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            use_distributed_optimizer=getattr(args, "distributed_optimizer", False),
            bf16=bf16,
        ),
        engine.model,
    )
    engine._set_optimizer_grad_scale_func()
    return engine, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cp", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--policy", choices=("tree", "sequence"), default="tree")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--model-seed", type=int, default=7)
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    if world % args.cp or args.cap < 128 or args.cap % 128:
        parser.error(
            "WORLD_SIZE must divide by CP and cap must be positive and 128-aligned"
        )
    if (
        args.hidden < 16
        or args.hidden % 16
        or args.layers < 1
        or args.steps < 1
        or args.warmup < 1
    ):
        parser.error(
            "hidden must be a positive multiple of 16; layers/steps/warmup must be positive"
        )
    if os.environ.get("AREAL_USE_TRITON_TREE_ATTN", "0") != "0":
        parser.error("This comparison requires FlexAttention for both CP=1 and CP>1")
    if os.environ.get("AREAL_FLEX_ATTENTION_BLOCK_SIZE", "128") != "128":
        parser.error("This benchmark assumes block size 128")
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = load_workload(args.workload)
    groups = partition(payload["records"], world // args.cp, args.policy)
    if payload["stats"]["max_sequence_length"] > args.cap:
        parser.error("cap is smaller than an original sequence")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(context_parallel_size=args.cp)
    try:
        engine, model = build_engine(args, payload, device)
        batch = make_batch(groups[mpu.get_data_parallel_rank()])
        # Account with the same greedy packing, outside the timed region.
        report = (
            packing_report(payload, world // args.cp, args.cap, args.policy)
            if dist.get_rank() == 0
            else None
        )
        numerator = torch.zeros((), device=device)

        def loss_fn(logprobs, entropy, inputs, **kwargs):
            value = (-logprobs * inputs["loss_mask"]).sum()
            numerator.add_(value.detach())
            return value / inputs["loss_mask"].sum().clamp_min(1)

        rows = []
        for step in range(args.warmup + args.steps):
            numerator.zero_()
            # Benchmark-only start alignment, not a production scheduler barrier.
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            stats = engine.train_batch(batch, loss_fn, lambda x: x["loss_mask"].sum())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            # Do not count replicated CP copies as additional training tokens/loss.
            dist.all_reduce(numerator, group=mpu.get_data_parallel_group())
            loss = float(numerator / payload["stats"]["supervised_tokens"])
            if not torch.isfinite(numerator):
                raise FloatingPointError("Non-finite loss")
            local = dict(
                rank=dist.get_rank(),
                dp_rank=mpu.get_data_parallel_rank(),
                cp_rank=mpu.get_context_parallel_rank(),
                seconds=elapsed,
                peak_allocated_bytes=allocated,
                peak_reserved_bytes=reserved,
                microbatches=int(stats["num_micro_batches"]),
                loss=loss,
            )
            ranks = [None] * world
            dist.all_gather_object(ranks, local, group=engine._cpu_group)
            if dist.get_rank() == 0:
                assert report is not None
                expected = max(r["real_microbatches"] for r in report["per_dp_rank"])
                if any(r["microbatches"] != expected for r in ranks):
                    raise AssertionError(
                        "Runtime microbatch counts differ from packing report"
                    )
                seconds = max(r["seconds"] for r in ranks)
                rows.append(
                    dict(
                        step=step,
                        warmup=step < args.warmup,
                        step_seconds=seconds,
                        original_tokens_per_second=payload["stats"]["original_tokens"]
                        / seconds,
                        supervised_tokens_per_second=payload["stats"][
                            "supervised_tokens"
                        ]
                        / seconds,
                        ranks=ranks,
                    )
                )
                engine.logger.info(
                    "Replay step %s: %.4f s, loss %.6f", step, seconds, loss
                )
        # Out-of-band final finite check; no oracle/backward reference in timing.
        finite = torch.ones((), dtype=torch.int32, device=device)
        for param in model.parameters():
            finite.mul_(torch.isfinite(param).all())
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Non-finite model parameters")
        if dist.get_rank() == 0:
            versions = {}
            for name in ("torch", "megatron-core", "transformer-engine", "mbridge"):
                versions[name] = importlib.metadata.version(name)
            result = dict(
                kind="synthetic_random_small_gpt_training_replay_not_full_rl",
                args={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                world_size=world,
                dp_size=world // args.cp,
                tp_size=1,
                pp_size=1,
                parameters=sum(p.numel() for p in model.parameters()),
                environment=dict(
                    versions=versions,
                    python=platform.python_version(),
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),
                    capability=torch.cuda.get_device_capability(),
                ),
                packing=report,
                timing_scope="CPU packing, transfers, forward/backward, gradient communication and optimizer; excludes start barrier and metrics collection. Warmup updates weights and is recorded but excluded from summary.",
                mean_step_seconds=sum(
                    r["step_seconds"] for r in rows if not r["warmup"]
                )
                / args.steps,
                steps=rows,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
