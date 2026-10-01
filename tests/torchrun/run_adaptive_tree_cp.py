# SPDX-License-Identifier: Apache-2.0
"""Persistent MCore Adam oracle: adaptive tree CP versus fixed CP1 execution."""

import argparse
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from areal.models.tree_attn.adaptive import AdaptiveTreeConfig
from areal.tools.benchmark_tree_workload import build_engine
from areal.tools.tree_workload import make_batch


def make_global_batch(step):
    lengths = (40, 180, 320, 48)
    length = lengths[step]
    generator = torch.Generator().manual_seed(101 + step)
    records = []
    for root in range(4):
        n = length - root * 5
        prefix = [root + 1] + torch.randint(
            12, 64, (n // 2 - 1,), generator=generator
        ).tolist()
        for branch in range(2 + (root == 0)):
            tail = [5 + branch] + torch.randint(
                12, 64, (n - len(prefix) - 1,), generator=generator
            ).tolist()
            records.append(
                dict(
                    input_ids=prefix + tail,
                    response_start=n - 1 if root == 3 else len(prefix),
                )
            )
    # Duplicate with an independent loss coefficient; prefix-only trajectory.
    records += [
        dict(records[0]),
        dict(input_ids=records[1]["input_ids"][:15], response_start=14),
    ]
    batch = make_batch(records)
    batch["coefficients"] = torch.linspace(
        0.3, 1.3, batch["input_ids"].numel()
    ).reshape_as(batch["input_ids"])
    batch["loss_mask"][2].zero_()
    return batch


def assert_close(actual, expected, bf16, name, *, parameter=False):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"Nonfinite {name}")
    if bf16:
        error = torch.linalg.vector_norm(actual.float() - expected.float())
        bound = (0.01 if parameter else 0.06) * torch.linalg.vector_norm(
            expected.float()
        ) + 2e-6
        assert error <= bound, (name, float(error), float(bound))
    else:
        torch.testing.assert_close(
            actual, expected, rtol=3e-3, atol=3e-6 if parameter else 3e-5, msg=name
        )


def check_optimizer(actual, reference, bf16):
    a, b = actual.optimizer.optimizer, reference.optimizer.optimizer
    for ga, gb in zip(a.param_groups, b.param_groups, strict=True):
        for pa, pb in zip(ga["params"], gb["params"], strict=True):
            assert_close(pa, pb, bf16, "optimizer master parameter", parameter=True)
            sa, sb = a.state[pa], b.state[pb]
            assert sa.keys() == sb.keys()
            for key in sa:
                if isinstance(sa[key], torch.Tensor):
                    assert_close(sa[key], sb[key], bf16, f"Adam {key}")
                else:
                    assert sa[key] == sb[key], key


def run(device, precision, checkpoint, per_token, distributed_optimizer):
    from megatron.core import parallel_state as mpu

    args = SimpleNamespace(
        precision=precision,
        checkpoint=checkpoint,
        layers=2,
        hidden=128,
        cp=1,
        cap=512,
        model_seed=7,
        per_token_loss=per_token,
        distributed_optimizer=distributed_optimizer,
        add_bias_linear=precision != "bf16",
        normalization="RMSNorm" if precision == "bf16" else "LayerNorm",
    )
    actual, actual_model = build_engine(args, {"vocab_size": 64}, device)
    reference, reference_model = build_engine(args, {"vocab_size": 64}, device)
    optimizers = []
    for engine in (actual, reference):
        chain = getattr(engine.optimizer, "chained_optimizers", [engine.optimizer])
        assert len(chain) == 1, "This dense fixture expects one optimizer"
        optimizers.append(chain[0])
    actual_optimizer, reference_optimizer = optimizers
    # Different policies must fail collectively before subgroup construction.
    try:
        actual.configure_adaptive_tree_parallelism(
            AdaptiveTreeConfig(min_dwell_steps=dist.get_rank())
        )
    except ValueError as exc:
        assert "differs across ranks" in str(exc)
    else:
        raise AssertionError("Inconsistent collective configuration was accepted")
    runtime = actual.configure_adaptive_tree_parallelism(
        AdaptiveTreeConfig(
            local_token_budget=128, max_tree_tokens=512, min_dwell_steps=0
        )
    )
    optimizer_identity = id(actual.optimizer)
    world_group = mpu.get_data_parallel_group(with_context_parallel=True)
    original_groups = [m.cp_group for m in runtime.attentions]
    calls = {"grad": 0, "no_grad": 0}
    raw = actual_model.module if precision == "bf16" else actual_model

    def count_calls(module, inputs, output):
        calls["grad" if torch.is_grad_enabled() else "no_grad"] += 1

    handle = raw.decoder.layers[0].register_forward_hook(count_calls)
    try:
        # An invalid leader batch must fail on all workers without hanging.
        try:
            actual.train_adaptive_tree_batch(
                {} if dist.get_rank() == 0 else None, None, None
            )
        except ValueError as exc:
            assert "planning failed" in str(exc)
        else:
            raise AssertionError("Invalid collective input was accepted")
        for step, cp in enumerate((1, 2, 4, 1)):
            batch = make_global_batch(step)
            global_weight = batch["loss_mask"].sum().to(device)
            observed = []
            for engine, dynamic in ((actual, True), (reference, False)):
                numerator = torch.zeros((), device=device)

                def loss_fn(logprobs, entropy, inputs, **kwargs):
                    value = (
                        (-logprobs + 0.02 * entropy)
                        * inputs["coefficients"]
                        * inputs["loss_mask"]
                    ).sum()
                    numerator.add_(value.detach())
                    return value / inputs["loss_mask"].sum().clamp_min(1)

                if dynamic:
                    # Force every boundary for the numerical oracle. Automatic
                    # choices depend on packing and are tested in the replay.
                    stats = engine.train_adaptive_tree_batch(
                        batch if dist.get_rank() == 0 else None,
                        loss_fn,
                        lambda x: x["loss_mask"].sum(),
                        cp_size=cp,
                    )
                    assert stats["adaptive_cp_size"] == cp, stats
                    assert stats["num_micro_batches"] == runtime.last_plan.microbatches
                    numerator.div_(cp)
                else:
                    rows = torch.arange(
                        dist.get_rank(),
                        batch["input_ids"].shape[0],
                        dist.get_world_size(),
                    )
                    local = {
                        key: value.index_select(0, rows) for key, value in batch.items()
                    }
                    engine.train_batch(local, loss_fn, lambda x: x["loss_mask"].sum())
                dist.all_reduce(numerator, group=world_group)
                observed.append(numerator / global_weight)
            assert_close(observed[0], observed[1], precision == "bf16", "loss")
            for (name, pa), (other, pb) in zip(
                actual_model.named_parameters(),
                reference_model.named_parameters(),
                strict=True,
            ):
                assert name == other
                if distributed_optimizer:
                    owned = pa in actual_optimizer.model_param_gbuf_map
                    assert owned == (pb in reference_optimizer.model_param_gbuf_map)
                    if owned:
                        ra = actual_optimizer._get_model_param_range_map(pa)["param"]
                        rb = reference_optimizer._get_model_param_range_map(pb)["param"]
                        assert (ra.start, ra.end) == (rb.start, rb.end)
                        assert_close(
                            pa.main_grad.view(-1)[ra.start : ra.end],
                            pb.main_grad.view(-1)[rb.start : rb.end],
                            precision == "bf16",
                            f"gradient shard {name}",
                        )
                else:
                    assert_close(
                        pa.main_grad,
                        pb.main_grad,
                        precision == "bf16",
                        f"gradient {name}",
                    )
                assert_close(
                    pa, pb, precision == "bf16", f"parameter {name}", parameter=True
                )
            check_optimizer(actual, reference, precision == "bf16")
            assert id(actual.optimizer) == optimizer_identity
            assert actual._active_tree_group is None
            assert [m.cp_group for m in runtime.attentions] == original_groups
            assert mpu.get_context_parallel_world_size() == 1
            assert actual.tf_config.context_parallel_size == 1
            state = runtime.state_dict()
            runtime.load_state_dict(state)
            assert runtime.state_dict() == state
            if dist.get_rank() == 0:
                actual.logger.info(
                    "Adaptive oracle PASS precision=%s checkpoint=%s per_token=%s step=%s CP=%s loss=%.7f",
                    precision,
                    checkpoint,
                    per_token,
                    step,
                    cp,
                    float(observed[0]),
                )
        if checkpoint:
            assert calls["grad"] == calls["no_grad"] and calls["grad"] > 0, calls
        else:
            assert calls["no_grad"] == 0 and calls["grad"] > 0, calls
        # A failed execution restores bindings but cannot be reused as a rollback.
        try:
            with runtime.execution(runtime.last_plan):
                raise RuntimeError("injected execution failure")
        except RuntimeError as exc:
            assert str(exc) == "injected execution failure"
        assert runtime.failed and not runtime.active
        assert actual._active_tree_group is None
        assert [m.cp_group for m in runtime.attentions] == original_groups
        try:
            actual.train_adaptive_tree_batch(None, None, None)
        except RuntimeError as exc:
            assert "failed" in str(exc)
        else:
            raise AssertionError("Failed runtime was reused")
    finally:
        handle.remove()
        runtime.close()
        dist.destroy_process_group(actual.cpu_group)
        dist.destroy_process_group(reference.cpu_group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--distributed-optimizer", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=8))
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(context_parallel_size=1)
    try:
        for per_token in (False, True):
            run(
                device,
                args.precision,
                args.checkpoint,
                per_token,
                args.distributed_optimizer,
            )
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
