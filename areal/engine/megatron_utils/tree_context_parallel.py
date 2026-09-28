# SPDX-License-Identifier: Apache-2.0

"""Packed-tree BSHD forward with contiguous Ulysses shards and tree-depth RoPE."""

from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import GPTModel

from areal.models.tree_attn.ulysses import TreeCPLayout


def supports_tree_rotary_positions(model: torch.nn.Module) -> bool:
    while hasattr(model, "module"):
        model = model.module
    return (
        type(model) is GPTModel
        and model.position_embedding_type == "rope"
        and not model.config.multi_latent_attention
        and not model.config.num_moe_experts
        and not getattr(model.config, "mtp_num_layers", None)
        and hasattr(model.rotary_pos_emb, "get_emb")
    )


@contextmanager
def tree_rotary_positions(
    model: torch.nn.Module, positions: torch.Tensor, global_tokens: int
):
    """Override only this GPT instance's preprocessing during this forward.

    MCore 0.19 standard RoPE ignores explicit position_ids and normally zigzag
    shards its frequency table. Replace its preprocessor's returned rotary tensor
    with tree-depth frequencies *before* TransformerBlock saves checkpoint inputs.
    No mutable mask/position state is consulted during backward recomputation.
    """
    while hasattr(model, "module"):
        model = model.module
    if not supports_tree_rotary_positions(model):
        raise NotImplementedError("Tree CP requires MCore GPTModel with standard RoPE")
    if model.config.multi_latent_attention or not hasattr(
        model.rotary_pos_emb, "get_emb"
    ):
        raise NotImplementedError(
            "Tree CP does not support MLA or this rotary embedding"
        )
    if positions.ndim != 2 or positions.shape[0] != 1:
        raise ValueError("Tree positions must have shape [1, N/CP]")
    original = model._preprocess
    previous = model.__dict__.get("_preprocess")

    def preprocess(this, *args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 6:
            raise RuntimeError("Unsupported MCore GPT preprocessing contract")
        rotary = this.rotary_pos_emb.get_emb(global_tokens).index_select(
            0, positions.reshape(-1)
        )
        return (result[0], rotary, *result[2:])

    model._preprocess = MethodType(preprocess, model)
    try:
        yield
    finally:
        if previous is None:
            del model._preprocess
        else:
            model._preprocess = previous


def tree_context_parallel_forward(
    model: torch.nn.Module,
    input_: dict[str, Any],
    *,
    fp32_output: bool | None = None,
    gather_cp_output: bool = False,
    is_vision_model: bool = False,
    use_padded_seq: bool = False,
    use_model_packed_seq: bool = False,
    return_hidden_states: bool = False,
) -> torch.Tensor:
    """Return CP-local padded logits; reconstruct only scalars in the loss."""
    if (
        is_vision_model
        or use_padded_seq
        or use_model_packed_seq
        or return_hidden_states
    ):
        raise NotImplementedError("Tree CP supports ordinary dense text logits only")
    # gather_cp_output is intentionally ignored: tree consumers always gather
    # scalars, never vocabulary logits, including forward-only calls.
    input_ids = input_["input_ids"]
    positions = input_["position_ids"]
    cp_group = mpu.get_context_parallel_group()
    layout = TreeCPLayout(
        input_ids.shape[-1], dist.get_world_size(cp_group), dist.get_rank(cp_group)
    )
    local_positions = layout.slice(positions, -1)
    mask = input_["attention_mask"]
    if mask.shape != (layout.global_tokens, layout.global_tokens):
        raise ValueError("Expected a global square tree mask")
    # Also make CP=1's reference padding safe; real-token visibility is unchanged.
    mask = mask | torch.eye(layout.global_tokens, dtype=torch.bool, device=mask.device)
    kwargs = dict(
        input_ids=layout.slice(input_ids, -1),
        position_ids=local_positions,
        attention_mask=mask,
        packed_seq_params=None,
    )
    if fp32_output is not None:
        kwargs["fp32_output"] = fp32_output
    with tree_rotary_positions(model, local_positions, layout.global_tokens):
        output = model(**kwargs)
    if mpu.is_pipeline_last_stage(
        ignore_virtual=False, vp_stage=getattr(model, "vp_stage", None)
    ):
        if output.ndim != 3 or output.shape[:2] != (1, layout.local_tokens):
            raise ValueError("Expected BSHD tree output logits [1, N/CP, V/TP]")
        output = output.squeeze(0)
    return output
