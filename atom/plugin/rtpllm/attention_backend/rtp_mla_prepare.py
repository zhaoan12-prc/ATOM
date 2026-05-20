"""M0 preparation helpers for GLM5 MLA in rtp-llm plugin mode."""

from __future__ import annotations

from functools import wraps
from typing import Callable, NamedTuple, Optional

import torch


class RTPMlaPrepareResult(NamedTuple):
    q: torch.Tensor
    compressed_kv: torch.Tensor
    k_pe: torch.Tensor
    positions: torch.Tensor
    topk_indices: Optional[torch.Tensor] = None


def _validate_prepare_base(
    *,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    if q.ndim != 3:
        raise ValueError(f"Expected q to be rank-3 [T,H,D], got shape {tuple(q.shape)}")
    if compressed_kv.ndim != 2:
        raise ValueError(
            "Expected compressed_kv to be rank-2 [T,kv_lora_rank], "
            f"got shape {tuple(compressed_kv.shape)}"
        )
    if k_pe.ndim != 2:
        raise ValueError(f"Expected k_pe to be rank-2 [T,D], got shape {tuple(k_pe.shape)}")
    if positions.ndim != 1:
        raise ValueError(
            f"Expected positions to be rank-1 [T], got shape {tuple(positions.shape)}"
        )


def build_m0_prepare_result(
    *,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    topk_indices: Optional[torch.Tensor] = None,
) -> RTPMlaPrepareResult:
    """Build the RTP MLA boundary object for M0 dense-only mode."""

    if topk_indices is not None:
        raise ValueError("M0 dense MLA mode must not receive topk_indices")
    _validate_prepare_base(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
    )
    return RTPMlaPrepareResult(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
        topk_indices=None,
    )


def build_m1_prepare_result(
    *,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    topk_indices: torch.Tensor,
) -> RTPMlaPrepareResult:
    """Build the RTP MLA boundary object for M1 indexer contract mode."""

    _validate_prepare_base(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
    )
    if topk_indices.ndim != 2:
        raise ValueError(
            "Expected topk_indices to be rank-2 [T,K], "
            f"got shape {tuple(topk_indices.shape)}"
        )
    if topk_indices.dtype != torch.int32:
        raise ValueError(f"Expected topk_indices dtype torch.int32, got {topk_indices.dtype}")
    if topk_indices.shape[0] != q.shape[0]:
        raise ValueError(
            "Expected topk_indices first dimension to match q tokens, "
            f"got {topk_indices.shape[0]} and {q.shape[0]}"
        )
    return RTPMlaPrepareResult(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
        topk_indices=topk_indices,
    )


def _resolve_index_topk(attn) -> int:
    for obj, attr in (
        (getattr(attn, "indexer", None), "index_topk"),
        (getattr(attn, "indexer", None), "topk_tokens"),
        (attn, "index_topk"),
        (getattr(attn, "config", None), "index_topk"),
    ):
        value = getattr(obj, attr, None) if obj is not None else None
        if value is not None:
            return int(value)
    raise AttributeError("GLM5 RTP MLA M1 indexer requires index_topk/topk_tokens")


def _get_topk_indices_buffer(attn) -> torch.Tensor:
    buffer = getattr(attn, "topk_indices_buffer", None)
    if buffer is None:
        buffer = getattr(attn, "_topk_indices_buffer", None)
    if buffer is None:
        indexer = getattr(attn, "indexer", None)
        buffer = getattr(indexer, "topk_indices_buffer", None)
    if buffer is None:
        raise AttributeError("GLM5 RTP MLA M1 indexer requires topk_indices_buffer")
    return buffer


def _should_emit_topk_indices(attn) -> bool:
    try:
        from atom.utils.forward_context import get_forward_context

        forward_context = get_forward_context()
    except Exception:
        return True

    context = getattr(forward_context, "context", None)
    if getattr(context, "is_dummy_run", False):
        return False
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if getattr(context, "is_prefill", False) and attn_metadata is not None:
        max_seqlen_k = getattr(attn_metadata, "max_seqlen_k", None)
        if max_seqlen_k is not None:
            try:
                return int(max_seqlen_k) > _resolve_index_topk(attn)
            except AttributeError:
                return True
    return True


def _prepare_qkv_native_compatible(attn, hidden_states, hidden_states_scale):
    if getattr(attn, "q_lora_rank", None) is not None:
        if getattr(attn, "fuse_qknorm_quant", False):
            try:
                from atom.models.deepseek_v2 import (
                    _fuse_qkv_a_proj_reduce_rmsnorm_quant,
                    use_triton_gemm,
                )
            except Exception:
                use_triton_gemm = lambda: False  # noqa: E731

            if use_triton_gemm():
                q_c, q_c_scale, kv_c_normed, k_pe = (
                    _fuse_qkv_a_proj_reduce_rmsnorm_quant(
                        hidden_states,
                        attn.fused_qkv_a_proj.weight,
                        attn.fused_qkv_a_proj.weight_scale,
                        attn.q_a_layernorm.weight,
                        attn.q_a_layernorm.eps,
                        attn.kv_a_layernorm.weight,
                        attn.kv_a_layernorm.eps,
                        attn.q_lora_rank,
                        attn.kv_lora_rank,
                        attn.qk_rope_head_dim,
                        dtype_quant=attn.quant_dtype,
                        hidden_states_quant_scale=hidden_states_scale,
                        shuffle=True,
                        scale_shuffle_padding=True,
                        group_size=128,
                        output_unquantized_inp1=False,
                        transpose_scale=True,
                    )
                )
                return q_c, q_c_scale, kv_c_normed, k_pe

        qkv_lora = attn.fused_qkv_a_proj(hidden_states, hidden_states_scale)
        q_c, kv_c, k_pe = torch.split(
            qkv_lora,
            [attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim],
            dim=-1,
        )
        if getattr(attn, "fuse_qknorm_quant", False) or getattr(attn, "fuse_qknorm", False):
            from atom.models.deepseek_v2 import _fuse_rmsnorm_quant

            (
                (hidden_states_or_q_c, hidden_states_or_q_c_scale),
                _,
                kv_c_normed,
                _,
            ) = _fuse_rmsnorm_quant(
                q_c,
                attn.q_a_layernorm.weight,
                attn.q_a_layernorm.eps,
                kv_c,
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.eps,
                None,
                dtype_quant=getattr(attn, "quant_dtype", None),
                shuffle=False,
                scale_shuffle_padding=False,
                group_size=128,
                quant_type=getattr(attn, "qknorm_quant_type", None),
                output_unquantized_inp1=False,
                transpose_scale=True,
            )
            return hidden_states_or_q_c, hidden_states_or_q_c_scale, kv_c_normed, k_pe

        hidden_states_or_q_c = attn.q_a_layernorm(q_c)
        kv_c_normed = attn.kv_a_layernorm(kv_c)
        return hidden_states_or_q_c, None, kv_c_normed, k_pe

    hidden_states_or_q_c = hidden_states
    kv_c, k_pe = torch.split(
        attn.kv_a_proj_with_mqa(hidden_states, hidden_states_scale),
        [attn.kv_lora_rank, attn.qk_rope_head_dim],
        dim=-1,
    )
    kv_c_normed = attn.kv_a_layernorm(kv_c)
    return hidden_states_or_q_c, None, kv_c_normed, k_pe


def _default_prepare_result(attn, positions, hidden_states) -> RTPMlaPrepareResult:
    hidden_states_scale = None
    if isinstance(hidden_states, tuple):
        hidden_states, hidden_states_scale = hidden_states

    hidden_states_or_q_c, hidden_states_or_q_c_scale, compressed_kv, k_pe = (
        _prepare_qkv_native_compatible(attn, hidden_states, hidden_states_scale)
    )
    q_proj = attn.q_b_proj if getattr(attn, "q_lora_rank", None) is not None else attn.q_proj
    q = q_proj(hidden_states_or_q_c, hidden_states_or_q_c_scale)
    num_heads = getattr(attn, "num_local_heads", None)
    if num_heads is None:
        num_heads = getattr(attn, "num_heads")
    num_heads = int(num_heads)
    q = q.reshape(-1, num_heads, attn.qk_head_dim)
    indexer = getattr(attn, "indexer", None)
    if indexer is not None:
        indexer(
            hidden_states,
            hidden_states_or_q_c,
            hidden_states_or_q_c_scale,
            positions,
            getattr(attn, "indexer_rope_emb", None),
        )
        if not _should_emit_topk_indices(attn):
            return build_m0_prepare_result(
                q=q,
                compressed_kv=compressed_kv,
                k_pe=k_pe,
                positions=positions,
                topk_indices=None,
            )
        index_topk = _resolve_index_topk(attn)
        topk_indices = _get_topk_indices_buffer(attn)[: q.shape[0], :index_topk]
        return build_m1_prepare_result(
            q=q,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            positions=positions,
            topk_indices=topk_indices,
        )
    return build_m0_prepare_result(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
        topk_indices=None,
    )


def forward_rtp_plugin_mode(
    attn,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    prepare_fn: Optional[Callable] = None,
) -> torch.Tensor:
    prepare = prepare_fn or _default_prepare_result
    prepared = prepare(attn, positions, hidden_states)
    mla_attn = getattr(attn, "mla_attn", None)
    if mla_attn is None:
        mla_attn = getattr(attn, "rtp_mla_attn", None)
    if mla_attn is None:
        raise AttributeError("GLM5 RTP MLA patch requires attn.mla_attn")
    attn_output = mla_attn(
        prepared.q,
        prepared.compressed_kv,
        prepared.k_pe,
        positions=prepared.positions,
        topk_indices=prepared.topk_indices,
    )
    if attn_output is not None:
        attn_output = attn_output.reshape(attn_output.shape[0], -1).contiguous()
    return attn.o_proj(attn_output)


def apply_deepseek_mla_rtpllm_patch(attention_cls=None) -> None:
    """Patch ``DeepseekV2MLAAttention.forward`` for rtp-llm plugin mode."""

    if attention_cls is None:
        from atom.models.deepseek_v2 import DeepseekV2MLAAttention

        attention_cls = DeepseekV2MLAAttention

    current_forward = attention_cls.forward
    if getattr(current_forward, "_rtpllm_patched", False):
        return None

    @wraps(current_forward)
    def _rtpllm_forward(self, positions, hidden_states):
        return forward_rtp_plugin_mode(self, positions, hidden_states)

    _rtpllm_forward._rtpllm_patched = True
    attention_cls.forward = _rtpllm_forward
    return None

