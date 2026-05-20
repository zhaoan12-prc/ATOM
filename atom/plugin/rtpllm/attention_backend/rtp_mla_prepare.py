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
    return RTPMlaPrepareResult(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
        topk_indices=None,
    )


def _default_prepare_result(attn, positions, hidden_states) -> RTPMlaPrepareResult:
    hidden_states_scale = None
    if isinstance(hidden_states, tuple):
        hidden_states, hidden_states_scale = hidden_states

    if getattr(attn, "q_lora_rank", None) is not None:
        qkv_lora = attn.fused_qkv_a_proj(hidden_states, hidden_states_scale)
        q_c, kv_c, k_pe = torch.split(
            qkv_lora,
            [attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim],
            dim=-1,
        )
        q_c = attn.q_a_layernorm(q_c)
        q = attn.q_b_proj(q_c)
    else:
        q = attn.q_proj(hidden_states, hidden_states_scale)
        kv_c, k_pe = torch.split(
            attn.kv_a_proj_with_mqa(hidden_states, hidden_states_scale),
            [attn.kv_lora_rank, attn.qk_rope_head_dim],
            dim=-1,
        )

    compressed_kv = attn.kv_a_layernorm(kv_c)
    q = q.reshape(-1, attn.num_heads, attn.qk_head_dim)
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

