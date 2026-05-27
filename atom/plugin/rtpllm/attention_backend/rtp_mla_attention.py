"""RTP-style MLA attention adapter for rtpllm plugin mode.

ATOM's `MLAAttention` (atom/model_ops/attention_mla.py) is decorated with
vLLM-only plugin-mode methods (forward_impl_plugin_mode / forward_impl_sparse_plugin_mode)
that rely on `vllm.distributed`, `vllm.platforms`, `vllm._custom_ops`, etc. — none of
which are importable in the rtpllm runtime. This module provides an MLA wrapper
that exposes the same constructor surface as `MLAAttention` (so ATOM model code
calling `Attention(use_mla=True, mla_modules=...)` is unchanged) but does the
forward against rtp-llm's `PyAttentionInputs` and the per-layer LayerKVCache
exposed by `RTPForwardContext`.

Dense vs sparse: GLM-5 / DeepSeek-V3.2 always have an indexer (DSA), so the
sparse path via aiter's `unified_attention_sparse_mla` is the primary case.
Dense MLA decode (no indexer) is not exercised by the current ATOM rtpllm
models; we raise rather than half-implement it.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from atom.model_ops.attention_mla import (
    MLAModules,
    _aiter_triton_fp8_bmm,
    dynamic_per_batched_tensor_quant,
)
from atom.model_ops.base_attention import BaseAttention
from atom.model_ops.utils import get_and_maybe_dequant_weights
from atom.plugin.prepare import is_plugin_mode, is_rtpllm
from atom.utils.forward_context import get_forward_context

try:
    import aiter
    from aiter import dtypes, fused_qk_rope_concat_and_cache_mla
    from aiter.ops.triton.attention.unified_attention_sparse_mla import (
        unified_attention_sparse_mla,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    aiter = None
    fused_qk_rope_concat_and_cache_mla = None
    unified_attention_sparse_mla = None
    dtypes = None

try:
    from rtp_llm.models_py.modules.factory.attention.rocm_mla_impl.aiter_mla_params import (
        build_aiter_mla_params,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    build_aiter_mla_params = None


logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.rtp_mla_attention")


def _reshape_mla_kv_buffer(
    raw: torch.Tensor, *, kv_lora_rank: int, qk_rope_head_dim: int
) -> torch.Tensor:
    """Coerce rtp-llm's per-layer MLA cache buffer into aiter's 4D layout.

    `unified_attention_sparse_mla` expects [num_pages, page_size, 1, kv_dim].
    """
    kv_dim = kv_lora_rank + qk_rope_head_dim
    if raw.dim() == 4:
        return raw
    if raw.dim() == 3 and raw.shape[-1] == kv_dim:
        return raw.unsqueeze(2)
    if raw.dim() == 2 and raw.shape[-1] % kv_dim == 0:
        page_size = raw.shape[-1] // kv_dim
        return raw.view(raw.shape[0], page_size, 1, kv_dim)
    raise ValueError(
        f"RTPMlaAttention: unexpected MLA kv_cache_base shape {tuple(raw.shape)} "
        f"for kv_dim={kv_dim}"
    )


class RTPMlaAttention(BaseAttention):
    """MLA attention adapter that drives ATOM-style MLA through rtp-llm metadata.

    Construction surface mirrors `atom.model_ops.attention_mla.MLAAttention`:
    `head_dim = kv_lora_rank + qk_rope_head_dim`, `num_kv_heads = 1`.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str = "bf16",
        layer_num: int = 0,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=use_mla,
            mla_modules=mla_modules,
            **kwargs,
        )
        if mla_modules is None:
            raise ValueError("RTPMlaAttention requires mla_modules.")
        if not use_mla:
            raise ValueError("RTPMlaAttention must be constructed with use_mla=True.")

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = float(scale)
        self.num_kv_heads = int(num_kv_heads)
        self.layer_num = int(layer_num)
        self.kv_cache_dtype = (
            "fp8" if str(kv_cache_dtype).startswith("fp8") else "auto"
        )

        self.q_lora_rank = mla_modules.q_lora_rank
        self.kv_lora_rank = int(mla_modules.kv_lora_rank)
        self.qk_nope_head_dim = int(mla_modules.qk_nope_head_dim)
        self.qk_rope_head_dim = int(mla_modules.qk_rope_head_dim)
        self.qk_head_dim = int(mla_modules.qk_head_dim)
        self.v_head_dim = int(mla_modules.v_head_dim)
        self.rotary_emb = mla_modules.rotary_emb
        self.q_proj = mla_modules.q_proj
        self.o_proj = mla_modules.o_proj
        self.kv_b_proj = mla_modules.kv_b_proj
        self.indexer = mla_modules.indexer
        # Indexer.topk_indices_buffer is created at indexer __init__ time and
        # the same tensor is written into by Indexer.forward each step.
        self.topk_indices_buffer = (
            mla_modules.indexer.topk_indices_buffer
            if mla_modules.indexer is not None
            else None
        )

        self.one_scale = torch.tensor(1.0, dtype=torch.float32)
        self._k_scale = self.one_scale
        self._q_scale = self.one_scale

        # W_K / W_V are produced by process_weights_after_loading. The aiter
        # MLA pipeline absorbs the kv_b_proj weight into Q (W_K) and into the
        # output (W_V), so kv_b_proj itself is never called at forward time.
        self.W_K: Optional[torch.Tensor] = None
        self.W_K_scale: Optional[torch.Tensor] = None
        self.W_V: Optional[torch.Tensor] = None
        self.W_V_scale: Optional[torch.Tensor] = None

        self._dense_warned = False

    # Match MLAAttention.process_weights_after_loading (fp8 branch). The fp4
    # path uses MXFP4 BMM and is only relevant for Quark MXFP4 checkpoints —
    # GLM-5-FP8 is plain fp8, so we keep this branch simple and add fp4 only
    # when an MXFP4 checkpoint shows up in rtpllm mode.
    def process_weights_after_loading(self) -> None:
        if dtypes is None:
            raise RuntimeError(
                "RTPMlaAttention requires aiter for process_weights_after_loading."
            )
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        expected = (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        if tuple(kv_b_proj_weight.shape) != expected:
            raise ValueError(
                f"RTPMlaAttention kv_b_proj weight shape mismatch: "
                f"got {tuple(kv_b_proj_weight.shape)}, expected {expected}"
            )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # Q-absorption operand: (N, P, L)
        W_K = W_UK.transpose(0, 1)
        # V-absorption operand: (N, L, V)
        W_V = W_UV.permute(1, 2, 0)
        self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
            W_K, dtype=dtypes.fp8
        )
        self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
            W_V, dtype=dtypes.fp8
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _q_proj_and_k_absorb(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states_or_q_c → Q, split, absorb W_K into Q-nope."""
        q = self.q_proj(x, x_scale).view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # (N, B, P) for the BMM
        q_nope = q_nope.transpose(0, 1)
        ql_nope = _aiter_triton_fp8_bmm(
            q_nope, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
        )
        # (B, N, L)
        return ql_nope, q_pe

    def _v_up_and_o_proj(self, o: torch.Tensor) -> torch.Tensor:
        # o: [B, N, L]
        x = o.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        x = _aiter_triton_fp8_bmm(
            x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
        )
        x = x.reshape(-1, self.num_heads * self.v_head_dim)
        return self.o_proj(x)

    def _resolve_layer_cache(self):
        fwd_ctx = get_forward_context()
        if fwd_ctx is None:
            raise ValueError("RTPMlaAttention requires forward context in plugin mode.")
        kv_cache_data = fwd_ctx.kv_cache_data
        if kv_cache_data is None:
            raise ValueError("RTPMlaAttention requires kv_cache_data in forward context.")
        layer_entry = kv_cache_data.get(f"layer_{self.layer_num}")
        if layer_entry is None or layer_entry.k_cache is None:
            raise ValueError(
                f"RTPMlaAttention missing layer cache for layer_{self.layer_num}."
            )
        layer_cache = layer_entry.k_cache
        attn_metadata = fwd_ctx.attn_metadata
        attn_inputs = attn_metadata.rtp_attn_inputs
        if attn_inputs is None:
            raise ValueError("RTPMlaAttention requires rtp_attn_inputs in attn_metadata.")
        return fwd_ctx, layer_cache, attn_inputs

    def _build_params(self, attn_inputs, device: torch.device):
        if build_aiter_mla_params is None:
            raise RuntimeError(
                "RTPMlaAttention requires rtp-llm's build_aiter_mla_params."
            )
        fwd_ctx = get_forward_context()
        # MLA kernel page size — for rtp-llm MLA cache, one page == kernel_seq_size_per_block tokens.
        kernel_seq_size_per_block = int(
            getattr(fwd_ctx.attn_metadata, "rtp_kernel_seq_size_per_block", 0) or 0
        )
        if kernel_seq_size_per_block <= 0:
            kernel_seq_size_per_block = int(
                getattr(fwd_ctx.attn_metadata, "rtp_seq_size_per_block", 16) or 16
            )
        return build_aiter_mla_params(
            attn_inputs=attn_inputs,
            page_size=kernel_seq_size_per_block,
            device=device,
        )

    def _forward_impl_plugin_mode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if (
            aiter is None
            or fused_qk_rope_concat_and_cache_mla is None
            or unified_attention_sparse_mla is None
        ):
            raise RuntimeError(
                "RTPMlaAttention requires aiter (fused_qk_rope_concat_and_cache_mla "
                "and unified_attention_sparse_mla) for rtpllm plugin mode."
            )
        if self.W_K is None or self.W_V is None:
            raise RuntimeError(
                "RTPMlaAttention.process_weights_after_loading must run before forward()."
            )
        if positions is None:
            raise ValueError("RTPMlaAttention requires positions for RoPE.")

        # query == hidden_states_or_q_c (post-norm latent)
        # key   == kv_c_normed [tokens, kv_lora_rank]
        # value == k_pe        [tokens, qk_rope_head_dim] or [tokens, 1, qk_rope_head_dim]
        fwd_ctx, layer_cache, attn_inputs = self._resolve_layer_cache()
        device = query.device

        if self.indexer is None or self.topk_indices_buffer is None:
            if not self._dense_warned:
                logger.error(
                    "RTPMlaAttention dense (non-sparse) MLA path not implemented; "
                    "this layer expects an indexer (DSA models like GLM-5)."
                )
                self._dense_warned = True
            raise NotImplementedError(
                "RTPMlaAttention currently supports DSA (sparse) MLA only. "
                "Dense MLA decode/prefill via aiter mla_prefill_fwd / mla_decode_fwd "
                "is not wired into rtpllm yet."
            )

        params = self._build_params(attn_inputs, device=device)
        total_tokens = int(params.qo_indptr[-1].item())

        # Q projection + W_K absorption → ql_nope is (B, N, kv_lora_rank), q_pe is (B, N, rope).
        ql_nope, q_pe = self._q_proj_and_k_absorb(query, q_scale)

        # rtp-llm MLA cache base: tolerate 2D / 3D / 4D layouts. The fused cache write
        # op operates on [num_pages, page_size, kv_dim].
        raw = getattr(layer_cache, "kv_cache_base", None)
        if raw is None:
            raise ValueError(
                f"RTPMlaAttention layer_{self.layer_num} missing kv_cache_base."
            )
        kv_dim = self.kv_lora_rank + self.qk_rope_head_dim
        cache_3d = raw.view(raw.shape[0], -1, kv_dim)

        # Pre-allocate fused q output buffer [B, N, L + rope]; fused op writes into it.
        out_dtype = ql_nope.dtype if ql_nope.dtype.is_floating_point else torch.bfloat16
        q_out = torch.empty(
            (total_tokens, self.num_heads, kv_dim),
            dtype=out_dtype,
            device=device,
        )

        positions_i64 = (
            positions if positions.dtype == torch.int64 else positions.to(torch.int64)
        )
        # key == kv_c_normed (B, kv_lora_rank), value == k_pe (B, qk_rope_head_dim) —
        # match ATOM's server-mode call which passes both as-is.
        fused_qk_rope_concat_and_cache_mla(
            ql_nope,
            q_pe,
            key,
            value,
            cache_3d,
            q_out,
            params.slot_mapping,
            self._k_scale,
            self._q_scale,
            positions_i64,
            self.rotary_emb.cos_cache,
            self.rotary_emb.sin_cache,
            is_neox=self.rotary_emb.is_neox_style,
            is_nope_first=True,
        )

        # Run sparse MLA via aiter triton kernel.
        kv_buffer = _reshape_mla_kv_buffer(
            raw,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        topk = self.topk_indices_buffer[:total_tokens]
        if topk.dim() == 3:
            # [tokens, h_kv=1, topk] → [tokens, topk]
            topk = topk[:, 0, :]
        topk = topk.contiguous()
        if topk.dtype != torch.int32:
            topk = topk.to(torch.int32)

        out = torch.empty(
            (total_tokens, self.num_heads, self.kv_lora_rank),
            dtype=q_out.dtype,
            device=device,
        )
        unified_attention_sparse_mla(
            q=q_out,
            kv=kv_buffer,
            out=out,
            cu_seqlens_q=params.cu_seqlens_q,
            max_seqlen_q=params.max_seqlen_q,
            seqused_k=params.seqused_k,
            max_seqlen_k=params.max_seqlen_k,
            softmax_scale=self.scale,
            topk_indices=topk,
            block_table=params.block_table,
            kv_lora_rank=self.kv_lora_rank,
        )

        return self._v_up_and_o_proj(out)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not is_plugin_mode() or not is_rtpllm():
            raise NotImplementedError(
                "RTPMlaAttention is only supported in rtpllm plugin mode."
            )
        return self._forward_impl_plugin_mode(
            query=query,
            key=key,
            value=value,
            positions=positions,
            q_scale=q_scale,
            **kwargs,
        )
