"""Dense MLA fallback for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

from typing import Any, Optional

import torch


class RTPDenseMlaBackend:
    """Small dense MLA backend used before the sparse kernel is wired.

    This backend intentionally avoids vLLM plugin metadata. It consumes the
    native GLM5 five-tuple already prepared by DeepseekV2MLAAttention and uses
    RTPForwardContext metadata only to recover per-sequence token ranges.
    """

    def __init__(self, *, mla_modules: Any) -> None:
        self.mla_modules = mla_modules
        self.kv_b_proj = getattr(mla_modules, "kv_b_proj", None)
        self.v_head_dim = int(getattr(mla_modules, "v_head_dim"))
        self.qk_nope_head_dim = getattr(mla_modules, "qk_nope_head_dim", None)
        self.qk_rope_head_dim = getattr(mla_modules, "qk_rope_head_dim", None)
        self._projection_checked = False

    @staticmethod
    def _get_query_start_loc(num_tokens: int, device: torch.device) -> torch.Tensor:
        try:
            from atom.utils.forward_context import get_forward_context

            attn_metadata = getattr(get_forward_context(), "attn_metadata", None)
            plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
            query_start_loc = getattr(plugin_metadata, "query_start_loc", None)
            if query_start_loc is None:
                query_start_loc = getattr(plugin_metadata, "rtp_cu_seqlens_q", None)
        except Exception:
            query_start_loc = None

        if query_start_loc is not None and int(query_start_loc.numel()) >= 2:
            query_start_loc = query_start_loc.to(device=device, dtype=torch.int64)
            if int(query_start_loc[0].item()) == 0 and int(query_start_loc[-1].item()) == num_tokens:
                return query_start_loc
        return torch.tensor([0, num_tokens], dtype=torch.int64, device=device)

    @staticmethod
    def _unwrap_linear_output(value: Any) -> torch.Tensor:
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected kv_b_proj to return Tensor, got {type(value)!r}.")
        return value

    def _project_kv(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.kv_b_proj is None:
            return None
        num_tokens, num_heads, qk_head_dim = q.shape
        rope_dim = int(self.qk_rope_head_dim or k_pe.shape[-1])
        nope_dim = int(self.qk_nope_head_dim or (qk_head_dim - rope_dim))
        if nope_dim <= 0:
            raise ValueError(
                f"Invalid MLA qk dims: qk_head_dim={qk_head_dim}, rope_dim={rope_dim}."
            )

        kv_nope = self._unwrap_linear_output(self.kv_b_proj(compressed_kv))
        if kv_nope.numel() == 0:
            raise ValueError("GLM5 RTP dense MLA kv_b_proj returned an empty tensor.")
        expected_last_dim = num_heads * (nope_dim + self.v_head_dim)
        if kv_nope.shape[-1] != expected_last_dim:
            raise ValueError(
                "GLM5 RTP dense MLA kv_b_proj output shape mismatch "
                f"(got={tuple(kv_nope.shape)}, expected_last_dim={expected_last_dim}, "
                f"num_heads={num_heads}, qk_nope_head_dim={nope_dim}, "
                f"v_head_dim={self.v_head_dim})."
            )
        if not self._projection_checked:
            self._projection_checked = True

        kv_nope = kv_nope.reshape(num_tokens, num_heads, nope_dim + self.v_head_dim)
        k_nope, value = kv_nope.split([nope_dim, self.v_head_dim], dim=-1)
        if k_pe.dim() == 2:
            k_pe = k_pe.unsqueeze(1)
        k_pe = k_pe.expand(num_tokens, num_heads, rope_dim)
        key = torch.cat((k_nope, k_pe), dim=-1)
        return key, value

    @staticmethod
    def _causal_attention(
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_loc: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for start_tensor, end_tensor in zip(query_start_loc[:-1], query_start_loc[1:]):
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            if end <= start:
                continue
            q_seg = q[start:end].float()
            k_seg = key[start:end].float()
            v_seg = value[start:end].float()
            scores = torch.einsum("tnd,snd->nts", q_seg, k_seg) * scale
            seq_len = end - start
            causal_mask = torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=q.device
            ).tril()
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            pieces.append(torch.einsum("nts,snd->tnd", probs, v_seg))
        if not pieces:
            return value.new_empty((0, value.shape[1], value.shape[2]))
        return torch.cat(pieces, dim=0)

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del kv_cache, layer_id, topk_indices
        projected = self._project_kv(q, compressed_kv, k_pe)
        if projected is None:
            return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))

        key, value = projected
        query_start_loc = self._get_query_start_loc(q.shape[0], q.device)
        scale = float(q.shape[-1] ** -0.5)
        output = self._causal_attention(q, key, value, query_start_loc, scale)
        return output.to(dtype=compressed_kv.dtype)
