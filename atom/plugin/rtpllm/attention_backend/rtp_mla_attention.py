"""RTP-style MLA adapter for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

from typing import Optional

import torch


class _M0DenseBackend:
    def __init__(self, v_head_dim: int) -> None:
        self.v_head_dim = int(v_head_dim)

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
        return q.new_empty((q.shape[0], q.shape[1], self.v_head_dim))


class RTPMLAAttention:
    """M0 skeleton for an RTP MLA adapter.

    This class intentionally does not inherit or wrap the full-attention adapter.
    M0 establishes the constructor/forward contract only; dense MLA execution is
    filled in before M1 indexer work starts.
    """

    use_mla = True

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        mla_modules = kwargs.get("mla_modules")
        self.dense_backend = kwargs.get("dense_backend")
        if self.dense_backend is None and mla_modules is not None:
            self.dense_backend = _M0DenseBackend(mla_modules.v_head_dim)
        self.kv_cache = kwargs.get("kv_cache")
        self.layer_id = int(kwargs.get("layer_id", kwargs.get("layer_num", 0)))

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk_indices is not None:
            raise ValueError("M0 RTPMLAAttention does not consume sparse topk_indices")
        if self.dense_backend is None:
            raise NotImplementedError(
                "RTPMLAAttention M0.5 requires a dense_backend for contract execution"
            )
        return self.dense_backend.forward(
            q,
            compressed_kv,
            k_pe,
            self.kv_cache,
            self.layer_id,
            topk_indices=None,
        )

    __call__ = forward


def apply_attention_mla_rtpllm_patch() -> None:
    """Switch ATOM's generic Attention symbol to the RTP MLA adapter."""

    import atom.model_ops as ops

    ops.RTPMLAAttention = RTPMLAAttention
    ops.Attention = RTPMLAAttention

