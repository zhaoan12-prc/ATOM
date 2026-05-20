"""Contract-executable sparse MLA backend for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

from typing import Optional

import torch


class _ContractSparseMlaImpl:
    """CPU/mock sparse implementation used before the real RTP kernel is wired."""

    def __init__(self, v_head_dim: int) -> None:
        self.v_head_dim = int(v_head_dim)
        self.calls = []

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        *,
        topk_indices: torch.Tensor,
        attn_metadata: object,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "q": q,
                "compressed_kv": compressed_kv,
                "k_pe": k_pe,
                "kv_cache": kv_cache,
                "layer_id": layer_id,
                "topk_indices": topk_indices,
                "attn_metadata": attn_metadata,
            }
        )
        return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))


class RTPSparseMlaBackend:
    """M2 sparse top-k consumption contract.

    This backend intentionally avoids importing RTP CUDA sparse kernels. It only
    validates and threads the sparse contract so M2.5 can replace the mock impl.
    """

    def __init__(
        self,
        *,
        dense_backend: object,
        sparse_impl: Optional[object] = None,
        v_head_dim: Optional[int] = None,
    ) -> None:
        self.dense_backend = dense_backend
        self.v_head_dim = int(
            v_head_dim
            if v_head_dim is not None
            else getattr(dense_backend, "v_head_dim")
        )
        self.sparse_impl = sparse_impl or _ContractSparseMlaImpl(self.v_head_dim)

    @staticmethod
    def _get_attn_metadata() -> object:
        try:
            from atom.utils.forward_context import get_forward_context

            return getattr(get_forward_context(), "attn_metadata", None)
        except Exception:
            return None

    @staticmethod
    def _validate_topk_indices(q: torch.Tensor, topk_indices: torch.Tensor) -> None:
        if topk_indices.ndim != 2:
            raise ValueError(
                "Expected topk_indices to be rank-2 [T,K], "
                f"got shape {tuple(topk_indices.shape)}"
            )
        if topk_indices.dtype != torch.int32:
            raise ValueError(
                f"Expected topk_indices dtype torch.int32, got {topk_indices.dtype}"
            )
        if topk_indices.shape[0] != q.shape[0]:
            raise ValueError(
                "Expected topk_indices first dimension to match q tokens, "
                f"got {topk_indices.shape[0]} and {q.shape[0]}"
            )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_indices is None:
            return self.dense_backend.forward(
                q,
                compressed_kv,
                k_pe,
                kv_cache,
                layer_id,
                topk_indices=None,
            )

        self._validate_topk_indices(q, topk_indices)
        return self.sparse_impl.forward(
            q,
            compressed_kv,
            k_pe,
            kv_cache,
            layer_id,
            topk_indices=topk_indices,
            attn_metadata=self._get_attn_metadata(),
        )
