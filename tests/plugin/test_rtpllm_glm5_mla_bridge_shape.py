"""Shape-level tests for the GLM5 M0 MLA bridge skeleton."""

import torch

from atom.plugin.rtpllm.attention_backend.rtp_mla_prepare import (
    RTPMlaPrepareResult,
    build_m0_prepare_result,
)


def test_m0_prepare_result_has_rtp_mla_boundary_shapes():
    q = torch.empty(2, 4, 256)
    compressed_kv = torch.empty(2, 512)
    k_pe = torch.empty(2, 64)
    positions = torch.arange(2, dtype=torch.int32)

    result = build_m0_prepare_result(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
    )

    assert isinstance(result, RTPMlaPrepareResult)
    assert result.q.ndim == 3
    assert result.compressed_kv.ndim == 2
    assert result.k_pe.ndim == 2
    assert result.positions.dtype == torch.int32
    assert result.topk_indices is None


def test_m0_prepare_result_rejects_topk_indices():
    q = torch.empty(2, 4, 256)
    compressed_kv = torch.empty(2, 512)
    k_pe = torch.empty(2, 64)
    positions = torch.arange(2, dtype=torch.int32)
    topk = torch.empty(2, 2048, dtype=torch.int32)

    try:
        build_m0_prepare_result(
            q=q,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            positions=positions,
            topk_indices=topk,
        )
    except ValueError as exc:
        assert "M0" in str(exc)
    else:
        raise AssertionError("M0 prepare should reject topk_indices")

