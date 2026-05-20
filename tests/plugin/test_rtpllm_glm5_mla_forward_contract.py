"""Contract-executable tests for GLM5 RTP MLA dense forward."""

from types import SimpleNamespace

import torch

from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention


class _FakeDenseBackend:
    def __init__(self, v_head_dim: int):
        self.v_head_dim = v_head_dim
        self.calls = []

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
        self.calls.append(
            {
                "q": q,
                "compressed_kv": compressed_kv,
                "k_pe": k_pe,
                "kv_cache": kv_cache,
                "layer_id": layer_id,
                "topk_indices": topk_indices,
            }
        )
        return q.new_empty((q.shape[0], q.shape[1], self.v_head_dim))


def test_rtp_mla_attention_calls_dense_backend_with_rtp_boundary():
    backend = _FakeDenseBackend(v_head_dim=16)
    attention = RTPMLAAttention(dense_backend=backend, layer_id=7, kv_cache="cache")
    q = torch.empty(3, 2, 12, dtype=torch.bfloat16)
    compressed_kv = torch.empty(3, 8, dtype=torch.bfloat16)
    k_pe = torch.empty(3, 4, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int32)

    output = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=None,
    )

    assert output.shape == (3, 2, 16)
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["q"] is q
    assert call["compressed_kv"] is compressed_kv
    assert call["k_pe"] is k_pe
    assert call["kv_cache"] == "cache"
    assert call["layer_id"] == 7
    assert call["topk_indices"] is None


def test_rtp_mla_attention_rejects_topk_in_m05():
    attention = RTPMLAAttention(dense_backend=_FakeDenseBackend(v_head_dim=16))
    q = torch.empty(1, 2, 12)
    compressed_kv = torch.empty(1, 8)
    k_pe = torch.empty(1, 4)
    positions = torch.arange(1, dtype=torch.int32)
    topk = torch.empty(1, 2048, dtype=torch.int32)

    try:
        attention.forward(
            q,
            compressed_kv,
            k_pe,
            positions=positions,
            topk_indices=topk,
        )
    except ValueError as exc:
        assert "topk" in str(exc)
    else:
        raise AssertionError("M0.5 dense forward must reject topk_indices")


def test_forward_rtp_plugin_mode_flattens_dense_output_before_o_proj():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_prepare import (
        RTPMlaPrepareResult,
        forward_rtp_plugin_mode,
    )

    q = torch.empty(3, 2, 12, dtype=torch.bfloat16)
    compressed_kv = torch.empty(3, 8, dtype=torch.bfloat16)
    k_pe = torch.empty(3, 4, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int32)
    backend = _FakeDenseBackend(v_head_dim=16)
    seen = {}

    class _FakeOProj:
        def __call__(self, tensor):
            seen["input_shape"] = tuple(tensor.shape)
            return tensor

    attn = SimpleNamespace(
        mla_attn=RTPMLAAttention(dense_backend=backend, layer_id=5),
        o_proj=_FakeOProj(),
    )

    def _prepare(_attn, _positions, _hidden_states):
        return RTPMlaPrepareResult(
            q=q,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            positions=positions,
            topk_indices=None,
        )

    output = forward_rtp_plugin_mode(
        attn,
        positions,
        torch.empty(3, 10),
        prepare_fn=_prepare,
    )

    assert seen["input_shape"] == (3, 32)
    assert output.shape == (3, 32)
    assert len(backend.calls) == 1


def test_rtp_mla_attention_builds_m0_backend_from_mla_modules():
    modules = SimpleNamespace(v_head_dim=16)
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)
    q = torch.empty(2, 4, 12)
    compressed_kv = torch.empty(2, 8)
    k_pe = torch.empty(2, 4)

    output = attention(q, compressed_kv, k_pe, positions=torch.arange(2))

    assert output.shape == (2, 4, 16)

