"""Contract-executable tests for GLM5 RTP MLA M1 indexer behavior."""

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.rtpllm.attention_backend import rtp_mla_prepare
from atom.plugin.rtpllm.attention_backend.rtp_mla_prepare import (
    RTPMlaPrepareResult,
    build_m0_prepare_result,
    build_m1_prepare_result,
)


def test_m0_prepare_result_rejects_topk_indices():
    q = torch.empty(2, 4, 256)
    compressed_kv = torch.empty(2, 512)
    k_pe = torch.empty(2, 64)
    positions = torch.arange(2, dtype=torch.int32)
    topk = torch.empty(2, 4, dtype=torch.int32)

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


def test_m1_prepare_result_accepts_int32_topk_indices_with_dynamic_k():
    q = torch.empty(2, 4, 256)
    compressed_kv = torch.empty(2, 512)
    k_pe = torch.empty(2, 64)
    positions = torch.arange(2, dtype=torch.int32)
    topk = torch.tensor([[3, 1, 0, 2], [2, 0, 1, 3]], dtype=torch.int32)

    result = build_m1_prepare_result(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        positions=positions,
        topk_indices=topk,
    )

    assert isinstance(result, RTPMlaPrepareResult)
    assert result.topk_indices is topk
    assert result.topk_indices.shape == (2, 4)
    assert result.topk_indices.dtype == torch.int32


@pytest.mark.parametrize(
    "topk_indices",
    [
        torch.empty(2, 4, dtype=torch.int64),
        torch.empty(2, 4, 1, dtype=torch.int32),
        torch.empty(3, 4, dtype=torch.int32),
    ],
)
def test_m1_prepare_result_rejects_invalid_topk_shape_or_dtype(topk_indices):
    q = torch.empty(2, 4, 256)
    compressed_kv = torch.empty(2, 512)
    k_pe = torch.empty(2, 64)
    positions = torch.arange(2, dtype=torch.int32)

    with pytest.raises(ValueError):
        build_m1_prepare_result(
            q=q,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            positions=positions,
            topk_indices=topk_indices,
        )


class _FakeIndexer:
    def __init__(self, attn, values):
        self.attn = attn
        self.values = values
        self.calls = []
        self.weights = torch.full(values.shape, 99.0, dtype=torch.float32)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.attn.indexer_called = True
        self.attn._topk_indices_buffer[: self.values.shape[0], : self.values.shape[1]].copy_(
            self.values
        )
        return self.weights


class _FakeLinear:
    def __init__(self, output):
        self.output = output

    def __call__(self, *_args, **_kwargs):
        return self.output


class _FakeNorm:
    def __call__(self, tensor):
        return tensor


class _FakeM1Attention:
    def __init__(self, hidden_states, topk_values):
        self.indexer_called = False
        self.q_lora_rank = 2
        self.kv_lora_rank = 3
        self.qk_rope_head_dim = 1
        self.num_heads = 2
        self.qk_head_dim = 4
        self.layer_num = 7
        self.index_topk = topk_values.shape[1]
        self._topk_indices_buffer = torch.full(
            (topk_values.shape[0], topk_values.shape[1] + 2),
            -1,
            dtype=torch.int32,
        )
        self.fused_qkv_a_proj = _FakeLinear(
            torch.arange(
                hidden_states.shape[0]
                * (self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=hidden_states.dtype,
            ).reshape(hidden_states.shape[0], -1)
        )
        self.q_a_layernorm = _FakeNorm()
        self.q_b_proj = _FakeLinear(
            torch.arange(
                hidden_states.shape[0] * self.num_heads * self.qk_head_dim,
                dtype=hidden_states.dtype,
            ).reshape(hidden_states.shape[0], -1)
        )
        self.kv_a_layernorm = _FakeNorm()
        self.indexer = _FakeIndexer(self, topk_values)
        self.mla_attn = SimpleNamespace()
        self.o_proj = _FakeLinear(None)

    @property
    def topk_indices_buffer(self):
        if not self.indexer_called:
            raise AssertionError("prepare must call indexer before slicing topk_indices_buffer")
        return self._topk_indices_buffer


def test_m1_default_prepare_calls_indexer_before_slicing_buffer_and_uses_buffer_topk():
    hidden_states = torch.empty(2, 8)
    positions = torch.arange(2, dtype=torch.int32)
    topk_values = torch.tensor([[4, 1, 3, 0], [2, 0, 1, 3]], dtype=torch.int32)
    attn = _FakeM1Attention(hidden_states, topk_values)

    result = rtp_mla_prepare._default_prepare_result(attn, positions, hidden_states)

    assert result.topk_indices is not None
    assert len(attn.indexer.calls) == 1
    assert result.topk_indices.shape == (2, 4)
    assert result.topk_indices.dtype == torch.int32
    assert torch.equal(result.topk_indices, topk_values)
    assert result.topk_indices is not attn.indexer.weights
    assert not torch.equal(result.topk_indices.to(torch.float32), attn.indexer.weights)


def test_m1_get_topk_indices_buffer_reads_real_indexer_owner_path():
    topk_buffer = torch.tensor([[4, 1, 3, 0]], dtype=torch.int32)
    attn = SimpleNamespace(indexer=SimpleNamespace(topk_indices_buffer=topk_buffer))

    assert rtp_mla_prepare._get_topk_indices_buffer(attn) is topk_buffer


def test_m1_should_emit_topk_returns_false_under_dummy_run(monkeypatch):
    from atom.utils import forward_context as forward_context_mod

    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=True, is_prefill=False),
        attn_metadata=SimpleNamespace(max_seqlen_k=4096),
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
    )

    attn = SimpleNamespace(index_topk=4)

    assert rtp_mla_prepare._should_emit_topk_indices(attn) is False


def test_m1_should_emit_topk_returns_false_when_prefill_seqlen_within_index_topk(
    monkeypatch,
):
    from atom.utils import forward_context as forward_context_mod

    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False, is_prefill=True),
        attn_metadata=SimpleNamespace(max_seqlen_k=4),
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
    )

    attn = SimpleNamespace(index_topk=4)

    assert rtp_mla_prepare._should_emit_topk_indices(attn) is False

