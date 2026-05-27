"""Contract-executable tests for GLM5 RTP MLA native forward."""

import builtins
import inspect
from types import SimpleNamespace

import pytest
import torch

from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention


_FORBIDDEN_CUDA_SPARSE_MODULES = (
    "flashmla_sparse",
    "flash_mla",
    "sparse_mla",
    "attention_mla_sparse",
)


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
        return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))


def test_rtp_mla_attention_keeps_legacy_dense_boundary_during_migration():
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


def _guard_sparse_kernel_imports(monkeypatch):
    original_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if any(part in _FORBIDDEN_CUDA_SPARSE_MODULES for part in name.split(".")):
            raise AssertionError(f"M1 dense contract must not import sparse kernel module: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def test_rtp_mla_attention_accepts_explicit_topk_and_passes_it_to_dense_backend(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    attention = RTPMLAAttention(dense_backend=_FakeDenseBackend(v_head_dim=16))
    q = torch.empty(1, 2, 12)
    compressed_kv = torch.empty(1, 8)
    k_pe = torch.empty(1, 4)
    positions = torch.arange(1, dtype=torch.int32)
    topk = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32)

    output = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=topk,
    )

    assert output.shape == (1, 2, 16)
    assert len(attention.dense_backend.calls) == 1
    assert attention.dense_backend.calls[0]["topk_indices"] is topk


def test_dense_backend_output_does_not_depend_on_topk_values(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    backend = _FakeDenseBackend(v_head_dim=16)
    attention = RTPMLAAttention(dense_backend=backend)
    q = torch.ones(2, 2, 12)
    compressed_kv = torch.empty(2, 8)
    k_pe = torch.empty(2, 4)
    positions = torch.arange(2, dtype=torch.int32)
    topk_a = torch.tensor([[3, 1, 0, 2], [2, 0, 1, 3]], dtype=torch.int32)
    topk_b = torch.tensor([[0, 2, 1, 3], [3, 1, 2, 0]], dtype=torch.int32)

    out_a = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=topk_a,
    )
    out_b = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=topk_b,
    )

    assert torch.equal(out_a, out_b)
    assert backend.calls[0]["topk_indices"] is topk_a
    assert backend.calls[1]["topk_indices"] is topk_b


def test_native_forward_signature_exposes_q_scale_argument():
    signature = inspect.signature(RTPMLAAttention.forward)

    assert "q_scale" in signature.parameters


@pytest.mark.parametrize("attr", ["q_proj", "o_proj", "kv_b_proj", "v_head_dim"])
def test_constructor_injects_native_mla_module_attributes(attr):
    modules = SimpleNamespace(
        q_proj=object(),
        o_proj=object(),
        kv_b_proj=object(),
        v_head_dim=16,
    )
    attention = RTPMLAAttention(mla_modules=modules)

    assert getattr(attention, attr) == getattr(modules, attr)


class _FakeQProj:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, query, q_scale=None):
        self.calls.append((query, q_scale))
        return self.output


class _FakeOProj:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.calls = []

    def __call__(self, tensor):
        self.calls.append(tensor)
        return tensor.new_empty((tensor.shape[0], self.hidden_dim))


def test_native_five_tuple_projects_latent_query_and_applies_o_proj(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    token_count = 3
    num_heads = 2
    qk_head_dim = 4
    v_head_dim = 5
    hidden_dim = 7
    query = torch.arange(token_count * 6, dtype=torch.float32).reshape(token_count, 6)
    q_scale = torch.ones(token_count, 1)
    projected_q = torch.arange(
        token_count * num_heads * qk_head_dim, dtype=torch.float32
    ).reshape(token_count, num_heads * qk_head_dim)
    compressed_kv = torch.empty(token_count, 8)
    k_rope = torch.empty(token_count, 3)
    positions = torch.arange(token_count, dtype=torch.int32)
    backend = _FakeDenseBackend(v_head_dim=v_head_dim)
    modules = SimpleNamespace(
        q_proj=_FakeQProj(projected_q),
        o_proj=_FakeOProj(hidden_dim=hidden_dim),
        kv_b_proj=object(),
        v_head_dim=v_head_dim,
        qk_head_dim=qk_head_dim,
        num_heads=num_heads,
        num_local_heads=num_heads,
    )
    attention = RTPMLAAttention(
        mla_modules=modules,
        dense_backend=backend,
        layer_num=5,
        kv_cache="kv-cache",
    )

    output = attention.forward(
        query,
        compressed_kv,
        k_rope,
        positions=positions,
        q_scale=q_scale,
    )

    assert modules.q_proj.calls == [(query, q_scale)]
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["q"].shape == (token_count, num_heads, qk_head_dim)
    assert torch.equal(call["q"].reshape(token_count, -1), projected_q)
    assert call["compressed_kv"] is compressed_kv
    assert call["k_pe"] is k_rope
    assert call["kv_cache"] == "kv-cache"
    assert call["layer_id"] == 5
    assert len(modules.o_proj.calls) == 1
    assert modules.o_proj.calls[0].shape == (token_count, num_heads * v_head_dim)
    assert output.shape == (token_count, hidden_dim)


def test_rtp_mla_attention_builds_m0_backend_from_mla_modules():
    modules = SimpleNamespace(v_head_dim=16)
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)
    q = torch.empty(2, 4, 12)
    compressed_kv = torch.empty(2, 8)
    k_pe = torch.empty(2, 4)

    output = attention(q, compressed_kv, k_pe, positions=torch.arange(2))

    assert output.shape == (2, 4, 16)


def test_rtp_mla_attention_defaults_to_sparse_backend_from_mla_modules(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    from atom.plugin.rtpllm.attention_backend.rtp_dense_mla_backend import (
        RTPDenseMlaBackend,
    )
    from atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend import (
        RTPSparseMlaBackend,
    )

    modules = SimpleNamespace(v_head_dim=16)
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    assert isinstance(attention.dense_backend, RTPSparseMlaBackend)
    assert isinstance(attention.dense_backend.dense_backend, RTPDenseMlaBackend)


class _FakeKVProj:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.calls = []

    def __call__(self, compressed_kv):
        self.calls.append(compressed_kv)
        return self.output.to(device=compressed_kv.device, dtype=compressed_kv.dtype)


def test_default_dense_mla_backend_computes_nonzero_attention(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    from atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend import (
        RTPSparseMlaBackend,
    )

    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
    compressed_kv = torch.ones(2, 4, dtype=torch.float32)
    # Per token: [k_nope_dim=2, v_head_dim=1].
    kv_projection = torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]])
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(kv_projection),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    output = attention(q, compressed_kv, q.new_empty((2, 0)), positions=torch.arange(2))

    assert isinstance(attention.dense_backend, RTPSparseMlaBackend)
    assert output.shape == (2, 1, 1)
    assert not torch.equal(output, torch.zeros_like(output))
    assert len(modules.kv_b_proj.calls) == 1
    assert modules.kv_b_proj.calls[0] is compressed_kv


def test_default_dense_mla_backend_rejects_bad_kv_projection_shape(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.randn(2, 1, 2)
    compressed_kv = torch.ones(2, 4)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.empty(2, 2)),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    with pytest.raises(ValueError, match="kv_b_proj output shape mismatch"):
        attention(q, compressed_kv, q.new_empty((2, 0)), positions=torch.arange(2))


def test_rtp_mla_attention_explicit_dense_backend_overrides_sparse_default(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    dense_backend = _FakeDenseBackend(v_head_dim=16)
    modules = SimpleNamespace(v_head_dim=16)

    attention = RTPMLAAttention(mla_modules=modules, dense_backend=dense_backend)

    assert attention.dense_backend is dense_backend

