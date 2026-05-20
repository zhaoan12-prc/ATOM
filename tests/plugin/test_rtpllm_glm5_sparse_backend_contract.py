"""Contract-executable tests for GLM5 RTP MLA M2 sparse topk consumption."""

import builtins
import importlib
import inspect
from types import SimpleNamespace

import torch


_SPARSE_BACKEND_MODULE = (
    "atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend"
)
_FORBIDDEN_CUDA_SPARSE_MODULES = (
    "flashmla_sparse",
    "flash_mla",
    "sparse_mla",
    "attention_mla_sparse",
)


def _guard_sparse_kernel_imports(monkeypatch):
    original_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if any(part in _FORBIDDEN_CUDA_SPARSE_MODULES for part in name.split(".")):
            raise AssertionError(f"M2 sparse contract must not import CUDA sparse kernel: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _load_sparse_backend(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    importlib.invalidate_caches()
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    return module.RTPSparseMlaBackend


class _FakeDenseBackend:
    def __init__(self, v_head_dim: int = 5):
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
        return q.new_full((q.shape[0], q.shape[1], self.v_head_dim), -1)


class _FakeSparseImpl:
    def __init__(self, v_head_dim: int = 5):
        self.v_head_dim = v_head_dim
        self.calls = []

    def forward(
        self,
        q,
        compressed_kv,
        k_pe,
        kv_cache,
        layer_id,
        *,
        topk_indices,
        attn_metadata,
    ):
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
        return q.new_full((q.shape[0], q.shape[1], self.v_head_dim), 7)


def _build_backend(backend_cls, dense_backend, sparse_impl):
    params = inspect.signature(backend_cls).parameters
    kwargs = {}
    if "dense_backend" not in params:
        raise AssertionError("RTPSparseMlaBackend must accept dense_backend= for dense fallback")
    kwargs["dense_backend"] = dense_backend

    if "sparse_impl" in params:
        kwargs["sparse_impl"] = sparse_impl
    else:
        raise AssertionError("RTPSparseMlaBackend must accept a mock sparse impl injection")

    if "v_head_dim" in params:
        kwargs["v_head_dim"] = dense_backend.v_head_dim
    return backend_cls(**kwargs)


def _make_inputs():
    return (
        torch.randn(3, 2, 4),
        torch.randn(3, 8),
        torch.randn(3, 3),
        SimpleNamespace(name="kv-cache"),
        11,
    )


def test_sparse_backend_passes_topk_through_unchanged(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[4, 1], [3, 0], [2, 1]], dtype=torch.int32)

    output = backend.forward(
        q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk
    )

    assert output.shape == (3, 2, sparse_impl.v_head_dim)
    assert dense_backend.calls == []
    assert len(sparse_impl.calls) == 1
    assert sparse_impl.calls[0]["topk_indices"] is topk
    assert sparse_impl.calls[0]["topk_indices"].dtype == torch.int32
    assert sparse_impl.calls[0]["topk_indices"].shape == (3, 2)


def test_sparse_backend_falls_back_to_dense_when_topk_is_none(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()

    output = backend.forward(
        q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None
    )

    assert output.shape == (3, 2, dense_backend.v_head_dim)
    assert len(dense_backend.calls) == 1
    assert sparse_impl.calls == []
    assert dense_backend.calls[0]["q"] is q
    assert dense_backend.calls[0]["compressed_kv"] is compressed_kv
    assert dense_backend.calls[0]["k_pe"] is k_pe
    assert dense_backend.calls[0]["kv_cache"] is kv_cache
    assert dense_backend.calls[0]["layer_id"] == layer_id
    assert dense_backend.calls[0]["topk_indices"] is None


def test_sparse_backend_threads_kv_cache_and_layer_id_to_sparse_impl(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32)

    backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk)

    call = sparse_impl.calls[0]
    assert call["q"] is q
    assert call["compressed_kv"] is compressed_kv
    assert call["k_pe"] is k_pe
    assert call["kv_cache"] is kv_cache
    assert call["layer_id"] == layer_id


def test_sparse_backend_pulls_attn_metadata_from_forward_context(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    from atom.utils import forward_context as forward_context_mod

    attn_metadata = SimpleNamespace(block_table="block-table", seq_lens="seq-lens")
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=attn_metadata,
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
    )
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32)

    backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk)

    assert sparse_impl.calls[0]["attn_metadata"] is attn_metadata


def test_sparse_backend_forward_signature_matches_dense_boundary(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)

    signature = inspect.signature(backend_cls.forward)
    params = signature.parameters

    assert list(params) == [
        "self",
        "q",
        "compressed_kv",
        "k_pe",
        "kv_cache",
        "layer_id",
        "topk_indices",
    ]
    assert params["topk_indices"].default is None
