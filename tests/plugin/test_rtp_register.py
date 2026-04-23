"""Tests for RTP-related paths in atom.plugin.register."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from atom.plugin import prepare as plugin_prepare


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


def _run_init_aiter_dist(config, mock_init_dist_env, mock_get_method):
    rank = config.plugin_config.rank
    tensor_parallel_size = config.tensor_parallel_size

    assert config.plugin_config.is_plugin_mode

    if config.plugin_config.is_rtp:
        if config.plugin_config.rtp_dist_init_addr is not None:
            dp_master_ip, dp_master_port = (
                config.plugin_config.rtp_dist_init_addr.split(":")
            )
        else:
            dp_master_ip = "127.0.0.1"
            dp_master_port = config.plugin_config.rtp_port_args.nccl_port
    else:
        raise ValueError("Expected RTP config")

    distributed_init_method = mock_get_method(dp_master_ip, dp_master_port)

    mock_init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )


def test_set_attn_cls_rtp_sets_radix_attention():
    plugin_prepare._set_framework_backbone("rtp")
    sentinel_radix = object()
    sentinel_paged = object()

    class _FakeOps:
        Attention = None
        RadixAttention = sentinel_radix
        PagedAttention = sentinel_paged

    if plugin_prepare.is_rtp():
        _FakeOps.Attention = _FakeOps.RadixAttention

    assert _FakeOps.Attention is sentinel_radix


def test_init_aiter_dist_rtp_with_dist_init_addr():
    config = _Obj(
        tensor_parallel_size=2,
        plugin_config=_Obj(
            is_plugin_mode=True,
            is_rtp=True,
            rank=3,
            rtp_dist_init_addr="10.0.0.6:30500",
            rtp_port_args=_Obj(nccl_port=29500),
        ),
        parallel_config=_Obj(data_parallel_size=1, data_parallel_rank=0),
    )

    mock_init_dist_env = MagicMock()
    mock_get_method = MagicMock(return_value="tcp://10.0.0.6:30500")

    _run_init_aiter_dist(config, mock_init_dist_env, mock_get_method)

    mock_get_method.assert_called_once_with("10.0.0.6", "30500")
    mock_init_dist_env.assert_called_once_with(
        tensor_model_parallel_size=2,
        rankID=3,
        backend="nccl",
        distributed_init_method="tcp://10.0.0.6:30500",
        data_parallel_size=1,
        data_parallel_rank=0,
    )


def test_init_aiter_dist_rtp_without_dist_init_addr():
    config = _Obj(
        tensor_parallel_size=4,
        plugin_config=_Obj(
            is_plugin_mode=True,
            is_rtp=True,
            rank=0,
            rtp_dist_init_addr=None,
            rtp_port_args=_Obj(nccl_port=31000),
        ),
        parallel_config=_Obj(data_parallel_size=1, data_parallel_rank=0),
    )

    mock_init_dist_env = MagicMock()
    mock_get_method = MagicMock(return_value="tcp://127.0.0.1:31000")

    _run_init_aiter_dist(config, mock_init_dist_env, mock_get_method)

    mock_get_method.assert_called_once_with("127.0.0.1", 31000)
    mock_init_dist_env.assert_called_once()


def test_register_custom_attention_to_rtp_uses_aiter_name():
    def _package(name: str) -> ModuleType:
        module = ModuleType(name)
        module.__path__ = []
        return module

    recorded = {}

    def _fake_register_attention_backend(name):
        recorded["backend_name"] = name

        def _decorator(factory):
            recorded["factory"] = factory
            return factory

        return _decorator

    class _FakeBackend:
        def __init__(self, runner):
            self.runner = runner

    fake_attention_registry = ModuleType(
        "rtp_llm.srt.layers.attention.attention_registry"
    )
    fake_attention_registry.register_attention_backend = _fake_register_attention_backend

    fake_prepare_mod = ModuleType("atom.plugin.prepare")
    fake_prepare_mod.is_vllm = lambda: False
    fake_prepare_mod.is_sglang = lambda: False
    fake_prepare_mod.is_rtp = lambda: True

    fake_modules = {
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.srt": _package("rtp_llm.srt"),
        "rtp_llm.srt.layers": _package("rtp_llm.srt.layers"),
        "rtp_llm.srt.layers.attention": _package("rtp_llm.srt.layers.attention"),
        "rtp_llm.srt.layers.attention.attention_registry": fake_attention_registry,
        "atom.models.qwen3": ModuleType("atom.models.qwen3"),
        "atom.models.qwen3_moe": ModuleType("atom.models.qwen3_moe"),
        "atom.models.qwen3_5": ModuleType("atom.models.qwen3_5"),
        "atom.models.glm4_moe": ModuleType("atom.models.glm4_moe"),
        "atom.models.deepseek_v2": ModuleType("atom.models.deepseek_v2"),
        "atom.config": ModuleType("atom.config"),
        "atom.plugin.prepare": fake_prepare_mod,
        "atom.plugin.rtp.attention_backend.rtp_attn_backend": ModuleType(
            "atom.plugin.rtp.attention_backend.rtp_attn_backend"
        ),
    }
    fake_modules["atom.models.qwen3"].Qwen3ForCausalLM = type("Qwen3ForCausalLM", (), {})
    fake_modules["atom.models.qwen3_moe"].Qwen3MoeForCausalLM = type(
        "Qwen3MoeForCausalLM", (), {}
    )
    fake_modules[
        "atom.models.qwen3_5"
    ].Qwen3_5ForConditionalGenerationTextOnly = type(
        "Qwen3_5ForConditionalGenerationTextOnly", (), {}
    )
    fake_modules[
        "atom.models.qwen3_5"
    ].Qwen3_5MoeForConditionalGenerationTextOnly = type(
        "Qwen3_5MoeForConditionalGenerationTextOnly", (), {}
    )
    fake_modules["atom.models.glm4_moe"].Glm4MoeForCausalLM = type(
        "Glm4MoeForCausalLM", (), {}
    )
    fake_modules["atom.models.deepseek_v2"].DeepseekV3ForCausalLM = type(
        "DeepseekV3ForCausalLM", (), {}
    )
    fake_modules["atom.config"].Config = type("Config", (), {})
    fake_modules[
        "atom.plugin.rtp.attention_backend.rtp_attn_backend"
    ].ATOMAttnBackendForRtp = _FakeBackend

    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.register", None)
        register_mod = importlib.import_module("atom.plugin.register")
        register_mod = importlib.reload(register_mod)
        register_mod._register_custom_attention_to_rtp()
        backend = recorded["factory"]("runner")

    assert recorded["backend_name"] == "aiter"
    assert isinstance(backend, _FakeBackend)
    assert backend.runner == "runner"

