"""Tests for ATOM's rtp plugin integration.

Covers config translation and plugin-mode dispatch behavior.
"""

import pytest

from atom.plugin import prepare as plugin_prepare
import atom.plugin.config as plugin_config


class _Obj:
    """Minimal attribute bag for faking nested configs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeCompilationConfig:
    def __init__(self, level, use_cudagraph, cudagraph_mode):
        self.level = level
        self.use_cudagraph = use_cudagraph
        self.cudagraph_mode = cudagraph_mode


class _FakeParallelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


def _patch_atom_config_module(monkeypatch):
    import atom.config as atom_config_module

    monkeypatch.setattr(atom_config_module, "Config", _FakeConfig, raising=False)
    monkeypatch.setattr(
        atom_config_module, "ParallelConfig", _FakeParallelConfig, raising=False
    )
    monkeypatch.setattr(
        atom_config_module, "CompilationConfig", _FakeCompilationConfig, raising=False
    )


def test_generate_rtp_config_translates_core_fields(monkeypatch):
    import atom.plugin.config as config_module
    import atom.plugin as plugin_module
    import atom.config as atom_config_module

    _patch_atom_config_module(monkeypatch)
    monkeypatch.setattr(plugin_module, "is_vllm", lambda: False, raising=False)
    monkeypatch.setattr(plugin_module, "is_sglang", lambda: False, raising=False)
    monkeypatch.setattr(plugin_module, "is_rtp", lambda: True, raising=False)
    monkeypatch.setattr(
        atom_config_module, "set_current_atom_config", lambda _cfg: None, raising=False
    )

    cfg = config_module.generate_atom_config_for_plugin_mode(
        config=_Obj(
            model_path="/fake/rtp/model",
            tp_size=4,
            dp_size=2,
            dp_rank=1,
            ep_size=1,
            context_length=4096,
            max_running_requests=64,
            mem_fraction_static=0.8,
            kv_cache_dtype="bf16",
        )
    )

    assert cfg.model == "/fake/rtp/model"
    assert cfg.tensor_parallel_size == 4
    assert cfg.max_num_seqs == 64
    assert cfg.max_model_len == 4096
    assert cfg.parallel_config.data_parallel_size == 2
    assert cfg.parallel_config.data_parallel_rank == 1
    assert cfg.plugin_config.is_plugin_mode is True
    assert cfg.plugin_config.is_rtp is True


def test_generate_atom_config_requires_plugin_mode(monkeypatch):
    import atom.plugin.config as config_module
    import atom.plugin as plugin_module
    import atom.config as atom_config_module

    monkeypatch.setattr(plugin_module, "is_vllm", lambda: False, raising=False)
    monkeypatch.setattr(plugin_module, "is_sglang", lambda: False, raising=False)
    monkeypatch.setattr(plugin_module, "is_rtp", lambda: False, raising=False)
    monkeypatch.setattr(
        atom_config_module, "set_current_atom_config", lambda _cfg: None, raising=False
    )

    with pytest.raises(ValueError, match="running in plugin mode"):
        config_module.generate_atom_config_for_plugin_mode(config=None)


def test_generate_rtp_config_requires_model_path(monkeypatch):
    _patch_atom_config_module(monkeypatch)
    with pytest.raises(RuntimeError, match="model_path"):
        plugin_config._generate_atom_config_from_rtp_config(
            _Obj(
                tp_size=1,
                dp_size=1,
                context_length=1024,
                max_running_requests=8,
            )
        )

