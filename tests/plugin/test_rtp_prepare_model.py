"""Tests for prepare_model orchestration in rtp plugin mode."""

import sys
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


def _make_fake_register_module(model_dict=None):
    mod = MagicMock()
    mod._ATOM_SUPPORTED_MODELS = model_dict or {}
    mod.register_ops_to_sglang = MagicMock()
    mod.register_ops_to_rtp = MagicMock()
    mod.init_aiter_dist = MagicMock()
    mod.set_attn_cls = MagicMock()
    return mod


def test_prepare_model_rtp_happy_path():
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model = MagicMock(name="FakeRtpModel")
    fake_model_cls = MagicMock(return_value=fake_model)

    fake_register = _make_fake_register_module(
        model_dict={"Qwen3ForCausalLM": fake_model_cls}
    )
    mock_gen_config = MagicMock(return_value=fake_atom_config)
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = mock_gen_config

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
        },
    ):
        config = _Obj(architectures=["Qwen3ForCausalLM"])
        result = plugin_prepare.prepare_model(config=config, engine="rtp")

    mock_gen_config.assert_called_once_with(config)
    fake_register.register_ops_to_rtp.assert_called_once_with(
        atom_config=fake_atom_config
    )
    fake_register.register_ops_to_sglang.assert_not_called()
    fake_register.set_attn_cls.assert_called_once()
    fake_register.init_aiter_dist.assert_called_once_with(config=fake_atom_config)
    fake_model_cls.assert_called_once_with(atom_config=fake_atom_config)
    assert result is fake_model


def test_prepare_model_rtp_llm_alias_happy_path():
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model = MagicMock(name="FakeRtpModelAlias")
    fake_model_cls = MagicMock(return_value=fake_model)

    fake_register = _make_fake_register_module(
        model_dict={"Qwen3ForCausalLM": fake_model_cls}
    )
    mock_gen_config = MagicMock(return_value=fake_atom_config)
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = mock_gen_config

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
        },
    ):
        config = _Obj(architectures=["Qwen3ForCausalLM"])
        result = plugin_prepare.prepare_model(config=config, engine="rtp_llm")

    mock_gen_config.assert_called_once_with(config)
    fake_register.register_ops_to_rtp.assert_called_once_with(
        atom_config=fake_atom_config
    )
    fake_model_cls.assert_called_once_with(atom_config=fake_atom_config)
    assert result is fake_model


def test_prepare_model_rtp_rejects_unsupported_architecture():
    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV3ForCausalLM": MagicMock()}
    )

    with patch.dict(sys.modules, {"atom.plugin.register": fake_register}):
        config = _Obj(architectures=["TotallyFakeModelArch"])
        with pytest.raises(ValueError, match="does not support"):
            plugin_prepare.prepare_model(config=config, engine="rtp")


def test_prepare_model_sets_framework_to_rtp():
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model_cls = MagicMock(return_value=MagicMock())

    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV3ForCausalLM": fake_model_cls}
    )
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
        },
    ):
        config = _Obj(architectures=["DeepseekV3ForCausalLM"])
        plugin_prepare.prepare_model(config=config, engine="rtp")

    assert plugin_prepare.is_rtp() is True
    assert plugin_prepare.is_plugin_mode() is True

