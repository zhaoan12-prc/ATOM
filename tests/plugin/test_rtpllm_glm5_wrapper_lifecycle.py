"""Lifecycle tests for the GLM5 rtp-llm wrapper."""

from contextlib import nullcontext
import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_fake_rtp_modules() -> dict[str, ModuleType]:
    fake_config_mod = ModuleType("rtp_llm.config.model_config")

    class _FakeModelConfig:
        pass

    fake_config_mod.ModelConfig = _FakeModelConfig

    fake_factory_register_mod = ModuleType("rtp_llm.model_factory_register")
    fake_factory_register_mod.register_model = MagicMock()
    fake_factory_register_mod._model_factory = {}
    fake_factory_register_mod._hf_architecture_2_ft = {}

    fake_deepseek_mod = ModuleType("rtp_llm.models.deepseek_v2")

    class _FakeDeepSeekV2:
        def _get_device_str(self):
            return "cpu"

        def _create_python_model(self):
            self.native_create_python_model_called = True

        def load(self, skip_python_model=False):
            self.native_load_called = skip_python_model

    fake_deepseek_mod.DeepSeekV2 = _FakeDeepSeekV2

    fake_weight_info_mod = ModuleType("rtp_llm.model_loader.model_weight_info")

    class _FakeModelWeights:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_weight_info_mod.ModelWeights = _FakeModelWeights

    fake_module_base_mod = ModuleType("rtp_llm.models_py.model_desc.module_base")

    class _FakeGptModelBase:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs

    fake_module_base_mod.GptModelBase = _FakeGptModelBase

    fake_ops_mod = ModuleType("rtp_llm.ops")

    class _FakeParallelismConfig:
        pass

    fake_ops_mod.ParallelismConfig = _FakeParallelismConfig

    fake_compute_ops_mod = ModuleType("rtp_llm.ops.compute_ops")

    class _FakePyModelInputs:
        pass

    class _FakePyModelOutputs:
        pass

    fake_compute_ops_mod.PyModelInputs = _FakePyModelInputs
    fake_compute_ops_mod.PyModelOutputs = _FakePyModelOutputs

    fake_weight_mod = ModuleType("rtp_llm.utils.model_weight")
    fake_weight_mod.W = SimpleNamespace(
        lm_head="lm_head",
        embedding="embedding",
        final_ln_gamma="final_ln_gamma",
    )

    fake_loader_mod = ModuleType("atom.model_loader.loader")

    class _FakeWeightsMapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_loader_mod.WeightsMapper = _FakeWeightsMapper
    fake_loader_mod.load_model_in_plugin_mode = MagicMock()

    return {
        "atom.model_loader": _package("atom.model_loader"),
        "atom.model_loader.loader": fake_loader_mod,
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.config": _package("rtp_llm.config"),
        "rtp_llm.config.model_config": fake_config_mod,
        "rtp_llm.model_factory_register": fake_factory_register_mod,
        "rtp_llm.models": _package("rtp_llm.models"),
        "rtp_llm.models.deepseek_v2": fake_deepseek_mod,
        "rtp_llm.model_loader": _package("rtp_llm.model_loader"),
        "rtp_llm.model_loader.model_weight_info": fake_weight_info_mod,
        "rtp_llm.models_py": _package("rtp_llm.models_py"),
        "rtp_llm.models_py.model_desc": _package("rtp_llm.models_py.model_desc"),
        "rtp_llm.models_py.model_desc.module_base": fake_module_base_mod,
        "rtp_llm.ops": fake_ops_mod,
        "rtp_llm.ops.compute_ops": fake_compute_ops_mod,
        "rtp_llm.utils": _package("rtp_llm.utils"),
        "rtp_llm.utils.model_weight": fake_weight_mod,
    }


def _make_wrapper_instance(cls):
    instance = cls.__new__(cls)
    instance.model_config = SimpleNamespace(
        num_layers=1,
        compute_dtype=torch.bfloat16,
    )
    instance.parallelism_config = SimpleNamespace()
    instance.max_generate_batch_size = 1
    instance.fmha_config = None
    instance.hw_kernel_config = None
    instance.device_resource_config = None
    return instance


def test_glm5_load_skip_python_model_does_not_create_atom_model():
    fake_modules = _install_fake_rtp_modules()

    with patch.dict(sys.modules, fake_modules), patch.dict(
        os.environ,
        {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        instance = _make_wrapper_instance(module.ATOMGlm5Moe)
        instance._create_python_model = MagicMock()

        instance.load(skip_python_model=True)

        instance._create_python_model.assert_not_called()
        assert instance.device == "cpu"
        assert isinstance(instance.model_weights_loader, module._NoopModelWeightsLoader)
        assert isinstance(instance.weight_manager, module._NoopWeightManager)


def _patch_optional_attr(module, attr):
    if hasattr(module, attr):
        return patch.object(module, attr)
    return nullcontext(MagicMock(name=attr))


def test_glm5_create_python_model_lets_prepare_model_own_mla_patching():
    fake_modules = _install_fake_rtp_modules()
    fake_atom_model = MagicMock(name="atom_model")
    fake_atom_model.to.return_value = fake_atom_model

    with patch.dict(
        sys.modules,
        fake_modules,
    ), patch.dict(
        os.environ,
        {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
    ), patch("atom.prepare_model", return_value=fake_atom_model, create=True) as prepare_model:
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        instance = _make_wrapper_instance(module.ATOMGlm5Moe)
        instance.device = "cpu"
        instance.weight = MagicMock()

        with _patch_optional_attr(
            module, "apply_attention_mla_rtpllm_patch"
        ) as mla_patch, _patch_optional_attr(
            module, "apply_deepseek_mla_rtpllm_patch"
        ) as deepseek_patch:
            result = instance._create_python_model()

        prepare_model.assert_called_once_with(config=instance, engine="rtpllm")
        mla_patch.assert_not_called()
        deepseek_patch.assert_not_called()
        load_model_in_plugin_mode = fake_modules[
            "atom.model_loader.loader"
        ].load_model_in_plugin_mode
        load_model_in_plugin_mode.assert_called_once()
        assert result is instance.py_model

