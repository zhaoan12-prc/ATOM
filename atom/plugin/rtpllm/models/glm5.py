"""GLM5 wrapper for rtp-llm external model loading."""

from __future__ import annotations

import logging
import os
from typing import Any

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs

from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import (
    apply_attention_mla_rtpllm_patch,
)
from atom.plugin.rtpllm.attention_backend.rtp_mla_prepare import (
    apply_deepseek_mla_rtpllm_patch,
)

logger = logging.getLogger("atom.plugin.rtpllm.models")


class _NoopWeightManager:
    def update(self, req):  # noqa: ANN001
        return None


class _NoopModelWeightsLoader:
    _py_eplb = None

    def load_lora_weights(self, adapter_name, lora_path, device):  # noqa: ANN001
        logger.warning(
            "No-op model_weights_loader received load_lora_weights(%s, %s, %s); "
            "external plugin mode uses ATOM model weights path only.",
            adapter_name,
            lora_path,
            device,
        )
        return None


class _ATOMGlm5MoeRuntime(GptModelBase):
    """rtp-llm runtime adapter backed by an ATOM GLM5 model."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        atom_model: Any,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.model = atom_model

    def load_weights(self):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None):  # noqa: ANN001
        raise NotImplementedError("GLM5 rtp-llm runtime forward is not implemented in M0")


class ATOMGlm5Moe(DeepSeekV2):
    """GLM5 model class that starts ATOM runtime in rtp-llm plugin mode."""

    @staticmethod
    def _is_external_plugin_mode() -> bool:
        modules = os.getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES", "")
        return "atom.plugin.rtpllm.models" in modules

    @staticmethod
    def _make_glm5_hf_mapper():
        from atom.model_loader.loader import WeightsMapper

        return WeightsMapper(
            orig_to_new_prefix={},
            orig_to_new_substr={
                "indexers_proj.": "indexer.weights_proj.",
            },
        )

    def load(self, skip_python_model: bool = False):
        if self._is_external_plugin_mode():
            self.device = self._get_device_str()
            self.weight = ModelWeights(
                num_layers=self.model_config.num_layers,
                device=self.device,
                dtype=self.model_config.compute_dtype,
            )
            self.model_weights_loader = _NoopModelWeightsLoader()
            self.py_eplb = self.model_weights_loader._py_eplb
            self.weight_manager = _NoopWeightManager()
            if skip_python_model:
                logger.info(
                    "External plugin mode: skip ATOM GLM5 python model creation as requested"
                )
                return
            self._create_python_model()
            logger.info(
                "External plugin mode: use ATOM GLM5 loading path and skip native load"
            )
            return

        super().load(skip_python_model=skip_python_model)

    def _create_python_model(self):
        if not self._is_external_plugin_mode():
            return super()._create_python_model()

        import atom
        from atom.model_loader.loader import load_model_in_plugin_mode

        apply_attention_mla_rtpllm_patch()
        apply_deepseek_mla_rtpllm_patch()

        atom_model = atom.prepare_model(config=self, engine="rtpllm")
        if atom_model is None:
            raise ValueError("ATOM failed to create GLM5 model for rtp-llm plugin")

        target_device = getattr(self, "device", None)
        if target_device is not None and hasattr(atom_model, "to"):
            atom_model = atom_model.to(target_device)

        atom_config = getattr(atom_model, "atom_config", None)
        if atom_config is None:
            atom_config = getattr(getattr(atom_model, "model", None), "atom_config", None)
        if atom_config is None:
            # M0 tests use mocked ATOM models; real loading must expose atom_config.
            atom_config = getattr(self, "atom_config", None)

        load_model_in_plugin_mode(
            model=atom_model,
            config=atom_config,
            prefix="model.",
            weights_mapper=self._make_glm5_hf_mapper(),
        )

        self.py_model = _ATOMGlm5MoeRuntime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM GLM5 runtime for rtp-llm plugin mode")
        return self.py_model

