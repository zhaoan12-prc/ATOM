import logging
import os
from typing import Any

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs

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


class _ATOMQwen35MoeRuntime(GptModelBase):
    """rtp-llm runtime adapter backed by ATOM model."""

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
        # ATOM weights should be loaded exactly once from ATOMQwen35Moe._create_python_model.
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        positions = inputs.attention_inputs.position_ids
        inputs_embeds = None
        if input_ids is None or input_ids.numel() == 0:
            inputs_embeds = inputs.input_hiddens

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )
        return PyModelOutputs(hidden_states)


class ATOMQwen35Moe(Qwen35Moe):
    """Qwen3.5-MoE model class that starts ATOM runtime in rtp-llm."""

    @staticmethod
    def _is_external_plugin_mode() -> bool:
        modules = os.getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES", "")
        return "atom.plugin.rtpllm.models" in modules

    def load(self):
        # External plugin mode: bypass rtp-llm native weight loading path and
        # use ATOM model loading only.
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
            self._create_python_model()
            logger.info(
                "External plugin mode: use ATOM loading path and skip rtp-llm native load"
            )
            return

        # Non-plugin mode keeps native behavior.
        super().load()

    def _create_python_model(self):
        # Non-external mode should keep native rtp-llm Python model path.
        if not self._is_external_plugin_mode():
            return super()._create_python_model()

        import atom
        from atom.model_loader.loader import load_model_in_plugin_mode

        atom_model = atom.prepare_model(config=self, engine="rtpllm")
        if atom_model is None:
            raise ValueError("ATOM failed to create qwen3.5-moe model for rtp-llm plugin")

        atom_config = getattr(atom_model, "atom_config", None)
        if atom_config is None:
            atom_config = getattr(getattr(atom_model, "language_model", None), "atom_config", None)
        if atom_config is None:
            raise ValueError(
                "Cannot get atom_config from prepared ATOM model in rtp-llm plugin mode"
            )

        # External plugin mode: load checkpoint once through ATOM loader.
        load_model_in_plugin_mode(
            model=atom_model,
            config=atom_config,
            prefix="model.",
        )

        self.py_model = _ATOMQwen35MoeRuntime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM qwen3.5-moe runtime for rtp-llm plugin mode")
        return self.py_model
