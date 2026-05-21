"""Generic ATOM-on-rtp-llm base class.

This inherits directly from rtp_llm.models.base_model.BaseModel and is **not**
tied to any specific model family in rtp-llm. The original
atom/plugin/rtpllm/models/qwen3_5.py inherits from rtp-llm's Qwen35Moe, which
requires that class to exist on the rtp-llm side. This module breaks that
coupling so you can plug ATOM models that rtp-llm has no native implementation
of.

Subclass contract (override the methods marked OVERRIDE):

  - `_create_config(cls, ckpt_path)`   OVERRIDE — parse HF config.json into ModelConfig
                                       (use config_helpers.py for the boring parts).
  - `get_weight_cls()`                 OVERRIDE-OPTIONAL — return a ModelDeployWeightInfo
                                       subclass. In plugin mode it's not exercised by
                                       the load path, so a stub is fine.

  - `_atom_engine_name(self)`          default "rtpllm" — passed to atom.prepare_model.
  - `_atom_apply_pre_create_patches()` OVERRIDE-OPTIONAL — apply attention patches before
                                       the ATOM model is constructed.
  - `_atom_make_weights_mapper(self)`  OVERRIDE-OPTIONAL — WeightsMapper for checkpoint
                                       prefix/substring rewrites.
  - `_atom_load_fused_expert_weights_fn(self)` OVERRIDE-OPTIONAL — for fused MoE checkpoints.
  - `_atom_inject_runtime_weights(self, atom_model)`
                                       OVERRIDE-OPTIONAL — extract lm_head / embed /
                                       final_ln into self.weight so RTP runtime can find
                                       them. Default handles the common naming.
  - `_atom_assert_norm_weights_loaded(self, atom_model)`
                                       OVERRIDE-OPTIONAL — guard against silently using
                                       default-initialized norm weights.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Type

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeights,
)
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.model_weight import W

from atom.model_loader.loader import WeightsMapper
from atom.plugin.rtpllm.models_v2.runtime import _ATOMRuntime

logger = logging.getLogger("atom.plugin.rtpllm.models_v2")


# ---------------------------------------------------------------------------
# No-op stand-ins for rtp-llm's loader machinery.
# In plugin mode ATOM owns the load path entirely, but BaseModel.load() and
# downstream rtp-llm code still expect these attributes to exist.
# ---------------------------------------------------------------------------


class _NoopWeightManager:
    def update(self, req):  # noqa: ANN001
        return None


class _NoopModelWeightsLoader:
    _py_eplb = None

    def load_lora_weights(self, adapter_name, lora_path, device):  # noqa: ANN001
        logger.warning(
            "No-op model_weights_loader received load_lora_weights(%s, %s, %s); "
            "plugin mode uses ATOM model weights path only.",
            adapter_name,
            lora_path,
            device,
        )
        return None


class _StubWeightInfo(ModelDeployWeightInfo):
    """Placeholder weight schema for ATOM-only models.

    Returned by the default get_weight_cls(); not exercised in plugin mode
    because we bypass create_model_loader/_load() entirely.
    """

    def _get_weight_info(self):  # noqa: D401
        return []


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ATOMRtpllmModelBase(BaseModel):
    """Generic ATOM model plugged into rtp-llm via BaseModel.

    Activate by setting:
        export RTP_LLM_EXTERNAL_MODEL_PACKAGES=atom.plugin.rtpllm.models_v2
        export MODEL_TYPE=<the type-string your subclass registered>
    """

    # ----- model-specific hooks (override in subclass) ---------------------

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:  # noqa: D401
        raise NotImplementedError("Subclass must implement _create_config")

    @staticmethod
    def get_weight_cls() -> Type[ModelDeployWeightInfo]:
        # Plugin mode never asks rtp-llm to load weights, so a stub is enough.
        return _StubWeightInfo

    def _atom_engine_name(self) -> str:
        return "rtpllm"

    def _atom_apply_pre_create_patches(self) -> None:
        """Apply RTP-specific monkey patches before ATOM model construction.

        Default: apply the attention backend patches that swap ATOM ops to the
        rtp-llm KV-cache-aware backend. Subclasses can extend (e.g. to add
        family-specific patches like qwen3_next).
        """
        from atom.plugin.rtpllm.attention_backend import (
            apply_attention_gdn_rtpllm_patch,
            apply_attention_mha_rtpllm_patch,
        )

        apply_attention_gdn_rtpllm_patch()
        apply_attention_mha_rtpllm_patch()

    def _atom_make_weights_mapper(self) -> WeightsMapper | None:
        return None

    def _atom_load_fused_expert_weights_fn(self):
        """Optional callable forwarded to ATOM's plugin loader.

        Return signature must match atom.model_loader.loader.load_model_in_plugin_mode's
        `load_fused_expert_weights_fn` parameter.
        """
        return None

    @staticmethod
    @contextmanager
    def _atom_loader_overrides(atom_model: Any):
        """Override-friendly context wrapping the actual checkpoint load call.

        Default is a passthrough. Subclasses can use this to e.g. disable
        rocm-aiter fused-shared-expert fusion when the checkpoint has standalone
        shared expert weights.
        """
        yield

    # ----- ATOM → rtp-llm weight bridging ----------------------------------

    @staticmethod
    def _first_named_param(module: Any, name: str) -> torch.Tensor | None:
        if module is None:
            return None
        for p_name, p in module.named_parameters(recurse=True):
            if p_name == name and p is not None:
                return p
        return None

    def _atom_inject_runtime_weights(self, atom_model: Any) -> None:
        """Make ATOM's lm_head / embedding / final_ln visible to rtp-llm runtime.

        Tries both the multimodal-style names (`language_model.*`) and the
        plain LM names (`model.*` / `lm_head.*`). Override if your model nests
        weights differently.
        """
        lm_head = self._first_named_param(atom_model, "language_model.lm_head.weight")
        if lm_head is None:
            lm_head = self._first_named_param(atom_model, "lm_head.weight")
        if lm_head is not None:
            self.weight.set_global_weight(W.lm_head, lm_head.detach())
            logger.info("Injected runtime lm_head: %s", tuple(lm_head.shape))
        else:
            logger.warning("No lm_head.weight found in ATOM model.")

        emb = self._first_named_param(atom_model, "language_model.model.embed_tokens.weight")
        if emb is None:
            emb = self._first_named_param(atom_model, "model.embed_tokens.weight")
        if emb is not None:
            self.weight.set_global_weight(W.embedding, emb.detach())
            logger.info("Injected runtime embedding: %s", tuple(emb.shape))

        norm = self._first_named_param(atom_model, "language_model.model.norm.weight")
        if norm is None:
            norm = self._first_named_param(atom_model, "model.norm.weight")
        if norm is not None:
            self.weight.set_global_weight(W.final_ln_gamma, norm.detach())
            logger.info("Injected runtime final_ln_gamma: %s", tuple(norm.shape))

    def _atom_assert_norm_weights_loaded(self, atom_model: Any) -> None:
        """Guard against silently using default-initialized GemmaRMSNorm weights."""
        candidates = [
            "language_model.model.layers.0.input_layernorm.weight",
            "model.layers.0.input_layernorm.weight",
        ]
        norm_w = None
        for name in candidates:
            norm_w = self._first_named_param(atom_model, name)
            if norm_w is not None:
                break
        if norm_w is None:
            raise ValueError(
                "Cannot locate layer-0 input_layernorm.weight after ATOM load. "
                "Override _atom_assert_norm_weights_loaded if your layer naming differs."
            )
        norm_cpu = norm_w.detach().float().reshape(-1).cpu()
        if norm_cpu.numel() == 0 or bool(torch.all(norm_cpu == 0)):
            raise ValueError(
                "Loaded layer-0 input_layernorm.weight is all zeros — "
                "checkpoint mapping/load mismatch."
            )

    # ----- rtp-llm BaseModel overrides -------------------------------------

    def load(self, skip_python_model: bool = False) -> None:
        """Replace rtp-llm native load with the ATOM plugin path.

        Unlike the v1 wrapper there is no native fallback — this class is
        intended for models rtp-llm has no implementation of.
        """
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
            logger.info("Plugin mode: skip ATOM python model creation as requested.")
            return

        self._create_python_model()
        logger.info(
            "Plugin mode load done for model_type=%s class=%s",
            self.model_config.model_type,
            type(self).__name__,
        )

    def _create_python_model(self):
        import atom
        from atom.model_loader.loader import load_model_in_plugin_mode

        target_device = torch.device(self.device if getattr(self, "device", None) else "cuda")
        target_dtype = self.model_config.compute_dtype

        old_default_dtype = torch.get_default_dtype()
        try:
            old_default_device = torch.get_default_device()
        except Exception:
            old_default_device = None

        torch.set_default_device(target_device)
        if target_dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            torch.set_default_dtype(target_dtype)

        try:
            self._atom_apply_pre_create_patches()

            atom_model = atom.prepare_model(config=self, engine=self._atom_engine_name())
            if atom_model is None:
                raise ValueError(
                    f"atom.prepare_model returned None for {type(self).__name__}"
                )
            atom_model = atom_model.to(target_device)

            atom_config = getattr(atom_model, "atom_config", None)
            if atom_config is None:
                atom_config = getattr(
                    getattr(atom_model, "language_model", None), "atom_config", None
                )
            if atom_config is None:
                raise ValueError(
                    "Cannot find atom_config on prepared ATOM model "
                    "(checked top-level and .language_model)."
                )

            with self._atom_loader_overrides(atom_model):
                load_model_in_plugin_mode(
                    model=atom_model,
                    config=atom_config,
                    prefix="model.",
                    weights_mapper=self._atom_make_weights_mapper(),
                    load_fused_expert_weights_fn=self._atom_load_fused_expert_weights_fn(),
                )

            self._atom_assert_norm_weights_loaded(atom_model)
            self._atom_inject_runtime_weights(atom_model)
        finally:
            torch.set_default_dtype(old_default_dtype)
            if old_default_device is not None:
                torch.set_default_device(old_default_device)
            else:
                torch.set_default_device("cpu")

        self.py_model = _ATOMRuntime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info(
            "Created ATOM runtime for %s (model_type=%s)",
            type(self).__name__,
            self.model_config.model_type,
        )
        return self.py_model


def is_atom_v2_plugin_loaded() -> bool:
    """True iff RTP_LLM_EXTERNAL_MODEL_PACKAGES asked for the v2 package."""
    return "atom.plugin.rtpllm.models_v2" in os.getenv(
        "RTP_LLM_EXTERNAL_MODEL_PACKAGES", ""
    )
