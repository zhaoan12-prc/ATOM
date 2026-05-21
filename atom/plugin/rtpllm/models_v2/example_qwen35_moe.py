"""Example: register qwen3.5-moe as a fully ATOM-owned rtp-llm model.

Unlike atom/plugin/rtpllm/models/qwen3_5.py, this file does NOT import
`rtp_llm.models.qwen3_next.qwen3_next.Qwen35Moe`. The class hierarchy is:

    ATOMQwen35MoeV2 → ATOMRtpllmModelBase → rtp_llm.models.base_model.BaseModel

so it works even if rtp-llm has no qwen3.5-moe implementation at all.

To use:
    export RTP_LLM_EXTERNAL_MODEL_PACKAGES=atom.plugin.rtpllm.models_v2
    export MODEL_TYPE=atom_qwen35_moe_v2

Use the same example as the template for any new ATOM-only model — copy this
file, change the registration string, and adjust _create_config / weights
mapper to match your checkpoint.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model

from atom.model_loader.loader import WeightsMapper
from atom.models.qwen3_5 import (
    detect_fused_expert_format,
    get_fused_expert_mapping,
    load_fused_expert_weights,
)
from atom.plugin.rtpllm.models_v2.atom_rtpllm_base import ATOMRtpllmModelBase
from atom.plugin.rtpllm.models_v2.config_helpers import (
    load_hf_config,
    parse_basic_into,
    parse_hybrid_attention_into,
    parse_linear_attention_into,
    parse_moe_into,
    parse_rmsnorm_into,
)

logger = logging.getLogger("atom.plugin.rtpllm.models_v2.example")


class ATOMQwen35MoeV2(ATOMRtpllmModelBase):
    """Qwen3.5-MoE driven entirely by ATOM, no rtp-llm Qwen class required."""

    # ---- HF config.json → ModelConfig (no rtp-llm Qwen class needed) ----

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_json = load_hf_config(ckpt_path, text_config_key="text_config")

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        parse_basic_into(config_json, config)
        parse_rmsnorm_into(config_json, config)
        cls._parse_rope(config_json, config)
        parse_moe_into(config_json, config)
        # Required for hybrid GDN/full-attn KV cache sizing.
        parse_hybrid_attention_into(config_json, config)
        parse_linear_attention_into(config_json, config)

        # Qwen3.5-MoE has qk_norm = True regardless of what the config says.
        config.qk_norm = True
        return config

    @staticmethod
    def _parse_rope(config_json: dict, config: ModelConfig) -> None:
        # Qwen3.5 stores rope params under a sub-dict.
        rope = config_json["rope_parameters"]
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = rope["rope_theta"]
        partial = rope["partial_rotary_factor"]
        config.partial_rotary_factor = partial
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * partial
        )

    # ---- ATOM-specific hooks --------------------------------------------

    def _atom_apply_pre_create_patches(self) -> None:
        super()._atom_apply_pre_create_patches()
        # Qwen3-Next-family patches; safe to apply for the MoE variant too.
        from atom.plugin.rtpllm.models.qwen3_next import apply_qwen3_next_rtpllm_patch

        apply_qwen3_next_rtpllm_patch()

    def _atom_make_weights_mapper(self) -> WeightsMapper:
        # Normalize checkpoint prefixes to ATOM's expected naming.
        return WeightsMapper(
            orig_to_new_substr={"attn.qkv.": "attn.qkv_proj."},
            orig_to_new_prefix={
                "model.language_model.model.": "model.language_model.",
                "model.language_model.lm_head.": "lm_head.",
            },
        )

    def _atom_load_fused_expert_weights_fn(self):
        def _fn(
            original_name: str,
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            if not detect_fused_expert_format(original_name):
                return False
            mapping = get_fused_expert_mapping()
            if not any(weight_name in original_name for _, weight_name, _ in mapping):
                return False
            return load_fused_expert_weights(
                original_name=original_name,
                name=name,
                params_dict=params_dict,
                loaded_weight=loaded_weight,
                shard_id=shard_id,
                num_experts=num_experts,
            )

        return _fn

    @staticmethod
    @contextmanager
    def _atom_loader_overrides(atom_model: Any):
        # If the checkpoint has standalone shared_expert weights, disable the
        # aiter fused shared-expert path during load so they get materialized
        # separately.
        has_standalone_shared_expert = any(
            ".shared_expert." in name for name, _ in atom_model.named_parameters()
        )
        if not has_standalone_shared_expert:
            yield
            return

        import atom.model_loader.loader as atom_loader

        origin = atom_loader.is_rocm_aiter_fusion_shared_expert_enabled
        atom_loader.is_rocm_aiter_fusion_shared_expert_enabled = lambda: False
        try:
            yield
        finally:
            atom_loader.is_rocm_aiter_fusion_shared_expert_enabled = origin


# Register two type strings:
#   "atom_qwen35_moe_v2"  — explicit selection
#   "qwen35_moe"          — only if you want to take over the built-in slot
# Registering the second one is OPTIONAL; uncomment only if you want this v2
# wrapper to claim the standard qwen35_moe name as well.
register_model("atom_qwen35_moe_v2", ATOMQwen35MoeV2, [])
# register_model("qwen35_moe", ATOMQwen35MoeV2, ["Qwen3_5MoeForConditionalGeneration"])
