"""Example: register GLM-5 (GlmMoeDsaForCausalLM) as a fully ATOM-owned rtp-llm model.

GLM-5 is structurally a DeepSeek-V3.2 derivative: MLA attention (q_lora /
kv_lora / nope+rope head dims) + DSA indexer + sigmoid-scored MoE with
shared experts. Its checkpoint at /mnt/raid1/pretrained_model/GLM-5-FP8 ships
the FP8 block-quantized weights for `GlmMoeDsaForCausalLM`.

Unlike rtp-llm's own DeepSeekV2 binding (rtp_llm/models/deepseek_v2.py),
this file does NOT inherit from any rtp-llm model class. The chain is:

    ATOMGlm5MoeDsaV2 → ATOMRtpllmModelBase → rtp_llm.models.base_model.BaseModel

so it works even if rtp-llm has no GLM-5 implementation. The actual forward
pass is run by ATOM's `atom.models.deepseek_v2.GlmMoeDsaForCausalLM`, picked
up through `atom.prepare_model` inside the base class.

To use:
    export RTP_LLM_EXTERNAL_MODEL_PACKAGES=atom.plugin.rtpllm.models_v2
    export MODEL_TYPE=atom_glm_moe_dsa_v2
    # ckpt: /mnt/raid1/pretrained_model/GLM-5-FP8
"""

from __future__ import annotations

import logging

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model

from atom.plugin.rtpllm.models_v2.atom_rtpllm_base import ATOMRtpllmModelBase
from atom.plugin.rtpllm.models_v2.config_helpers import (
    load_hf_config,
    parse_basic_into,
    parse_rmsnorm_into,
)

logger = logging.getLogger("atom.plugin.rtpllm.models_v2.glm5")


class ATOMGlm5MoeDsaV2(ATOMRtpllmModelBase):
    """GLM-5 (MLA + DSA + MoE) driven entirely by ATOM, no rtp-llm DeepSeek class required."""

    # ---- HF config.json → ModelConfig ------------------------------------

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_json = load_hf_config(ckpt_path)

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        parse_basic_into(config_json, config)
        parse_rmsnorm_into(config_json, config)
        # GLM-5 uses SiGLU like the helpers default; nothing to override.

        cls._parse_mla(config_json, config)
        cls._parse_rope(config_json, config)
        cls._parse_moe(config_json, config)
        cls._parse_dsa_indexer(config_json, config)

        return config

    # ---- MLA: q_lora / kv_lora / nope+rope head dims ----------------------

    @staticmethod
    def _parse_mla(config_json: dict, config: ModelConfig) -> None:
        """DeepSeek-V3.2 / GLM-5 MLA fields.

        Override `parse_basic_into`'s `size_per_head` because for MLA the
        effective per-head dim is `qk_nope_head_dim + qk_rope_head_dim`, not
        the HF `head_dim` (which on GLM-5 is 64 == rope-only).
        """
        config.attn_config.use_mla = True

        q_lora_rank = config_json.get("q_lora_rank")
        config.attn_config.q_lora_rank = (
            int(q_lora_rank) if q_lora_rank is not None else 0
        )
        kv_lora_rank = config_json.get("kv_lora_rank")
        config.attn_config.kv_lora_rank = (
            int(kv_lora_rank) if kv_lora_rank is not None else 0
        )

        config.attn_config.nope_head_dim = config_json["qk_nope_head_dim"]
        config.attn_config.rope_head_dim = config_json["qk_rope_head_dim"]
        config.attn_config.v_head_dim = config_json["v_head_dim"]
        config.attn_config.size_per_head = (
            config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
        )

    # ---- RoPE: simple style (GLM-5 has no rope_scaling) -------------------

    @staticmethod
    def _parse_rope(config_json: dict, config: ModelConfig) -> None:
        """GLM-5 ships plain RoPE (style=1) — no YARN block in config.json.

        rope_dim is the rope-only part of the head; rope is applied to the
        last `rope_head_dim` slice of qk, so offset = nope_head_dim.
        rope_interleave=True in the HF config → non-NeoX layout.
        """
        rope_params = config_json.get("rope_parameters") or config_json
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = rope_params.get("rope_theta", 10000.0)
        config.attn_config.rope_config.dim = config.attn_config.rope_head_dim
        config.attn_config.rope_config.offset = config.attn_config.nope_head_dim

        rope_interleave = config_json.get("rope_interleave", True)
        config.attn_config.rope_config.is_neox_style = not rope_interleave
        indexer_rope_interleave = config_json.get("indexer_rope_interleave", False)
        config.attn_config.rope_config.indexer_is_neox_style = (
            not indexer_rope_interleave
        )

    # ---- MoE: sigmoid-scored, shared experts, first-k dense layers --------

    @staticmethod
    def _parse_moe(config_json: dict, config: ModelConfig) -> None:
        """DeepSeek-style MoE (different key names than Qwen MoE).

        - `num_experts` → `n_routed_experts`
        - shared experts contribute `n_shared_experts * moe_inter_size` to
          the dense FFN slot (`inter_size`), and `moe_style = 2`.
        - sparse layers start at `first_k_dense_replace` and recur every
          `moe_layer_freq` layers.
        """
        scoring_func = config_json.get("scoring_func")
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1
        elif scoring_func is not None:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")

        config.routed_scaling_factor = config_json["routed_scaling_factor"]
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["n_routed_experts"]
        moe_intermediate_size = config_json["moe_intermediate_size"]
        config.moe_inter_size = moe_intermediate_size
        config.moe_n_group = config_json.get("n_group", 1)
        config.moe_topk_group = config_json.get("topk_group", 1)

        n_shared_experts = config_json.get("n_shared_experts", 0) or 0
        config.inter_size = n_shared_experts * moe_intermediate_size
        config.has_moe_norm = config_json.get("norm_topk_prob", False)
        config.moe_style = 2  # shared + expert

        moe_step = config_json["moe_layer_freq"]
        first_k_dense_replace = config_json["first_k_dense_replace"]
        config.moe_layer_index = [
            i
            for i in range(config.num_layers)
            if i >= first_k_dense_replace and i % moe_step == 0
        ]

    # ---- DSA: lightning indexer for sparse attention ----------------------

    @staticmethod
    def _parse_dsa_indexer(config_json: dict, config: ModelConfig) -> None:
        """V3.2-style sparse-attention indexer. Only present on DSA configs."""
        if config_json.get("index_topk") is None:
            return
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = config_json["index_head_dim"]
        config.attn_config.indexer_head_num = config_json["index_n_heads"]
        config.attn_config.indexer_topk = config_json["index_topk"]


# Explicit-only registration: `MODEL_TYPE=atom_glm_moe_dsa_v2` picks this up.
# We deliberately do NOT claim the HF arch `GlmMoeDsaForCausalLM` because
# rtp-llm's built-in `glm_5` (rtp_llm/models/deepseek_v2.py:819) already
# owns it, and double-claiming raises a conflict at import time that takes
# the whole external_model_packages loader down with it.
register_model("atom_glm_moe_dsa_v2", ATOMGlm5MoeDsaV2, [])
