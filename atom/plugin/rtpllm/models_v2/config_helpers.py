"""HF config.json → rtp-llm ModelConfig helpers.

These helpers exist so that a new ATOM-only model class does not need to
reimplement the full _parse_basic_config / _parse_rope_config / _parse_moe_config
chain that lives in rtp-llm per-family model files. They cover the common
fields; family-specific fields (mrope, hybrid attention, linear attention, etc.)
should be set by the subclass after calling parse_basic_into.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rtp_llm.config.model_config import ModelConfig


def load_hf_config(ckpt_path: str, text_config_key: str | None = None) -> dict[str, Any]:
    """Load ckpt_path/config.json; if `text_config_key` is given, descend into it.

    Some HF configs nest the language-model fields under a sub-key (e.g. Qwen3.5
    multimodal stores them under `text_config`). Pass that key here and you'll
    get the inner dict directly.
    """
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {ckpt_path}")
    with open(config_path) as f:
        cfg = json.loads(f.read())
    if text_config_key and text_config_key in cfg:
        cfg = cfg[text_config_key]
    return cfg


def parse_basic_into(config_json: dict, config: ModelConfig) -> None:
    """Fill the universally-required fields on `config` from HF config dict."""
    config.attn_config.head_num = config_json["num_attention_heads"]
    config.attn_config.kv_head_num = config_json.get(
        "num_key_value_heads", config_json["num_attention_heads"]
    )
    if "head_dim" in config_json:
        config.attn_config.size_per_head = config_json["head_dim"]
    else:
        config.attn_config.size_per_head = (
            config_json["hidden_size"] // config_json["num_attention_heads"]
        )
    config.num_layers = config_json["num_hidden_layers"]
    config.hidden_size = config_json["hidden_size"]
    config.vocab_size = config_json["vocab_size"]
    config.max_seq_len = config_json.get("max_position_embeddings", 4096)
    config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)


def parse_rmsnorm_into(config_json: dict, config: ModelConfig) -> None:
    """Common rmsnorm + SiGLU defaults (Qwen / Llama style)."""
    config.layernorm_eps = config_json.get("rms_norm_eps", 1e-6)
    config.norm_type = "rmsnorm"
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.qk_norm = bool(config_json.get("qk_norm", False))
    config.activation_type = "SiGLU"


def parse_rope_into(config_json: dict, config: ModelConfig) -> None:
    """Plain RoPE (style=1) parsing; subclasses with mrope/hybrid override."""
    rope_params = config_json.get("rope_parameters") or config_json
    config.attn_config.rope_config.style = 1
    config.attn_config.rope_config.base = rope_params.get(
        "rope_theta", config_json.get("rope_theta", 10000.0)
    )
    partial = rope_params.get("partial_rotary_factor", 1.0)
    config.partial_rotary_factor = partial
    config.attn_config.rope_config.dim = int(config.attn_config.size_per_head * partial)


def parse_moe_into(config_json: dict, config: ModelConfig) -> None:
    """MoE fields (Qwen-MoE / Mixtral style). Skip if model is dense."""
    if "num_experts" not in config_json:
        return
    config.moe_k = config_json["num_experts_per_tok"]
    config.expert_num = config_json["num_experts"]
    config.moe_inter_size = config_json["moe_intermediate_size"]
    if "shared_expert_intermediate_size" in config_json:
        config.inter_size = config_json["shared_expert_intermediate_size"]
        config.moe_style = 2
    else:
        config.moe_style = 1
    config.has_moe_norm = config_json.get("norm_topk_prob", True)

    moe_step = config_json.get("decoder_sparse_step", 1)
    config.moe_layer_index = [
        i for i in range(config.num_layers) if (i + 1) % moe_step == 0
    ]


def parse_hybrid_attention_into(config_json: dict, config: ModelConfig) -> None:
    """Hybrid attention layout (full-attn vs linear-attn per layer).

    Required so the rtp-llm KV cache allocator knows which layers are linear
    (GDN) and need an ssm/conv state buffer instead of the standard attention
    KV layout. Skip silently if the model is not hybrid.
    """
    if "full_attention_interval" not in config_json:
        return
    from rtp_llm.ops import HybridAttentionType

    step = config_json["full_attention_interval"]
    config.hybrid_attention_config.enable_hybrid_attention = True
    config.hybrid_attention_config.hybrid_attention_types = [
        HybridAttentionType.NONE if (i + 1) % step == 0 else HybridAttentionType.LINEAR
        for i in range(config.num_layers)
    ]


def parse_linear_attention_into(config_json: dict, config: ModelConfig) -> None:
    """Linear-attention (GDN) per-layer shape fields.

    Required for hybrid models with GDN layers — the C++ KV cache allocator
    uses these to size the per-layer ssm_state + conv_state buffers. Skip if
    the model has no linear-attention layers.
    """
    if "linear_conv_kernel_dim" not in config_json:
        return
    la = config.linear_attention_config
    la.linear_conv_kernel_dim = config_json["linear_conv_kernel_dim"]
    la.linear_key_head_dim = config_json["linear_key_head_dim"]
    la.linear_num_key_heads = config_json["linear_num_key_heads"]
    la.linear_num_value_heads = config_json["linear_num_value_heads"]
    la.linear_value_head_dim = config_json["linear_value_head_dim"]
