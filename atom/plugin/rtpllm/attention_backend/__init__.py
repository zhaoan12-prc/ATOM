from .rtp_mla_attention import RTPMLAAttention, apply_attention_mla_rtpllm_patch
from .rtp_mla_metadata import (
    GLM5_RTP_BRIDGE_MODE,
    GLM5_RTP_BRIDGE_MODE_M0_DENSE,
    GLM5_RTP_OWNERSHIP,
    RTPMlaPluginMetadata,
)
from .rtp_mla_prepare import RTPMlaPrepareResult, build_m0_prepare_result


def __getattr__(name):
    if name in {"RTPAttention", "RTPFullAttention"}:
        from .rtp_full_attention import RTPAttention, RTPFullAttention

        return {"RTPAttention": RTPAttention, "RTPFullAttention": RTPFullAttention}[name]
    if name == "apply_attention_gdn_rtpllm_patch":
        from .attention_gdn import apply_attention_gdn_rtpllm_patch

        return apply_attention_gdn_rtpllm_patch
    if name == "apply_attention_mha_rtpllm_patch":
        from .attention_switch import apply_attention_mha_rtpllm_patch

        return apply_attention_mha_rtpllm_patch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RTPAttention",
    "RTPFullAttention",
    "RTPMLAAttention",
    "GLM5_RTP_BRIDGE_MODE",
    "GLM5_RTP_BRIDGE_MODE_M0_DENSE",
    "GLM5_RTP_OWNERSHIP",
    "RTPMlaPluginMetadata",
    "RTPMlaPrepareResult",
    "apply_attention_gdn_rtpllm_patch",
    "apply_attention_mha_rtpllm_patch",
    "apply_attention_mla_rtpllm_patch",
    "build_m0_prepare_result",
]

