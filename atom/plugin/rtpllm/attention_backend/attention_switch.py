import logging

from atom.plugin.prepare import is_rtpllm

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.attention_switch")

_PATCHED = False
_ORIGINAL_ATTENTION_CLS = None


def _make_rtp_attention_dispatcher(mha_cls, mla_cls):
    """Build a dispatcher that picks MLA or MHA RTP adapter at construction.

    ATOM model code instantiates attention as either
      `Attention(..., use_mla=True, mla_modules=...)`  (DeepSeek/GLM MLA)
    or
      `Attention(..., num_heads=..., num_kv_heads=...)`  (Qwen / dense MHA).

    We patch `atom.model_ops.Attention` to this dispatcher in rtpllm plugin
    mode so the call site stays unchanged.
    """

    def _dispatch(*args, **kwargs):
        use_mla = bool(kwargs.get("use_mla", False))
        cls = mla_cls if use_mla else mha_cls
        return cls(*args, **kwargs)

    return _dispatch


def apply_attention_mha_rtpllm_patch() -> None:
    """Switch ATOM Attention to RTP-style adapters for rtpllm plugin mode."""

    global _PATCHED, _ORIGINAL_ATTENTION_CLS
    if _PATCHED:
        return

    import atom.model_ops as ops
    from .rtp_full_attention import RTPFullAttention
    from .rtp_mla_attention import RTPMlaAttention

    if _ORIGINAL_ATTENTION_CLS is None:
        _ORIGINAL_ATTENTION_CLS = getattr(ops, "Attention", None)

    if is_rtpllm():
        ops.RTPFullAttention = RTPFullAttention
        ops.RTPMlaAttention = RTPMlaAttention
        ops.Attention = _make_rtp_attention_dispatcher(
            mha_cls=RTPFullAttention, mla_cls=RTPMlaAttention
        )
        logger.info(
            "Applied RTP-LLM attention patch: atom.model_ops.Attention now "
            "dispatches to RTPFullAttention (MHA) or RTPMlaAttention (MLA)."
        )

    _PATCHED = True
