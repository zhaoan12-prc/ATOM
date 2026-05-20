"""Static guards for the GLM5 rtp-llm plugin path."""

from pathlib import Path


_ATOM_ROOT = Path(__file__).resolve().parents[2]


def _read_plugin_file(relative_path: str) -> str:
    return (_ATOM_ROOT / relative_path).read_text()


def test_glm5_wrapper_does_not_use_mha_or_qwen_patches():
    source = _read_plugin_file("atom/plugin/rtpllm/models/glm5.py")

    assert "RTPFullAttention" not in source
    assert "apply_attention_mha_rtpllm_patch" not in source
    assert "apply_attention_gdn_rtpllm_patch" not in source
    assert "apply_qwen3_next_rtpllm_patch" not in source


def test_glm5_wrapper_does_not_reference_deepseek_mla_patch():
    source = _read_plugin_file("atom/plugin/rtpllm/models/glm5.py")

    assert "apply_deepseek_mla_rtpllm_patch" not in source


def test_rtp_mla_prepare_does_not_keep_native_forward_mirror_helpers():
    assert not (
        _ATOM_ROOT / "atom/plugin/rtpllm/attention_backend/rtp_mla_prepare.py"
    ).exists()


def test_glm5_mla_backend_is_not_full_attention_adapter():
    source = _read_plugin_file("atom/plugin/rtpllm/attention_backend/rtp_mla_attention.py")

    assert "class RTPMLAAttention" in source
    assert "use_mla" in source
    assert "RTPFullAttention" not in source

