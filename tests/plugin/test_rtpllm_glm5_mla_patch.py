"""Patch behavior tests for GLM5 RTP MLA forward."""

from types import SimpleNamespace

from atom.plugin.rtpllm.attention_backend.rtp_mla_prepare import (
    apply_deepseek_mla_rtpllm_patch,
)


class _FakeAttention:
    def forward(self, positions, hidden_states):
        return ("original", positions, hidden_states)


def test_apply_deepseek_mla_patch_marks_forward_and_keeps_wrapped():
    apply_deepseek_mla_rtpllm_patch(_FakeAttention)

    assert getattr(_FakeAttention.forward, "_rtpllm_patched") is True
    assert _FakeAttention.forward.__wrapped__ is not None


def test_apply_deepseek_mla_patch_is_idempotent():
    apply_deepseek_mla_rtpllm_patch(_FakeAttention)
    first_forward = _FakeAttention.forward
    first_wrapped = first_forward.__wrapped__

    apply_deepseek_mla_rtpllm_patch(_FakeAttention)

    assert _FakeAttention.forward is first_forward
    assert _FakeAttention.forward.__wrapped__ is first_wrapped


def test_patched_forward_calls_plugin_mode():
    from atom.plugin.rtpllm.attention_backend import rtp_mla_prepare

    calls = []

    def _fake_plugin_mode(attn, positions, hidden_states, **kwargs):
        calls.append((attn, positions, hidden_states, kwargs))
        return "patched"

    original = rtp_mla_prepare.forward_rtp_plugin_mode
    rtp_mla_prepare.forward_rtp_plugin_mode = _fake_plugin_mode
    try:
        apply_deepseek_mla_rtpllm_patch(_FakeAttention)
        result = _FakeAttention().forward("pos", "hidden")
    finally:
        rtp_mla_prepare.forward_rtp_plugin_mode = original

    assert result == "patched"
    assert len(calls) == 1
    assert isinstance(calls[0][0], _FakeAttention)
    assert calls[0][1:] == ("pos", "hidden", {})

