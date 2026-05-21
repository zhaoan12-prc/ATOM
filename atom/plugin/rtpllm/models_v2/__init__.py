"""ATOM v2 rtp-llm plugin package.

This package is fully decoupled from rtp-llm's per-model class hierarchy.
Models live as siblings of `atom.plugin.rtpllm.models` and inherit from the
generic `ATOMRtpllmModelBase` (which itself inherits from `rtp_llm.models.base_model.BaseModel`).

Loaded via:
    RTP_LLM_EXTERNAL_MODEL_PACKAGES=atom.plugin.rtpllm.models_v2

(comma-joined with any other plugin packages you also want loaded.)

Add a new model by:
    1. Writing a subclass of ATOMRtpllmModelBase (see example_qwen35_moe.py)
    2. Calling register_model("your_model_type", YourClass, [...]) at module top level
    3. Importing it here so the registration side-effect fires
"""

# Importing pulls in the registration side-effect at the bottom of the file.
from atom.plugin.rtpllm.models_v2 import example_glm5_moe_dsa  # noqa: F401
from atom.plugin.rtpllm.models_v2 import example_qwen35_moe  # noqa: F401

__all__: list[str] = []
