# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RTP wrapper for DeepSeek MLA patch helpers.

Current RTP integration reuses the same DeepSeek patching implementation
introduced for sglang. This wrapper keeps RTP call sites framework-pure and
provides a dedicated extension point for future RTP-only behavior.
"""

from atom.plugin.sglang.attention_backend.sgl_attention_mla import (
    setup_deepseek_for_sglang,
)


def setup_deepseek_for_rtp(model):
    setup_deepseek_for_sglang(model)

