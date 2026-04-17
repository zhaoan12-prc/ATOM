# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RTP-LLM attention backend adapter.

RTP plugin path currently reuses the same ATOM AITER attention backend
implementation used by sglang. Keep this wrapper so RTP-specific behavior can
be added without touching the sglang backend module.
"""

from atom.plugin.sglang.attention_backend.sgl_attn_backend import (
    ATOMAttnBackendForSgl,
)


class ATOMAttnBackendForRtp(ATOMAttnBackendForSgl):
    pass

