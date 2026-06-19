"""Metadata and static contracts for GLM5 MLA in rtp-llm plugin mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

GLM5_RTP_MLA_MODE_DENSE = "dense"
GLM5_RTP_MLA_MODE = GLM5_RTP_MLA_MODE_DENSE


GLM5_RTP_OWNERSHIP = {
    "main_q_norm": "DeepseekV2MLAAttention",
    "main_kv_norm": "DeepseekV2MLAAttention",
    "main_rope": "RTPMLAAttention",
    "main_kv_cache": "RTPMLAAttention",
    "indexer_k_norm": "Indexer",
    "indexer_rope": "Indexer",
    "indexer_cache": "Indexer",
    "topk_selector": "Indexer",
}


@dataclass(frozen=True)
class RTPMlaPluginMetadata:
    """Metadata shared by GLM5 RTP MLA attention paths."""

    is_prefill: bool
    slot_mapping: Optional[torch.Tensor] = None
    block_table: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
