"""Generic rtp-llm runtime adapter for ATOM models.

This is the model-agnostic equivalent of `_ATOMQwen35MoeRuntime` from the v1
plugin (atom/plugin/rtpllm/models/qwen3_5.py). Any model wrapped by
ATOMRtpllmModelBase ends up driven through this runtime, regardless of family.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs

from atom.plugin.rtpllm.utils import RTPForwardContext

logger = logging.getLogger("atom.plugin.rtpllm.models_v2.runtime")


class _ATOMRuntime(GptModelBase):
    """Generic rtp-llm runtime adapter backed by an ATOM model.

    Compared to the v1 `_ATOMQwen35MoeRuntime`, this class is model-agnostic:
    it does not import any specific model family, so the same runtime can
    serve qwen3.5-moe, any future ATOM-only model, etc.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        atom_model: Any,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.model = atom_model
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            raise RuntimeError("ATOM model has no parameters; cannot determine device/dtype.")
        self._model_device = first_param.device
        self._model_dtype = first_param.dtype
        # Cache layer maps once to avoid per-forward model.modules() traversal.
        self._rtp_layer_maps = RTPForwardContext.collect_layer_maps(model=self.model)
        # aiter's rope_cached_positions_2c_fwd_inplace (used by MLA RoPE in
        # atom.models.deepseek_v2) is compiled with INT64 positions only and
        # rejects int32 with "does not support positions dtype Int". Non-MLA
        # paths (e.g. Qwen3.5-MoE) keep int32 — that's what v1 plugin uses.
        use_mla = bool(getattr(getattr(model_config, "attn_config", None), "use_mla", False))
        self._position_dtype = torch.int64 if use_mla else torch.int32

    def load_weights(self):
        # ATOM weights are loaded exactly once from ATOMRtpllmModelBase._create_python_model.
        return None

    def initialize(self, init_resource: Any) -> bool:
        ok = super().initialize(init_resource)
        self._wire_dsa_indexer_caches(init_resource)
        return ok

    def _wire_dsa_indexer_caches(self, init_resource: Any) -> None:
        # DeepSeek-V3.2-style DSA models (e.g. GLM-5) create per-layer
        # `DeepseekV32IndexerCache` modules whose `kv_cache[0]` is the
        # indexer K cache buffer. In vLLM plugin mode this slot is filled by
        # vLLM's static_forward_context wiring; rtp-llm has no equivalent, so
        # we inject the per-layer scale buffer that SingleConfigCreator sized
        # specifically for the indexer (kv_scale_stride_bytes =
        # (indexer_dim + indexer_dim/128 * 4) * seq_size_per_block when
        # attn_config.is_sparse is True). Without this the buffer stays
        # empty and `kv_cache.view(-1, runner_block_size, kv_cache.shape[-1])`
        # in atom.models.deepseek_v2.sparse_attn_indexer fails with
        # "cannot reshape tensor of 0 elements into shape [-1, 16, 0]".
        kv_cache = getattr(init_resource, "kv_cache", None)
        if kv_cache is None:
            return
        scale_by_layer = getattr(kv_cache, "kv_scale_base_by_layer", None)
        if not scale_by_layer:
            return

        try:
            from atom.models.deepseek_v2 import DeepseekV32IndexerCache
        except ImportError:
            return

        wired = 0
        for module in self.model.modules():
            if not isinstance(module, DeepseekV32IndexerCache):
                continue
            layer_idx = self._parse_layer_idx_from_prefix(getattr(module, "prefix", ""))
            if layer_idx is None or layer_idx >= len(scale_by_layer):
                logger.warning(
                    "Skipping indexer cache wiring: prefix=%r layer_idx=%s scale_layers=%d",
                    getattr(module, "prefix", ""),
                    layer_idx,
                    len(scale_by_layer),
                )
                continue
            raw = scale_by_layer[layer_idx]
            if raw is None or raw.numel() == 0:
                logger.warning(
                    "Indexer scale buffer for layer %d is empty; sparse attention will fail.",
                    layer_idx,
                )
                continue
            head_dim = int(module.head_dim)
            byte_view = raw.view(torch.uint8).contiguous().view(-1)
            if int(byte_view.numel()) % head_dim != 0:
                logger.warning(
                    "Indexer scale buffer for layer %d not aligned to head_dim=%d (numel=%d).",
                    layer_idx,
                    head_dim,
                    int(byte_view.numel()),
                )
                continue
            shaped = byte_view.view(-1, head_dim)
            module.kv_cache = [shaped]
            wired += 1

        logger.info("Wired DeepseekV32IndexerCache buffers for %d layers.", wired)

    @staticmethod
    def _parse_layer_idx_from_prefix(prefix: str) -> int | None:
        # Expected prefix: "model.layers.{idx}.self_attn.indexer.k_cache".
        if not prefix:
            return None
        parts = prefix.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False) -> Any:
        # ATOM models drive attention through RTPAttention via RTPForwardContext;
        # rtp-llm's AttnImplFactory is bypassed entirely. ROCm has no MLA impls
        # registered in PREFILL_MLA_IMPS/DECODE_MLA_IMPS, so the default
        # GptModelBase.prepare_fmha_impl would raise "can not find mla type"
        # for sparse-MLA models like GLM-5. forward() ignores fmha_impl anyway.
        return None

    def _get_model_device(self) -> torch.device:
        return self._model_device

    def _get_model_dtype(self) -> torch.dtype:
        return self._model_dtype

    @staticmethod
    def _get_token_num(inputs: PyModelInputs, input_ids: torch.Tensor | None) -> int:
        if input_ids is not None and input_ids.numel() > 0:
            return int(input_ids.numel())
        if inputs.input_hiddens is not None and inputs.input_hiddens.numel() > 0:
            return int(inputs.input_hiddens.shape[0])
        return 0

    @staticmethod
    def _build_positions_from_attention_inputs(
        attn_inputs: Any, model_device: torch.device
    ) -> torch.Tensor | None:
        if attn_inputs is None:
            return None

        input_lengths = getattr(attn_inputs, "input_lengths", None)
        if input_lengths is None or input_lengths.numel() == 0:
            return None
        input_lengths_i32 = input_lengths.to(
            device=model_device, dtype=torch.int32, non_blocking=True
        ).contiguous()

        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if is_prefill:
            prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
            if prefix_lengths is None or prefix_lengths.numel() == 0:
                return None
            prefix_lengths_i32 = prefix_lengths.to(
                device=model_device, dtype=torch.int32, non_blocking=True
            ).contiguous()
            if int(prefix_lengths_i32.numel()) < int(input_lengths_i32.numel()):
                return None
            starts = prefix_lengths_i32[: int(input_lengths_i32.numel())]
        else:
            sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)
            if sequence_lengths is None or sequence_lengths.numel() == 0:
                return None
            sequence_lengths_i32 = sequence_lengths.to(
                device=model_device, dtype=torch.int32, non_blocking=True
            ).contiguous()
            if int(sequence_lengths_i32.numel()) < int(input_lengths_i32.numel()):
                return None
            starts = (
                sequence_lengths_i32[: int(input_lengths_i32.numel())]
                - input_lengths_i32
                + 1
            )

        token_starts = torch.repeat_interleave(starts, input_lengths_i32)
        if token_starts.numel() == 0:
            return None
        per_seq_base = input_lengths_i32.cumsum(dim=0) - input_lengths_i32
        token_ordinal = (
            torch.cumsum(
                torch.repeat_interleave(
                    torch.ones_like(input_lengths_i32), input_lengths_i32
                ),
                dim=0,
            )
            - 1
        )
        token_ordinal = token_ordinal - torch.repeat_interleave(
            per_seq_base, input_lengths_i32
        )
        return (token_starts + token_ordinal).to(dtype=torch.int32).contiguous()

    def _extract_combo_positions(
        self, inputs: PyModelInputs, model_device: torch.device
    ) -> torch.Tensor | None:
        bert_inputs = getattr(inputs, "bert_embedding_inputs", None)
        if bert_inputs is None:
            return None
        combo_position_ids = getattr(bert_inputs, "combo_position_ids", None)
        if combo_position_ids is None or combo_position_ids.numel() == 0:
            return None
        return combo_position_ids.to(
            device=model_device, dtype=self._position_dtype, non_blocking=True
        ).contiguous()

    def _extract_positions(
        self, inputs: PyModelInputs, model_device: torch.device, token_num: int
    ) -> torch.Tensor:
        attn_inputs = getattr(inputs, "attention_inputs", None)
        if attn_inputs is None:
            raise ValueError(
                "RTP plugin requires inputs.attention_inputs to provide position_ids."
            )
        positions = getattr(attn_inputs, "position_ids", None)
        if positions is None or positions.numel() == 0:
            positions = self._extract_combo_positions(inputs=inputs, model_device=model_device)
        if positions is None or positions.numel() == 0:
            positions = self._build_positions_from_attention_inputs(
                attn_inputs=attn_inputs, model_device=model_device
            )
        if positions is None or positions.numel() == 0:
            raise ValueError(
                "RTP plugin requires real position metadata from attention_inputs "
                "(position_ids or input/prefix/sequence lengths)."
            )
        positions = positions.to(
            device=model_device, dtype=self._position_dtype, non_blocking=True
        ).contiguous()
        pos_tokens = (
            int(positions.shape[-1]) if positions.dim() > 0 else int(positions.numel())
        )
        if token_num > 0 and pos_tokens != token_num:
            rebuilt = self._build_positions_from_attention_inputs(
                attn_inputs=attn_inputs, model_device=model_device
            )
            rebuilt_tokens = (
                int(rebuilt.shape[-1])
                if rebuilt is not None and rebuilt.dim() > 0
                else (int(rebuilt.numel()) if rebuilt is not None else -1)
            )
            if rebuilt is not None and rebuilt_tokens == token_num:
                positions = rebuilt.to(
                    device=model_device, dtype=self._position_dtype, non_blocking=True
                ).contiguous()
            elif pos_tokens > token_num:
                positions = positions[..., -token_num:].contiguous()
            else:
                raise ValueError(
                    "RTP plugin position_ids/token_num mismatch "
                    f"(position_ids_tokens={pos_tokens}, token_num={token_num})."
                )
        return positions

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        model_device = self._get_model_device()
        model_dtype = self._get_model_dtype()
        input_ids = inputs.input_ids
        inputs_embeds = None

        if input_ids is not None and input_ids.numel() > 0 and input_ids.device != model_device:
            input_ids = input_ids.to(device=model_device, non_blocking=True)

        token_num = self._get_token_num(inputs=inputs, input_ids=input_ids)
        positions = self._extract_positions(
            inputs=inputs, model_device=model_device, token_num=token_num
        )

        if input_ids is None or input_ids.numel() == 0:
            inputs_embeds = inputs.input_hiddens
            if (
                inputs_embeds is not None
                and inputs_embeds.numel() > 0
                and inputs_embeds.device != model_device
            ):
                inputs_embeds = inputs_embeds.to(device=model_device, non_blocking=True)
            if (
                inputs_embeds is not None
                and inputs_embeds.numel() > 0
                and inputs_embeds.dtype != model_dtype
            ):
                inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        with RTPForwardContext.bind(
            model=self.model,
            runtime=self,
            inputs=inputs,
            positions=positions,
            layer_maps=self._rtp_layer_maps,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=inputs_embeds,
            )
        return PyModelOutputs(hidden_states)
