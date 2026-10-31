# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 Thinker execution for Qwen3-Omni-Duplex."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.audio_window import (
    audio_window_from_duplex,
    select_window_span,
)


class Qwen3OmniDuplexForConditionalGeneration(Qwen3OmniMoeForConditionalGeneration):
    """Qwen3-Omni runtime with exact duplex audio windows on Stage 0."""

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.model_stage == "thinker":
            self.has_preprocess = True
            self.set_custom_preprocess(self.thinker_preprocess)

    def thinker_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **info: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Replace one scheduled prefill span with the exact audio window."""

        duplex = info.get("duplex")
        if not isinstance(duplex, Mapping) or duplex.get("data_plane") is not True:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}

        prompt_len = _optional_int(info.get("_omni_prompt_len"), info.get("duplex_prompt_len"))
        token_offset = _optional_int(info.get("_omni_num_computed_tokens"), info.get("duplex_token_offset")) or 0
        if prompt_len is not None and token_offset >= prompt_len:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}

        window = audio_window_from_duplex(
            duplex,
            embed_tokens=self._embed_boundary_tokens,
            extract_features=self._extract_audio_features,
            encode_audio=self._encode_audio_features,
            device=input_ids.device,
        )
        span_ids, span_embeds = select_window_span(
            window,
            token_offset=token_offset,
            span_length=int(input_ids.shape[0]),
        )
        if input_embeds is not None:
            span_embeds = span_embeds.to(dtype=input_embeds.dtype)
        return span_ids, span_embeds, {}

    def _embed_boundary_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.thinker.embed_input_ids(token_ids)

    def _extract_audio_features(self, samples: Any, sample_rate_hz: int) -> tuple[torch.Tensor, int]:
        processor = self._duplex_feature_extractor()
        features = processor(
            samples,
            sampling_rate=sample_rate_hz,
            return_tensors="pt",
            return_attention_mask=True,
        )
        mask = features.attention_mask.bool()
        valid = features.input_features.permute(0, 2, 1)[mask].permute(1, 0)
        return valid, int(mask.sum())

    def _encode_audio_features(self, features: torch.Tensor, feature_length: int) -> torch.Tensor:
        device = features.device
        if device.type == "cpu" and hasattr(self.thinker, "audio_tower"):
            device = next(self.thinker.audio_tower.parameters()).device
        features = features.to(device=device, dtype=torch.get_default_dtype())
        lengths = torch.tensor([feature_length], dtype=torch.long, device=device)
        encoded = self.thinker._process_audio_input(
            {
                "input_features": features,
                "audio_feature_lengths": lengths,
            }
        )
        if len(encoded) != 1:
            raise ValueError("Qwen3-Omni-Duplex audio window must encode as one item")
        return encoded[0]

    def _duplex_feature_extractor(self) -> Any:
        extractor = getattr(self, "_qwen3_omni_duplex_feature_extractor", None)
        if extractor is None:
            from vllm.transformers_utils.processor import cached_processor_from_config

            extractor = cached_processor_from_config(self.vllm_config.model_config).feature_extractor
            self._qwen3_omni_duplex_feature_extractor = extractor
        return extractor


def _optional_int(primary: object, fallback: object) -> int | None:
    value = primary if primary is not None else fallback
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
