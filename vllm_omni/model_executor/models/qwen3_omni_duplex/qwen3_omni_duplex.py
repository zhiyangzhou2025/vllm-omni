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
    AUDIO_WINDOW_PROMPT_TOKENS,
    audio_window_from_duplex,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    Stage0Windows,
    ThinkerProtocolError,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKEN_IDS

# The scheduler drops a finished chunk's sampled tokens, so the boundary tokens
# the protocol appends after the sampled one have to ride back to the engine
# and be replayed with the next append. The runner forwards exactly the keys a
# model names here, and only on the step the segment finishes.
FORCED_TOKEN_KEY = "duplex_forced_token_ids"


class Qwen3OmniDuplexForConditionalGeneration(Qwen3OmniMoeForConditionalGeneration):
    """Qwen3-Omni runtime with exact duplex audio windows on Stage 0."""

    # The runner only calls prepare_duplex_sampling for models that opt in.
    prefer_model_sampler = True
    omni_client_multimodal_output_keys = (FORCED_TOKEN_KEY,)

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._stage0_windows = Stage0Windows()
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
        span_length = int(input_ids.shape[0])
        # Decode steps carry one already-sampled token. Only a prefill span can
        # hold the audio window.
        if prompt_len is not None and token_offset >= prompt_len:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}
        window_start = _window_start(prompt_len, token_offset, span_length)
        if window_start is None:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}

        window = audio_window_from_duplex(
            duplex,
            embed_tokens=self._embed_boundary_tokens,
            extract_features=self._extract_audio_features,
            encode_audio=self._encode_audio_features,
            device=input_ids.device,
        )
        # Tokens before the window are the system prefix or forced boundary
        # tokens replayed from the previous window. They keep their embeddings.
        if input_embeds is not None and int(input_embeds.shape[0]) == span_length:
            embeds = input_embeds.clone()
        else:
            embeds = self.embed_input_ids(input_ids)
        ids = input_ids.clone()
        ids[window_start:] = window.input_ids
        embeds[window_start:] = window.embeddings.to(dtype=embeds.dtype)
        return ids, embeds, {}

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

    def prepare_duplex_sampling(self, logits: torch.Tensor, sampling_metadata: object, rows: object) -> None:
        """Mask every logit the current window phase does not allow.

        The runner calls this before ``sample`` and only for rows whose prompt
        carries a duplex payload. A window that has already closed has nothing
        to constrain: its request stops on the protocol token.
        """

        del sampling_metadata
        if self.model_stage != "thinker" or not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            return
        for row in rows:
            row_idx = getattr(row, "row_idx", -1)
            if row_idx < 0 or row_idx >= logits.shape[0]:
                continue
            window = self._stage0_windows.window_for(row.request_id)
            if window.closed:
                continue
            _mask_illegal_logits(logits[row_idx], window)

    def sample(self, logits: torch.Tensor, sampling_metadata: object) -> object:
        """Sample, then advance the window that produced the token.

        The forced boundary tokens are not fed here. The scheduler discards a
        finished chunk's sampled tokens, and the engine replays the protocol's
        boundary tokens with the next append.
        """

        output = self._sample_with_parent(logits, sampling_metadata)
        if self.model_stage != "thinker" or output is None:
            return output
        sampled = getattr(output, "sampled_token_ids", None)
        if not isinstance(sampled, torch.Tensor) or sampled.ndim != 2:
            return output
        req_ids = getattr(getattr(sampling_metadata, "input_batch", None), "req_ids", ())
        for row_idx, request_id in enumerate(req_ids):
            if row_idx >= sampled.shape[0]:
                break
            token_id = int(sampled[row_idx, 0].item())
            if token_id < 0:
                continue
            window = self._stage0_windows._windows.get(request_id)
            if window is None or window.closed:
                continue
            try:
                window.observe(token_id)
            except ThinkerProtocolError:
                # An illegal token means the constraint was bypassed. Drop the
                # window so the next one starts clean instead of wedging here.
                self._stage0_windows._windows.pop(request_id, None)
        return output

    def post_sample_talker_mtp(
        self,
        *,
        req_ids: list[str],
        req_infos: list[dict[str, Any]],
        **kwargs: object,
    ) -> torch.Tensor:
        """Report each closed window's replay tokens on the client channel.

        The runner calls this after sampling and merges the returned tensor's
        rows into the step's multimodal outputs, which is the only per-request
        channel that reaches the engine. A window that is still open reports an
        empty row; the runner drops empty rows before they leave the worker.
        """

        del req_infos, kwargs
        # A closed window replays one token after listen or chunk_eos, and two
        # after turn_eos. The row count has to equal the request count: the
        # runner slices a tensor per request only when its first dimension
        # matches the batch. -1 marks a window that is still open.
        rows = [self._closed_window_replay(request_id) or (-1,) for request_id in req_ids]
        width = max(len(row) for row in rows)
        codes = torch.full((len(req_ids), width), -1, dtype=torch.long)
        for row_idx, row in enumerate(rows):
            codes[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        return codes

    def finalize_multimodal_outputs_from_cpu_snapshot(self, snapshot: object) -> dict[str, object]:
        """Rename the replay rows to the key the engine reads.

        The runner stores this hook's return under ``codes.audio``. The engine
        only forwards keys the model names, so the rows have to come back under
        that name.
        """

        merged = dict(snapshot) if isinstance(snapshot, Mapping) else {}
        codes = merged.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else None
        if isinstance(audio, list):
            merged[FORCED_TOKEN_KEY] = audio
        return merged

    def _closed_window_replay(self, request_id: str) -> tuple[int, ...]:
        window = self._stage0_windows._windows.get(request_id)
        if window is None or not window.closed:
            return ()
        # The window is spent once its replay tokens are reported. The next
        # append starts a new one.
        del self._stage0_windows._windows[request_id]
        return window.forced_token_ids

    def _sample_with_parent(self, logits: torch.Tensor, sampling_metadata: object) -> object:
        return super().sample(logits, sampling_metadata)

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


def _window_start(prompt_len: int | None, token_offset: int, span_length: int) -> int | None:
    """Locate the audio window inside one scheduled prefill span.

    The window is the last nine tokens of the append. Anything before it was
    replayed from the previous window or is the system prefix, and a span that
    ends before the window belongs entirely to that prefix.
    """

    if prompt_len is None:
        return 0 if span_length == AUDIO_WINDOW_PROMPT_TOKENS else None
    window_abs_start = prompt_len - AUDIO_WINDOW_PROMPT_TOKENS
    if token_offset > window_abs_start or token_offset + span_length < prompt_len:
        return None
    return window_abs_start - token_offset


def _mask_illegal_logits(row_logits: torch.Tensor, window: object) -> None:
    """Keep the row's legal logits and set every other one to ``-inf``."""

    vocab = int(row_logits.shape[-1])
    present = [token_id for token_id in DUPLEX_TOKEN_IDS if token_id < vocab]
    values = {token_id: float(row_logits[token_id].item()) for token_id in present}
    # Text tokens are legal in the text phases and are not listed individually,
    # so a phase that keeps them keeps every non-protocol logit as well.
    legal = set(window.constrain(values).keys())
    if legal - set(present):
        illegal = [token_id for token_id in present if token_id not in legal]
        if illegal:
            row_logits[illegal] = float("-inf")
        return
    masked = row_logits.clone()
    row_logits.fill_(float("-inf"))
    for token_id in legal:
        if token_id < vocab:
            row_logits[token_id] = masked[token_id]
