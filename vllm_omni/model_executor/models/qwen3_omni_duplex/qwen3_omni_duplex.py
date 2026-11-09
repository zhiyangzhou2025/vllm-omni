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
from vllm_omni.model_executor.models.qwen3_omni_duplex.artifact import read_user_signal
from vllm_omni.model_executor.models.qwen3_omni_duplex.audio_window import (
    AUDIO_WINDOW_PROMPT_TOKENS,
    audio_window_from_duplex,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    Stage0Windows,
    ThinkerProtocolError,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKEN_IDS

# Stands in for every text token when asking a window what its phase allows.
# Any id outside the protocol set works; the window never reads its value.
_TEXT_PROBE_TOKEN_ID = 0


class Qwen3OmniDuplexForConditionalGeneration(Qwen3OmniMoeForConditionalGeneration):
    """Qwen3-Omni runtime with exact duplex audio windows on Stage 0."""

    # The runner only calls prepare_duplex_sampling for models that opt in.
    prefer_model_sampler = True

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The parent exposes talker_mtp for every stage, and the runner enables
        # its MTP path whenever the attribute is a method. Stage 0 is the
        # thinker: it has no talker, and that path would demand mtp_inputs the
        # thinker preprocess never produces.
        if self.model_stage == "thinker":
            # Reading the manifest here rejects a pre-v3 checkpoint at startup
            # rather than once per session.
            self._stage0_windows = Stage0Windows(read_user_signal(self.vllm_config.model_config.model))
            self.talker_mtp = None
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
        # Tokens before the window are the system prefix or the previous
        # window's <|unit_end|>. They keep their own embeddings.
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
        """Constrain, then let the runner's sampler draw the token.

        The runner calls this only because ``prefer_model_sampler`` is set, and
        that flag exists so the runner invokes ``prepare_duplex_sampling`` before
        sampling. The constraint is already applied to ``logits`` by then. The
        thinker has no sampler of its own: the parent forwards to the stage
        module, which does not implement one. Returning ``None`` is the runner's
        contract for declining the step, and it then samples with its own sampler
        from the masked logits.

        ``_sample_with_parent`` stays so tests can substitute a deterministic
        draw. A substitute that returns tokens advances the window here, because
        no runner calls ``post_sample_talker_mtp`` with those tokens afterwards.
        """

        output = self._sample_with_parent(logits, sampling_metadata)
        if output is None:
            return None
        self._advance_windows(getattr(output, "sampled_token_ids", None), self._sampling_request_ids(sampling_metadata))
        return output

    def post_sample_talker_mtp(
        self,
        *,
        req_ids: list[str],
        req_infos: list[dict[str, Any]],
        input_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Advance each window with the token the runner sampled.

        The runner calls this after its own sampler and passes the sampled token
        per request as ``input_ids``. That is the only place the sampled token
        reaches the model, so the window advances here. The returned rows are
        empty: the next append reserves its own ``<|unit_end|>`` slot, so nothing
        has to travel back through the engine.
        """

        del req_infos, kwargs
        self._advance_windows(input_ids, req_ids)
        return torch.full((len(req_ids), 1), -1, dtype=torch.long)

    def _sample_with_parent(self, logits: torch.Tensor, sampling_metadata: object) -> object:
        # The runner samples from the masked logits when this returns None.
        del logits, sampling_metadata
        return None

    def _advance_windows(self, sampled: object, request_ids: object) -> None:
        """Move each window to the phase its sampled token produces.

        The sampled token is one column per request. A window that was not
        constrained this step, or that already closed, is left untouched.
        """

        if self.model_stage != "thinker" or not isinstance(sampled, torch.Tensor) or not sampled.ndim:
            return
        tokens = sampled.reshape(sampled.shape[0], -1) if sampled.ndim > 1 else sampled.unsqueeze(-1)
        for row_idx, request_id in enumerate(request_ids):
            if row_idx >= tokens.shape[0]:
                break
            token_id = int(tokens[row_idx, 0].item())
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

    @staticmethod
    def _sampling_request_ids(sampling_metadata: object) -> tuple[object, ...]:
        req_ids = getattr(getattr(sampling_metadata, "input_batch", None), "req_ids", ())
        return tuple(req_ids)

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

    The window is the last nine tokens of the append. Anything before it is the
    system prefix or the previous window's ``<|unit_end|>``, and a span that
    ends before the window belongs entirely to those.
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
    # so ask about one of them instead: a phase that keeps text keeps the probe,
    # and a phase that forces a protocol token drops it. Passing protocol tokens
    # alone cannot tell the two apart, which masked text out of every phase.
    values[_TEXT_PROBE_TOKEN_ID] = 0.0
    legal = set(window.constrain(values).keys())
    if _TEXT_PROBE_TOKEN_ID in legal:
        illegal = [token_id for token_id in present if token_id not in legal]
        if illegal:
            row_logits[illegal] = float("-inf")
        return
    masked = row_logits.clone()
    row_logits.fill_(float("-inf"))
    candidates = [token_id for token_id in legal if token_id < vocab]
    for token_id in candidates:
        row_logits[token_id] = masked[token_id]
    # The phase keeps only these few candidates, and ``<|unit_end|>`` is often
    # the only one. It is also the request's stop token, so an upstream
    # processor can mask it. The row would then be all ``-inf``, the sampler
    # would draw an arbitrary token, and the window would reject it with no
    # trace of why.
    if not bool(torch.isfinite(row_logits[candidates]).any()):
        raise ThinkerProtocolError(f"no drawable token left among {sorted(candidates)}")
