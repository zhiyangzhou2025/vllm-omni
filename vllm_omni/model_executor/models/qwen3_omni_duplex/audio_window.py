# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact embedding layout for one Qwen3-Omni-Duplex audio window."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_TOKENS_PER_UNIT,
    AUDIO_WINDOW_PROMPT_TOKENS,
    INPUT_SAMPLES_PER_UNIT,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
INPUT_SAMPLE_RATE_HZ = 16000


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """The nine embeddings and IDs consumed by one Stage 0 scheduler span."""

    input_ids: torch.Tensor
    embeddings: torch.Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1 or int(self.input_ids.shape[0]) != AUDIO_WINDOW_PROMPT_TOKENS:
            raise ValueError(f"audio window requires exactly {AUDIO_WINDOW_PROMPT_TOKENS} token IDs")
        if self.embeddings.ndim != 2 or int(self.embeddings.shape[0]) != AUDIO_WINDOW_PROMPT_TOKENS:
            raise ValueError(f"audio window requires exactly {AUDIO_WINDOW_PROMPT_TOKENS} embeddings")
        if self.input_ids.device != self.embeddings.device:
            raise ValueError("audio window token IDs and embeddings must share a device")


def select_window_span(
    window: AudioWindow,
    *,
    token_offset: int,
    span_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the scheduled slice of one exact window.

    Padding or truncation would change the Thinker KV length, so a span that
    does not fit the nine-token window is rejected.
    """

    if token_offset < 0 or span_length < 0:
        raise ValueError("token offset and span length must be non-negative")
    end = token_offset + span_length
    if end > AUDIO_WINDOW_PROMPT_TOKENS:
        raise ValueError(
            f"scheduled span [{token_offset}, {end}) exceeds the {AUDIO_WINDOW_PROMPT_TOKENS}-token audio window"
        )
    return window.input_ids[token_offset:end], window.embeddings[token_offset:end]


def build_audio_window(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    extract_features: Callable[[np.ndarray, int], tuple[torch.Tensor, int]],
    encode_audio: Callable[[torch.Tensor, int], torch.Tensor],
    device: torch.device,
) -> AudioWindow:
    """Build ``unit_start, audio_start, audio x 6, audio_end`` embeddings."""

    samples = _validated_waveform(waveform, sample_rate_hz)
    features, feature_length = extract_features(samples, sample_rate_hz)
    audio_embeddings = encode_audio(features, feature_length).to(device)
    if audio_embeddings.ndim != 2 or int(audio_embeddings.shape[0]) != AUDIO_TOKENS_PER_UNIT:
        raise ValueError(f"audio tower must return exactly {AUDIO_TOKENS_PER_UNIT} embeddings")

    boundary_ids = torch.tensor(
        [DUPLEX_TOKENS.unit_start, AUDIO_START_TOKEN_ID, AUDIO_END_TOKEN_ID],
        dtype=torch.long,
        device=device,
    )
    boundary_embeddings = embed_tokens(boundary_ids).to(device=device, dtype=audio_embeddings.dtype)
    if boundary_embeddings.ndim != 2 or int(boundary_embeddings.shape[0]) != 3:
        raise ValueError("token embedding must return one vector per boundary token")
    if int(boundary_embeddings.shape[1]) != int(audio_embeddings.shape[1]):
        raise ValueError("audio and token embeddings have different hidden sizes")

    input_ids = torch.tensor(
        [
            DUPLEX_TOKENS.unit_start,
            AUDIO_START_TOKEN_ID,
            *([DUPLEX_TOKENS.unit_start] * AUDIO_TOKENS_PER_UNIT),
            AUDIO_END_TOKEN_ID,
        ],
        dtype=torch.long,
        device=device,
    )
    embeddings = torch.cat(
        [
            boundary_embeddings[0:1],
            boundary_embeddings[1:2],
            audio_embeddings,
            boundary_embeddings[2:3],
        ],
        dim=0,
    )
    return AudioWindow(input_ids=input_ids, embeddings=embeddings)


def audio_window_from_duplex(
    duplex: Mapping[str, object],
    *,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    extract_features: Callable[[np.ndarray, int], tuple[torch.Tensor, int]],
    encode_audio: Callable[[torch.Tensor, int], torch.Tensor],
    device: torch.device,
) -> AudioWindow:
    """Decode the shared duplex append payload and build its audio window."""

    if duplex.get("data_plane") is not True or duplex.get("model_type") != "qwen3_omni_duplex":
        raise ValueError("payload is not a Qwen3-Omni-Duplex data-plane append")
    payload = duplex.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Qwen3-Omni-Duplex append requires an audio payload")
    audio = payload.get("audio")
    waveform = _coerce_waveform(audio)
    sample_rate = payload.get("sample_rate_hz", INPUT_SAMPLE_RATE_HZ)
    try:
        sample_rate_hz = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("audio sample rate must be an integer") from exc
    return build_audio_window(
        waveform=waveform,
        sample_rate_hz=sample_rate_hz,
        embed_tokens=embed_tokens,
        extract_features=extract_features,
        encode_audio=encode_audio,
        device=device,
    )


def _validated_waveform(waveform: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    if sample_rate_hz != INPUT_SAMPLE_RATE_HZ:
        raise ValueError(f"Qwen3-Omni-Duplex audio must be {INPUT_SAMPLE_RATE_HZ} Hz")
    samples = np.asarray(waveform)
    if samples.ndim != 1 or samples.shape != (INPUT_SAMPLES_PER_UNIT,):
        raise ValueError(f"Qwen3-Omni-Duplex audio window must contain exactly {INPUT_SAMPLES_PER_UNIT} samples")
    if not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("Qwen3-Omni-Duplex audio window must contain floating-point samples")
    return samples


def _coerce_waveform(audio: object) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio
    if isinstance(audio, bytes | bytearray | memoryview):
        return np.frombuffer(bytes(audio), dtype="<f4")
    raise ValueError("Qwen3-Omni-Duplex audio payload must be float32 PCM bytes or an array")
