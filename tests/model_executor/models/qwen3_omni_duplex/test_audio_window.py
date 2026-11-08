# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.audio_window import (
    AUDIO_END_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    audio_window_from_duplex,
    select_window_span,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def _embed(token_ids: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.tensor([float(token_id), 1.0]) for token_id in token_ids.tolist()])


def _extract(samples: np.ndarray, sample_rate_hz: int) -> tuple[torch.Tensor, int]:
    assert samples.shape == (7680,) and sample_rate_hz == 16000
    return torch.zeros(128, 48), 48


def _encode(features: torch.Tensor, feature_length: int) -> torch.Tensor:
    assert tuple(features.shape) == (128, 48) and feature_length == 48
    return torch.arange(12, dtype=torch.float32).reshape(6, 2)


def _window():
    return audio_window_from_duplex(
        {
            "data_plane": True,
            "model_type": "qwen3_omni_duplex",
            "payload": {"audio": np.zeros(7680, dtype=np.float32).tobytes(), "sample_rate_hz": 16000},
        },
        embed_tokens=_embed,
        extract_features=_extract,
        encode_audio=_encode,
        device=torch.device("cpu"),
    )


def test_audio_window_uses_exact_protocol_layout() -> None:
    window = _window()

    assert window.input_ids.tolist() == [
        DUPLEX_TOKENS.unit_start,
        AUDIO_START_TOKEN_ID,
        *([DUPLEX_TOKENS.unit_start] * 6),
        AUDIO_END_TOKEN_ID,
    ]
    expected = torch.tensor(
        [
            [float(DUPLEX_TOKENS.unit_start), 1.0],
            [float(AUDIO_START_TOKEN_ID), 1.0],
            *torch.arange(12, dtype=torch.float32).reshape(6, 2).tolist(),
            [float(AUDIO_END_TOKEN_ID), 1.0],
        ]
    )
    assert torch.equal(window.embeddings, expected)


def test_scheduled_span_rejects_positions_outside_window() -> None:
    window = _window()
    ids, embeddings = select_window_span(window, token_offset=7, span_length=2)

    assert ids.tolist() == [DUPLEX_TOKENS.unit_start, AUDIO_END_TOKEN_ID]
    assert embeddings.shape == (2, 2)
    with pytest.raises(ValueError, match="audio window"):
        select_window_span(window, token_offset=8, span_length=2)


def test_audio_window_decodes_base64_payload() -> None:
    # The duplex append carries the samples as base64 text. The engine never
    # decodes them, so the worker has to.
    import base64

    encoded = base64.b64encode(np.zeros(7680, dtype="<f4").tobytes()).decode()
    window = audio_window_from_duplex(
        {
            "data_plane": True,
            "model_type": "qwen3_omni_duplex",
            "payload": {"audio": encoded, "sample_rate_hz": 16000},
        },
        embed_tokens=_embed,
        extract_features=_extract,
        encode_audio=_encode,
        device=torch.device("cpu"),
    )

    assert window.input_ids.shape == (9,)
    with pytest.raises(ValueError, match="7680"):
        audio_window_from_duplex(
            {"data_plane": True, "model_type": "qwen3_omni_duplex", "payload": {"audio": b"1234"}},
            embed_tokens=_embed,
            extract_features=_extract,
            encode_audio=_encode,
            device=torch.device("cpu"),
        )
