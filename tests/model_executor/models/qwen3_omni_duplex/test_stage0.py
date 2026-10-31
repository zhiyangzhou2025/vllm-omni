# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_WINDOW_PROMPT_TOKENS,
    build_audio_window_prompt,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def test_audio_window_reserves_exact_thinker_tokens() -> None:
    prompt = build_audio_window_prompt(
        request_id="request",
        fence=DuplexFence(session_id="session", epoch=0, turn_id=1),
        session_config={},
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload={"audio": b"pcm"},
        final=False,
    )

    assert prompt["prompt_token_ids"] == [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS
    assert len(prompt["prompt_token_ids"]) == 9
    duplex = prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["audio_tokens_per_unit"] == 6
    assert duplex["input_samples_per_unit"] == 7680
    assert duplex["scheduler_token_budget"] == 9


def test_audio_window_rejects_missing_audio() -> None:
    with pytest.raises(ValueError, match="audio payload"):
        build_audio_window_prompt(
            request_id="request",
            fence=DuplexFence(session_id="session", epoch=0, turn_id=1),
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload={},
            final=False,
        )
