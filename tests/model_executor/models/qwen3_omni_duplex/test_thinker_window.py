# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.qwen3_omni_duplex.thinker_window import (
    ThinkerProtocolError,
    ThinkerWindowState,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def _state(**kwargs) -> ThinkerWindowState:
    values = {"asr_enabled": False, "max_generated_tokens": 8, "max_text_tokens": 3}
    values.update(kwargs)
    return ThinkerWindowState(tokens=DUPLEX_TOKENS, **values)


def test_listen_feeds_listen_and_unit_end() -> None:
    state = _state()
    step = state.observe(DUPLEX_TOKENS.listen)

    assert step.feed_token_ids == (DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end)
    assert step.closed
    assert not step.end_of_turn


def test_turn_eos_forces_a_chunk_eos_sample_before_the_window_closes() -> None:
    state = _state()
    speak = state.observe(DUPLEX_TOKENS.speak)
    text = state.observe(100)
    turn = state.observe(DUPLEX_TOKENS.turn_eos)
    chunk = state.observe(DUPLEX_TOKENS.chunk_eos)

    assert speak.feed_token_ids == (DUPLEX_TOKENS.speak,)
    assert text.text_token_id == 100
    # turn_eos is sampled by the runner and fed like any decode token; it ends
    # the turn but not the window.
    assert turn.feed_token_ids == (DUPLEX_TOKENS.turn_eos,)
    assert turn.end_of_turn
    assert not turn.closed
    # The forced chunk_eos is the window's stop token and closes it.
    assert chunk.feed_token_ids == (DUPLEX_TOKENS.chunk_eos, DUPLEX_TOKENS.unit_end)
    assert chunk.closed
    assert chunk.end_of_turn
    with pytest.raises(ThinkerProtocolError):
        state.observe(101)


def test_post_turn_eos_keeps_only_chunk_eos() -> None:
    state = _state()
    state.observe(DUPLEX_TOKENS.speak)
    state.observe(100)
    state.observe(DUPLEX_TOKENS.turn_eos)

    logits = state.control_logits({DUPLEX_TOKENS.chunk_eos: 1.0, DUPLEX_TOKENS.turn_eos: 2.0, 101: 9.0})

    assert logits == {DUPLEX_TOKENS.chunk_eos: 1.0}


def test_system2_action_order_and_reserved_text_rejection() -> None:
    state = _state(active_generation=True)
    state.observe(DUPLEX_TOKENS.speak)
    state.observe(DUPLEX_TOKENS.cancel)
    state.observe(DUPLEX_TOKENS.launch)

    assert state.actions == ("cancel", "launch")
    with pytest.raises(ThinkerProtocolError, match="reserved token"):
        state.observe(DUPLEX_TOKENS.listen)


def test_asr_budget_exposes_only_asr_eos() -> None:
    state = _state(asr_enabled=True, max_text_tokens=1)
    state.observe(DUPLEX_TOKENS.asr)
    state.observe(100)

    assert set(state.control_logits({DUPLEX_TOKENS.asr_eos: 2.0, 101: 5.0})) == {DUPLEX_TOKENS.asr_eos}


def test_control_logits_keep_only_listen_and_speak() -> None:
    logits = _state().control_logits(
        {DUPLEX_TOKENS.listen: 1.0, DUPLEX_TOKENS.speak: 2.0, 100: 10.0},
        listen_bias=0.5,
    )

    assert logits == {DUPLEX_TOKENS.listen: 1.5, DUPLEX_TOKENS.speak: 2.0}
