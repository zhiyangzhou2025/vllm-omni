# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    MAX_GENERATED_TOKENS_PER_WINDOW,
    MAX_TEXT_TOKENS_PER_WINDOW,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.thinker_window import (
    ThinkerProtocolError,
    ThinkerWindowState,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def _state(**kwargs) -> ThinkerWindowState:
    values = {"user_signal": "none", "max_generated_tokens": 8, "max_text_tokens": 3}
    values.update(kwargs)
    return ThinkerWindowState(tokens=DUPLEX_TOKENS, **values)


def test_listen_keeps_the_window_open_for_unit_end() -> None:
    state = _state()
    listen = state.observe(DUPLEX_TOKENS.listen)

    assert listen.phase == "post_listen"
    assert not listen.closed
    # Only unit_end can follow, and it is the window's stop token.
    assert state.control_logits({DUPLEX_TOKENS.unit_end: 1.0, DUPLEX_TOKENS.listen: 9.0}) == {
        DUPLEX_TOKENS.unit_end: 1.0
    }
    unit_end = state.observe(DUPLEX_TOKENS.unit_end)
    assert unit_end.closed
    assert not unit_end.end_of_turn


def test_turn_eos_forces_a_unit_end_sample_before_the_window_closes() -> None:
    state = _state()
    state.observe(DUPLEX_TOKENS.speak)
    text = state.observe(100)
    turn = state.observe(DUPLEX_TOKENS.turn_eos)
    unit_end = state.observe(DUPLEX_TOKENS.unit_end)

    assert text.text_token_id == 100
    # turn_eos is sampled by the runner and fed like any decode token; it ends
    # the turn but not the window.
    assert turn.end_of_turn
    assert not turn.closed
    # The forced unit_end is the window's stop token and closes it.
    assert unit_end.closed
    assert unit_end.end_of_turn
    with pytest.raises(ThinkerProtocolError):
        state.observe(101)


def test_post_turn_eos_keeps_only_unit_end() -> None:
    state = _state()
    state.observe(DUPLEX_TOKENS.speak)
    state.observe(100)
    state.observe(DUPLEX_TOKENS.turn_eos)

    logits = state.control_logits({DUPLEX_TOKENS.unit_end: 1.0, DUPLEX_TOKENS.turn_eos: 2.0, 101: 9.0})

    assert logits == {DUPLEX_TOKENS.unit_end: 1.0}


def test_activity_signal_opens_the_window_and_hands_over_to_control() -> None:
    state = _state(user_signal="activity")

    assert state.control_logits({DUPLEX_TOKENS.idle: 1.0, DUPLEX_TOKENS.non_idle: 2.0, 100: 9.0}) == {
        DUPLEX_TOKENS.idle: 1.0,
        DUPLEX_TOKENS.non_idle: 2.0,
    }
    assert state.observe(DUPLEX_TOKENS.non_idle).phase == "control"
    with pytest.raises(ThinkerProtocolError, match="control must be"):
        state.observe(DUPLEX_TOKENS.non_idle)


def test_chunk_eos_is_masked_in_every_phase() -> None:
    logits = {
        DUPLEX_TOKENS.chunk_eos: 9.0,
        DUPLEX_TOKENS.unit_end: 1.0,
        DUPLEX_TOKENS.turn_eos: 1.0,
        DUPLEX_TOKENS.listen: 1.0,
        DUPLEX_TOKENS.speak: 1.0,
        100: 1.0,
    }
    speak = _state()
    assert DUPLEX_TOKENS.chunk_eos not in speak.control_logits(logits)
    speak.observe(DUPLEX_TOKENS.speak)
    assert DUPLEX_TOKENS.chunk_eos not in speak.control_logits(logits)
    speak.observe(100)
    assert DUPLEX_TOKENS.chunk_eos not in speak.control_logits(logits)
    with pytest.raises(ThinkerProtocolError, match="reserved token"):
        speak.observe(DUPLEX_TOKENS.chunk_eos)


def test_system2_action_order_and_reserved_text_rejection() -> None:
    state = _state(active_generation=True)
    state.observe(DUPLEX_TOKENS.speak)
    state.observe(DUPLEX_TOKENS.cancel)
    state.observe(DUPLEX_TOKENS.launch)

    assert state.actions == ("cancel", "launch")
    with pytest.raises(ThinkerProtocolError, match="reserved token"):
        state.observe(DUPLEX_TOKENS.listen)


def test_asr_budget_exposes_only_asr_eos() -> None:
    state = _state(user_signal="transcript", max_text_tokens=1)
    state.observe(DUPLEX_TOKENS.asr)
    state.observe(100)

    assert set(state.control_logits({DUPLEX_TOKENS.asr_eos: 2.0, 101: 5.0})) == {DUPLEX_TOKENS.asr_eos}


def test_the_shipped_budgets_let_the_tightest_window_spend_all_its_text() -> None:
    # ``transcript`` is the tightest mode: five protocol tokens can precede the
    # text phase and ``<|unit_end|>`` follows it. A text budget set just below
    # the generation budget would make that prefix eat into the text, closing
    # the window early and truncating the turn.
    state = ThinkerWindowState(
        tokens=DUPLEX_TOKENS,
        user_signal="transcript",
        max_generated_tokens=MAX_GENERATED_TOKENS_PER_WINDOW,
        max_text_tokens=MAX_TEXT_TOKENS_PER_WINDOW,
        active_generation=True,
    )
    for token_id in (
        DUPLEX_TOKENS.asr,
        DUPLEX_TOKENS.asr_eos,
        DUPLEX_TOKENS.speak,
        DUPLEX_TOKENS.cancel,
        DUPLEX_TOKENS.launch,
    ):
        state.observe(token_id)

    for _ in range(MAX_TEXT_TOKENS_PER_WINDOW):
        # Text stays drawable for every budgeted token.
        assert 100 in state.control_logits({100: 1.0, DUPLEX_TOKENS.unit_end: 2.0})
        state.observe(100)

    assert state.assistant_text_tokens == MAX_TEXT_TOKENS_PER_WINDOW
    assert state.observe(DUPLEX_TOKENS.unit_end).closed


def test_control_logits_keep_only_listen_and_speak() -> None:
    logits = _state().control_logits(
        {DUPLEX_TOKENS.listen: 1.0, DUPLEX_TOKENS.speak: 2.0, 100: 10.0},
        listen_bias=0.5,
    )

    assert logits == {DUPLEX_TOKENS.listen: 1.5, DUPLEX_TOKENS.speak: 2.0}
