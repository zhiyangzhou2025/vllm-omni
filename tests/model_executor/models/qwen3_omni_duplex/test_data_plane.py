# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 projects a finished window from the one token the engine received."""

from types import SimpleNamespace

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.data_plane import (
    Qwen3OmniDuplexDataPlane,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def _result(token_ids: list[int], *, finished: bool = False, text: str = "") -> dict[str, object]:
    # The session runs on DELTA output, so the projected step carries only the
    # token it sampled; the window lives in the cumulative view.
    completion = SimpleNamespace(text=text, token_ids=token_ids[-1:], cumulative_token_ids=token_ids)
    output = SimpleNamespace(request_id="req", stage_id=0, finished=finished, outputs=[completion])
    return {"data_plane_outputs": [output]}


def test_listen_window_projects_as_listen() -> None:
    (projected,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end]))

    assert projected["is_listen"] is True
    assert projected["model_listen"] is True
    assert projected["end_of_turn"] is False


def test_turn_end_window_projects_as_end_of_turn() -> None:
    # The protocol samples turn_eos, then constrains unit_end as the stop, so
    # the turn-end window's last two tokens are that pair.
    (projected,) = Qwen3OmniDuplexDataPlane().project(
        _result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end])
    )

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is True


def test_mid_turn_speak_window_projects_as_neither() -> None:
    # unit_end closes the window but not the turn.
    (projected,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end]))

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is False


def test_a_step_inside_a_window_decides_nothing() -> None:
    # Only the step that samples unit_end closes a window.
    (projected,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.speak, 100]))

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is False


def test_earlier_windows_do_not_disturb_the_decision() -> None:
    # One resumable request holds every window, so the cumulative view grows
    # across windows; only the last two tokens decide.
    (projected,) = Qwen3OmniDuplexDataPlane().project(
        _result(
            [
                DUPLEX_TOKENS.listen,
                DUPLEX_TOKENS.unit_end,
                DUPLEX_TOKENS.speak,
                100,
                DUPLEX_TOKENS.turn_eos,
                DUPLEX_TOKENS.unit_end,
            ]
        )
    )

    assert projected["end_of_turn"] is True


def test_an_asr_token_is_not_reported_as_the_assistant_speaking() -> None:
    # ``transcript`` windows transcribe the user between asr and asr_eos. The
    # tokens detokenize like any other text, so projecting them verbatim put the
    # user's words inside the assistant's transcript, e.g. a turn reading
    # "<user question> <assistant answer>" as one assistant utterance.
    (user,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.asr, 100], text="what the user said"))
    (assistant,) = Qwen3OmniDuplexDataPlane().project(
        _result(
            [DUPLEX_TOKENS.asr, 100, DUPLEX_TOKENS.asr_eos, DUPLEX_TOKENS.speak, 101],
            text="what the model answers",
        )
    )

    assert user["text"] == ""
    assert assistant["text"] == "what the model answers"


def test_the_window_is_tagged_with_the_session_model_turn() -> None:
    # The context is a dataclass, not a mapping. Reading it as a mapping tagged
    # every window with turn 0, and the session dropped each model turn after
    # the first because 0 was behind the turn it had already completed.
    window = _result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end])

    (inside,) = Qwen3OmniDuplexDataPlane().project(
        window, context=DuplexDataPlaneContext(turn_id=7, active_response_turn_id=5, response_format="pcm16")
    )
    (between,) = Qwen3OmniDuplexDataPlane().project(
        window, context=DuplexDataPlaneContext(turn_id=7, active_response_turn_id=None)
    )

    # A response is open, so the window belongs to the turn that response serves.
    assert inside["model_turn_id"] == 5
    assert inside["audio_format"] == "pcm16"
    # No response is open, so the window opens the session's current turn.
    assert between["model_turn_id"] == 7
