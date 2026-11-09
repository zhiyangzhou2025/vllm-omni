# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 projects a finished window from the one token the engine received."""

from types import SimpleNamespace

from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.data_plane import (
    Qwen3OmniDuplexDataPlane,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def _result(token_ids: list[int], *, finished: bool = False) -> dict[str, object]:
    # The session runs on DELTA output, so the projected step carries only the
    # token it sampled; the window lives in the cumulative view.
    completion = SimpleNamespace(text="", token_ids=token_ids[-1:], cumulative_token_ids=token_ids)
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
