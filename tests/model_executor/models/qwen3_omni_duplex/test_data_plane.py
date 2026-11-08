# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 projects a finished window from the one token the engine received."""

from types import SimpleNamespace

from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.data_plane import (
    Qwen3OmniDuplexDataPlane,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def _result(token_ids: list[int], *, finished: bool = False) -> dict[str, object]:
    output = SimpleNamespace(
        request_id="req",
        stage_id=0,
        finished=finished,
        outputs=[SimpleNamespace(text="", token_ids=token_ids)],
    )
    return {"data_plane_outputs": [output]}


def test_listen_window_projects_as_listen() -> None:
    (projected,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.listen]))

    assert projected["is_listen"] is True
    assert projected["model_listen"] is True
    assert projected["end_of_turn"] is False


def test_turn_end_window_projects_as_end_of_turn() -> None:
    # The protocol samples turn_eos, then constrains chunk_eos as the stop, so
    # the turn-end window's last two tokens are that pair.
    (projected,) = Qwen3OmniDuplexDataPlane().project(
        _result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.chunk_eos])
    )

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is True


def test_chunk_end_window_projects_as_neither() -> None:
    # chunk_eos closes the window but not the turn, and it carries no speech.
    (projected,) = Qwen3OmniDuplexDataPlane().project(
        _result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.chunk_eos])
    )

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is False


def test_cumulative_tokens_keep_the_turn_end_decision() -> None:
    # The engine repeats earlier windows' tokens; only the last two decide.
    (projected,) = Qwen3OmniDuplexDataPlane().project(
        _result(
            [
                DUPLEX_TOKENS.listen,
                DUPLEX_TOKENS.speak,
                100,
                DUPLEX_TOKENS.turn_eos,
                DUPLEX_TOKENS.chunk_eos,
            ]
        )
    )

    assert projected["end_of_turn"] is True
