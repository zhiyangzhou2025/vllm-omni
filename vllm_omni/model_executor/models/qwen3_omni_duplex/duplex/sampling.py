# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Window protocol shared by the Stage 0 worker and the duplex engine.

The worker constrains sampling. The engine classifies a finished window from
the tokens it received, because the scheduler drops every sampled token of a
finished chunk and the worker has no channel back to the engine.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm_omni.model_executor.models.qwen3_omni_duplex.thinker_window import (
    ThinkerProtocolError,
    ThinkerWindowState,
    UserSignalMode,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

# Worst-case protocol steps a window spends before its first text token, plus
# the step that closes it. ``transcript`` is the tightest of the three user
# signal modes: ``<|asr|>``, ``<|asr_eos|>``, ``<|speak|>``, ``<|cancel|>`` and
# ``<|launch|>`` can all precede the text phase, and ``<|unit_end|>`` follows it.
WINDOW_PROTOCOL_OVERHEAD = 6
# The reference runtime's serving budgets. The generation budget must leave the
# text phase at least one step once the protocol prefix is paid for, so the text
# budget is not simply one below it.
MAX_GENERATED_TOKENS_PER_WINDOW = 16
MAX_TEXT_TOKENS_PER_WINDOW = 8


class Stage0Window:
    """One request's window, from its first sampled token to ``<|unit_end|>``."""

    def __init__(self, user_signal: UserSignalMode) -> None:
        self.state = ThinkerWindowState(
            tokens=DUPLEX_TOKENS,
            user_signal=user_signal,
            max_generated_tokens=MAX_GENERATED_TOKENS_PER_WINDOW,
            max_text_tokens=MAX_TEXT_TOKENS_PER_WINDOW,
        )
        self.closed = False

    def constrain(self, logits: dict[int, float]) -> dict[int, float]:
        return self.state.control_logits(logits)

    def observe(self, token_id: int) -> None:
        """Accept one sampled token and record whether it closed the window."""

        self.closed = self.state.observe(token_id).closed


class Stage0Windows:
    """Window states keyed by request, reset whenever a window closes."""

    def __init__(self, user_signal: UserSignalMode) -> None:
        self._user_signal = user_signal
        self._windows: dict[str, Stage0Window] = {}

    def window_for(self, request_id: str) -> Stage0Window:
        window = self._windows.get(request_id)
        if window is None or window.closed:
            window = Stage0Window(self._user_signal)
            self._windows[request_id] = window
        return window


def window_decision(token_ids: Sequence[int]) -> str | None:
    """Classify a finished window from the tokens the engine received.

    Every window stops on ``<|unit_end|>``, so the decision is the token ahead
    of it. Text tokens arrive on earlier steps and never finish a segment.
    """

    if len(token_ids) < 2 or token_ids[-1] != DUPLEX_TOKENS.unit_end:
        return None
    previous = token_ids[-2]
    if previous == DUPLEX_TOKENS.listen:
        return "listen"
    if previous == DUPLEX_TOKENS.turn_eos:
        return "turn_end"
    return "speak"


def window_token_ids(output: object) -> list[int]:
    """Read a stage output's whole token history, not the step's delta.

    One session is one resumable request, and it runs on DELTA output, so both
    the stage output's ``token_ids`` and the orchestrator's segment snapshot
    hold only the step being reported. ``window_decision`` needs the token ahead
    of ``<|unit_end|>``, which the step that closes a listen window never
    carries. The Omni output processor attaches the cumulative view for exactly
    this.
    """

    completion = next(iter(getattr(output, "outputs", ()) or ()), output)
    raw = getattr(completion, "cumulative_token_ids", None)
    if raw is None:
        raw = getattr(completion, "token_ids", None)
    if raw is None:
        return []
    try:
        return [int(token_id) for token_id in raw]
    except (TypeError, ValueError):
        return []


__all__ = [
    "MAX_GENERATED_TOKENS_PER_WINDOW",
    "MAX_TEXT_TOKENS_PER_WINDOW",
    "WINDOW_PROTOCOL_OVERHEAD",
    "Stage0Window",
    "Stage0Windows",
    "ThinkerProtocolError",
    "window_decision",
    "window_token_ids",
]
