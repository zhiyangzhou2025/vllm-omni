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

# The reference runtime caps a window at 16 generated tokens. The text budget is
# one below that, leaving room for the closing token.
MAX_GENERATED_TOKENS_PER_WINDOW = 16
MAX_TEXT_TOKENS_PER_WINDOW = MAX_GENERATED_TOKENS_PER_WINDOW - 1


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


__all__ = [
    "MAX_GENERATED_TOKENS_PER_WINDOW",
    "MAX_TEXT_TOKENS_PER_WINDOW",
    "Stage0Window",
    "Stage0Windows",
    "ThinkerProtocolError",
    "window_decision",
]
