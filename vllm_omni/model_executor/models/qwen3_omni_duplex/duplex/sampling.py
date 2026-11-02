# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Window protocol shared by the Stage 0 worker and the duplex engine.

The worker constrains sampling. The engine replays forced boundary tokens,
because the scheduler drops every sampled token of a finished chunk and the
worker has no channel back to the engine. Both sides therefore advance the
same protocol state; the engine does it from the tokens it already received.
"""

from __future__ import annotations

from vllm_omni.model_executor.models.qwen3_omni_duplex.thinker_window import (
    ThinkerProtocolError,
    ThinkerWindowState,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

# The reference runtime caps a window at 16 generated tokens. Stage 0 has no
# ASR, so the text budget is one below that, leaving room for the closing token.
MAX_GENERATED_TOKENS_PER_WINDOW = 16
MAX_TEXT_TOKENS_PER_WINDOW = MAX_GENERATED_TOKENS_PER_WINDOW - 1


class Stage0Window:
    """One request's window, from its first sampled token to ``<|unit_end|>``."""

    def __init__(self) -> None:
        self.state = ThinkerWindowState(
            tokens=DUPLEX_TOKENS,
            asr_enabled=False,
            max_generated_tokens=MAX_GENERATED_TOKENS_PER_WINDOW,
            max_text_tokens=MAX_TEXT_TOKENS_PER_WINDOW,
        )
        self.forced_token_ids: tuple[int, ...] = ()
        self.closed = False

    def constrain(self, logits: dict[int, float]) -> dict[int, float]:
        return self.state.control_logits(logits)

    def observe(self, token_id: int) -> bool:
        """Accept one sampled token. Returns whether the window closed."""

        step = self.state.observe(token_id)
        if not step.closed:
            return False
        # The sampled token is fed by the runner. Only the tokens the protocol
        # appends after it are missing from the KV cache.
        self.forced_token_ids = step.feed_token_ids[1:]
        self.closed = True
        return True


class Stage0Windows:
    """Window states keyed by request, reset whenever a window closes."""

    def __init__(self) -> None:
        self._windows: dict[str, Stage0Window] = {}

    def window_for(self, request_id: str) -> Stage0Window:
        window = self._windows.get(request_id)
        if window is None or window.closed:
            window = Stage0Window()
            self._windows[request_id] = window
        return window

    def take_forced_token_ids(self, request_id: str) -> tuple[int, ...] | None:
        """Return the closed window's replay tokens and drop its state."""

        window = self._windows.get(request_id)
        if window is None or not window.closed:
            return None
        del self._windows[request_id]
        return window.forced_token_ids


def forced_token_ids(sampled_token_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Boundary tokens a finished window must replay before the next one.

    Replaying the protocol from the sampled tokens is exact: the worker sampled
    under the same rules, and a window that never closed replays nothing.
    """

    window = Stage0Window()
    for token_id in sampled_token_ids:
        try:
            window.observe(token_id)
        except ThinkerProtocolError:
            return ()
    return window.forced_token_ids if window.closed else ()
