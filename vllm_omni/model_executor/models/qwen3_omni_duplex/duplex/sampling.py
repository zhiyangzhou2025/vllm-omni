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
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKEN_IDS, DUPLEX_TOKENS

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

    def __init__(self, user_signal: UserSignalMode, *, active_generation: bool = False) -> None:
        self.state = ThinkerWindowState(
            tokens=DUPLEX_TOKENS,
            user_signal=user_signal,
            max_generated_tokens=MAX_GENERATED_TOKENS_PER_WINDOW,
            max_text_tokens=MAX_TEXT_TOKENS_PER_WINDOW,
            active_generation=active_generation,
        )
        self.closed = False

    def constrain(self, logits: dict[int, float]) -> dict[int, float]:
        return self.state.control_logits(logits)

    def observe(self, token_id: int) -> None:
        """Accept one sampled token and record whether it closed the window."""

        self.closed = self.state.observe(token_id).closed


class Stage0Windows:
    """Window states keyed by request, reset whenever a window closes.

    Whether a System 2 generation is running outlives the window: a closed
    window's ``<|launch|>`` / ``<|cancel|>`` carry it into the next one, and a
    final tool result injected ahead of a window ends it.
    """

    def __init__(self, user_signal: UserSignalMode) -> None:
        self._user_signal = user_signal
        self._windows: dict[str, Stage0Window] = {}
        self._active: dict[str, bool] = {}

    def window_for(self, request_id: str, *, system2_result: str | None = None) -> Stage0Window:
        window = self._windows.get(request_id)
        if window is None or window.closed:
            active = self._active.get(request_id, False)
            if window is not None:
                actions = window.state.actions
                if "launch" in actions:
                    active = True
                elif "cancel" in actions:
                    active = False
            if system2_result == "final":
                active = False
            self._active[request_id] = active
            window = Stage0Window(self._user_signal, active_generation=active)
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


def closed_window_count(token_ids: Sequence[int]) -> int:
    """Count the windows that already closed, which is also the newest index.

    ``<|unit_end|>`` closes every window, so counting it counts the windows.
    The count doubles as the zero-based index of the window the next token
    belongs to, which is how a window is dated: window ``k`` carries the audio
    of ``[k * UNIT_DURATION_MS, (k + 1) * UNIT_DURATION_MS)``.
    """

    return sum(1 for token_id in token_ids if token_id == DUPLEX_TOKENS.unit_end)


def window_text_role(token_ids: Sequence[int]) -> str | None:
    """Name whose words the newest token is, or ``None`` for a protocol token.

    A ``transcript`` window transcribes the user between ``<|asr|>`` and
    ``<|asr_eos|>``. Those tokens detokenize like any other text, so reporting
    them as the assistant's transcript interleaves both speakers in one stream.
    A text token belongs to whichever phase opener precedes it, and ``<|asr|>``
    is the only opener that makes it the user's words.
    """

    if not token_ids or token_ids[-1] in DUPLEX_TOKEN_IDS:
        return None
    for token_id in reversed(token_ids[:-1]):
        if token_id in DUPLEX_TOKEN_IDS:
            return "user" if token_id == DUPLEX_TOKENS.asr else "assistant"
    return "assistant"


def window_user_activity(token_ids: Sequence[int]) -> str | None:
    """Read an ``activity`` window's ``idle`` / ``non_idle`` verdict.

    The ``activity`` phase forces that pair before anything else, so the
    verdict is the newest window's opening token. The other two user signal
    modes open on something else and yield ``None``.
    """

    window = _newest_window(token_ids)
    if not window:
        return None
    if window[0] == DUPLEX_TOKENS.idle:
        return "idle"
    if window[0] == DUPLEX_TOKENS.non_idle:
        return "non_idle"
    return None


def window_actions(token_ids: Sequence[int]) -> list[int]:
    """The newest window's System 2 actions, ``<|launch|>`` / ``<|cancel|>``, in order."""

    actions = (DUPLEX_TOKENS.launch, DUPLEX_TOKENS.cancel)
    return [token_id for token_id in _newest_window(token_ids) if token_id in actions]


def _newest_window(token_ids: Sequence[int]) -> Sequence[int]:
    """Slice off the newest window, whether it just closed or is still open.

    The final token is excluded from the search so that the window a closing
    ``<|unit_end|>`` belongs to is the one returned, rather than an empty slice
    after it.
    """

    for offset in range(len(token_ids) - 1, 0, -1):
        if token_ids[offset - 1] == DUPLEX_TOKENS.unit_end:
            return token_ids[offset:]
    return token_ids


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
    "closed_window_count",
    "window_actions",
    "window_decision",
    "window_text_role",
    "window_token_ids",
    "window_user_activity",
]
