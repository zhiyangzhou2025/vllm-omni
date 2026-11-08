# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Protocol state for one Qwen3-Omni-Duplex Thinker window.

The checkpoint-backed sequence is::

    <|unit_start|> environment [user signal]
    (<|listen|> | <|speak|> [actions] [text] [<|turn_eos|>])
    <|unit_end|>

``<|unit_end|>`` is the one token that ends a window, for LISTEN and SPEAK
alike, and the model is supervised on it. The user signal block depends on the
checkpoint: nothing, an ASR transcript, or a single activity flag.

This module owns only token legality. Model execution supplies embeddings, KV
state, and sampled token IDs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import (
    DUPLEX_TOKEN_IDS,
    DuplexTokenIDs,
)

UserSignalMode = Literal["none", "transcript", "activity"]

ThinkerPhase = Literal[
    "activity",
    "asr_start",
    "asr",
    "control",
    "action_or_text",
    "launch_or_text",
    "text",
    "post_listen",
    "post_turn_eos",
    "closed",
]

_START_PHASE: dict[UserSignalMode, ThinkerPhase] = {
    "none": "control",
    "transcript": "asr_start",
    "activity": "activity",
}


class ThinkerProtocolError(ValueError):
    """A sampled token cannot legally continue the current window."""


@dataclass(frozen=True, slots=True)
class ThinkerWindowStep:
    """One accepted sampled token and the state it leaves the window in."""

    sampled_token_id: int
    phase: ThinkerPhase
    closed: bool
    asr_token_id: int | None = None
    text_token_id: int | None = None
    end_of_turn: bool = False


@dataclass(slots=True)
class ThinkerWindowState:
    """Advance one window from its first sampled token through ``<|unit_end|>``."""

    tokens: DuplexTokenIDs
    user_signal: UserSignalMode
    max_generated_tokens: int
    max_text_tokens: int
    phase: ThinkerPhase = "control"
    generated_tokens: int = 0
    asr_text_tokens: int = 0
    assistant_text_tokens: int = 0
    actions: tuple[str, ...] = ()
    active_generation: bool = False

    def __post_init__(self) -> None:
        if self.max_generated_tokens <= 0:
            raise ValueError("max_generated_tokens must be positive")
        if not 0 <= self.max_text_tokens < self.max_generated_tokens:
            raise ValueError("max_text_tokens must be >= 0 and < max_generated_tokens")
        if self.user_signal not in _START_PHASE:
            raise ValueError(f"unknown user signal mode {self.user_signal!r}")
        self.phase = _START_PHASE[self.user_signal]

    def control_logits(
        self,
        logits: Mapping[int, float],
        *,
        listen_bias: float = 0.0,
        launch_bias: float = 0.0,
        cancel_bias: float = 0.0,
    ) -> dict[int, float]:
        """Return the legal logits for the current phase."""

        if self.phase == "closed":
            raise ThinkerProtocolError("window is already closed")
        values = dict(logits)
        if self.phase == "activity":
            return self._force_pair(values, self.tokens.idle, self.tokens.non_idle)
        if self.phase == "asr_start":
            return self._force(values, self.tokens.asr)
        if self.phase == "asr":
            if self.asr_text_tokens >= self.max_text_tokens:
                return self._force(values, self.tokens.asr_eos)
            return self._mask(values, allowed={self.tokens.asr_eos})
        if self.phase == "control":
            result = self._force_pair(values, self.tokens.listen, self.tokens.speak)
            if listen_bias:
                result[self.tokens.listen] += listen_bias
            return result
        if self.phase in {"post_listen", "post_turn_eos"}:
            # ``<|listen|>`` and ``<|turn_eos|>`` both end free generation. The
            # reference runtime force-feeds ``<|unit_end|>`` here; constraining
            # the next step to it instead makes that token the request's stop.
            return self._force(values, self.tokens.unit_end)
        allowed = {self.tokens.unit_end, self.tokens.turn_eos}
        bias: tuple[int, float] | None = None
        if self.phase == "action_or_text":
            allowed.add(self.tokens.launch)
            if self.active_generation:
                allowed.add(self.tokens.cancel)
                bias = (self.tokens.cancel, cancel_bias)
            else:
                bias = (self.tokens.launch, launch_bias)
        elif self.phase == "launch_or_text":
            allowed.add(self.tokens.launch)
            bias = (self.tokens.launch, launch_bias)
        # Spending the last budgeted token on text would end the window by
        # length instead of on ``<|unit_end|>``, so close it one step early.
        if self.assistant_text_tokens >= self.max_text_tokens or self.generated_tokens >= self.max_generated_tokens - 1:
            return self._force(values, self.tokens.unit_end)
        result = self._mask(values, allowed=allowed)
        if bias is not None and bias[1]:
            result[bias[0]] += bias[1]
        return result

    def observe(self, token_id: int) -> ThinkerWindowStep:
        """Accept one sampled token and report the phase it leads to."""

        if self.phase == "closed":
            raise ThinkerProtocolError("window is already closed")
        if self.generated_tokens >= self.max_generated_tokens:
            raise ThinkerProtocolError("generation budget is exhausted")
        self.generated_tokens += 1
        step = self._observe(token_id)
        if step.closed:
            self.phase = "closed"
        return step

    def _observe(self, token_id: int) -> ThinkerWindowStep:
        if self.phase == "activity":
            if token_id not in {self.tokens.idle, self.tokens.non_idle}:
                raise ThinkerProtocolError("user activity must be <|idle|> or <|non_idle|>")
            self.phase = "control"
            return self._step(token_id)
        if self.phase == "asr_start":
            self._expect(token_id, self.tokens.asr, "window must start with <|asr|>")
            self.phase = "asr"
            return self._step(token_id)
        if self.phase == "asr":
            if token_id == self.tokens.asr_eos:
                self.phase = "control"
                return self._step(token_id)
            self._reject_reserved(token_id, "reserved token inside ASR payload")
            self.asr_text_tokens += 1
            return self._step(token_id, asr_token_id=token_id)
        if self.phase in {"post_listen", "post_turn_eos"}:
            self._expect(token_id, self.tokens.unit_end, "window must close on <|unit_end|>")
            return self._close(token_id, end_of_turn=self.phase == "post_turn_eos")
        if self.phase == "control":
            if token_id == self.tokens.listen:
                # The window stays open for the constrained ``<|unit_end|>``.
                self.phase = "post_listen"
                return self._step(token_id)
            self._expect(token_id, self.tokens.speak, "control must be <|listen|> or <|speak|>")
            self.phase = "action_or_text"
            return self._step(token_id)
        if self.phase == "action_or_text" and token_id in {self.tokens.launch, self.tokens.cancel}:
            return self._action(token_id)
        if self.phase == "launch_or_text" and token_id == self.tokens.launch:
            return self._action(token_id)
        if self.phase in {"action_or_text", "launch_or_text"}:
            self.phase = "text"
        if token_id == self.tokens.unit_end:
            return self._close(token_id)
        if token_id == self.tokens.turn_eos:
            # The turn ends, but the window closes on the ``<|unit_end|>`` the
            # next step is constrained to. The runner feeds this token normally.
            self.phase = "post_turn_eos"
            return self._step(token_id, end_of_turn=True)
        self._reject_reserved(token_id, "reserved token inside assistant text")
        self.assistant_text_tokens += 1
        return self._step(token_id, text_token_id=token_id)

    def _action(self, token_id: int) -> ThinkerWindowStep:
        if token_id == self.tokens.cancel:
            if not self.active_generation or self.tokens.cancel in self._action_ids():
                raise ThinkerProtocolError("illegal <|cancel|>")
            self.actions += ("cancel",)
            self.phase = "launch_or_text"
        else:
            if self.tokens.launch in self._action_ids():
                raise ThinkerProtocolError("duplicate <|launch|>")
            self.actions += ("launch",)
            self.phase = "text"
        return self._step(token_id)

    def _action_ids(self) -> set[int]:
        names = {"launch": self.tokens.launch, "cancel": self.tokens.cancel}
        return {names[name] for name in self.actions}

    def _close(self, token_id: int, *, end_of_turn: bool = False) -> ThinkerWindowStep:
        return ThinkerWindowStep(
            sampled_token_id=token_id,
            phase="closed",
            closed=True,
            end_of_turn=end_of_turn,
        )

    def _step(
        self,
        sampled_token_id: int,
        *,
        asr_token_id: int | None = None,
        text_token_id: int | None = None,
        end_of_turn: bool = False,
    ) -> ThinkerWindowStep:
        return ThinkerWindowStep(
            sampled_token_id=sampled_token_id,
            phase=self.phase,
            closed=False,
            asr_token_id=asr_token_id,
            text_token_id=text_token_id,
            end_of_turn=end_of_turn,
        )

    def _expect(self, actual: int, expected: int, message: str) -> None:
        if actual != expected:
            raise ThinkerProtocolError(message)

    def _reject_reserved(self, token_id: int, message: str) -> None:
        if token_id in DUPLEX_TOKEN_IDS:
            raise ThinkerProtocolError(message)

    @staticmethod
    def _force(logits: Mapping[int, float], token_id: int) -> dict[int, float]:
        if token_id not in logits:
            raise ThinkerProtocolError(f"missing logit for token {token_id}")
        return {token_id: logits[token_id]}

    @classmethod
    def _force_pair(cls, logits: Mapping[int, float], first: int, second: int) -> dict[int, float]:
        return {**cls._force(logits, first), **cls._force(logits, second)}

    @staticmethod
    def _mask(logits: Mapping[int, float], *, allowed: set[int]) -> dict[int, float]:
        return {
            token_id: value
            for token_id, value in logits.items()
            if token_id not in DUPLEX_TOKEN_IDS or token_id in allowed
        }
