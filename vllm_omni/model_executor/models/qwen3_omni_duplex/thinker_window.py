# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Protocol state for one Qwen3-Omni-Duplex Thinker window.

The checkpoint-backed sequence is::

    <|unit_start|> environment [<|asr|> payload <|asr_eos|>]
    (<|listen|> | <|speak|> [actions] [text] [<|turn_eos|>] <|chunk_eos|>)
    <|unit_end|>

This module owns only token legality and forced boundary tokens. Model
execution supplies embeddings, KV state, and sampled token IDs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import (
    DUPLEX_TOKEN_IDS,
    DuplexTokenIDs,
)

ThinkerPhase = Literal[
    "asr_start",
    "asr",
    "control",
    "action_or_text",
    "launch_or_text",
    "text",
    "post_turn_eos",
    "closed",
]


class ThinkerProtocolError(ValueError):
    """A sampled token cannot legally continue the current window."""


@dataclass(frozen=True, slots=True)
class ThinkerWindowStep:
    """One accepted sampled token and the IDs that must enter Thinker KV."""

    sampled_token_id: int
    feed_token_ids: tuple[int, ...]
    phase: ThinkerPhase
    closed: bool
    asr_token_id: int | None = None
    text_token_id: int | None = None
    end_of_turn: bool = False


@dataclass(slots=True)
class ThinkerWindowState:
    """Advance one window from its first sampled token through ``<|unit_end|>``."""

    tokens: DuplexTokenIDs
    asr_enabled: bool
    max_generated_tokens: int
    max_text_tokens: int
    phase: ThinkerPhase = "asr_start"
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
        if not self.asr_enabled:
            self.phase = "control"

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
        if self.phase == "post_turn_eos":
            # The turn ended on ``<|turn_eos|>``; the protocol samples
            # ``<|chunk_eos|>`` next, and that sample is the window's stop.
            return self._force(values, self.tokens.chunk_eos)
        allowed = {self.tokens.chunk_eos, self.tokens.turn_eos}
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
        if self.assistant_text_tokens >= self.max_text_tokens:
            return self._force(values, self.tokens.chunk_eos)
        result = self._mask(values, allowed=allowed)
        if bias is not None and bias[1]:
            result[bias[0]] += bias[1]
        return result

    def observe(self, token_id: int) -> ThinkerWindowStep:
        """Accept one sampled token and report every token that must be fed."""

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
        if self.phase == "asr_start":
            self._expect(token_id, self.tokens.asr, "window must start with <|asr|>")
            self.phase = "asr"
            return self._step(token_id, (token_id,))
        if self.phase == "asr":
            if token_id == self.tokens.asr_eos:
                self.phase = "control"
                return self._step(token_id, (token_id,))
            self._reject_reserved(token_id, "reserved token inside ASR payload")
            self.asr_text_tokens += 1
            return self._step(token_id, (token_id,), asr_token_id=token_id)
        if self.phase == "control":
            if token_id == self.tokens.listen:
                return self._close((token_id, self.tokens.unit_end))
            self._expect(token_id, self.tokens.speak, "control must be <|listen|> or <|speak|>")
            self.phase = "action_or_text"
            return self._step(token_id, (token_id,))
        if self.phase == "action_or_text" and token_id in {self.tokens.launch, self.tokens.cancel}:
            return self._action(token_id)
        if self.phase == "launch_or_text" and token_id == self.tokens.launch:
            return self._action(token_id)
        if self.phase in {"action_or_text", "launch_or_text"}:
            self.phase = "text"
        if token_id == self.tokens.chunk_eos:
            # A ``<|chunk_eos|>`` after ``<|turn_eos|>`` closes the turn too.
            return self._close((token_id, self.tokens.unit_end), end_of_turn=self.phase == "post_turn_eos")
        if token_id == self.tokens.turn_eos:
            # The turn ends, but the window closes on the ``<|chunk_eos|>`` the
            # protocol samples next. The runner feeds this token normally.
            self.phase = "post_turn_eos"
            return self._step(token_id, (token_id,), end_of_turn=True)
        self._reject_reserved(token_id, "reserved token inside assistant text")
        self.assistant_text_tokens += 1
        return self._step(token_id, (token_id,), text_token_id=token_id)

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
        return self._step(token_id, (token_id,))

    def _action_ids(self) -> set[int]:
        names = {"launch": self.tokens.launch, "cancel": self.tokens.cancel}
        return {names[name] for name in self.actions}

    def _close(self, feed_token_ids: tuple[int, ...], *, end_of_turn: bool = False) -> ThinkerWindowStep:
        step = self._step(feed_token_ids[0], feed_token_ids, end_of_turn=end_of_turn)
        return ThinkerWindowStep(
            sampled_token_id=step.sampled_token_id,
            feed_token_ids=step.feed_token_ids,
            phase="closed",
            closed=True,
            end_of_turn=end_of_turn,
        )

    def _step(
        self,
        sampled_token_id: int,
        feed_token_ids: tuple[int, ...],
        *,
        asr_token_id: int | None = None,
        text_token_id: int | None = None,
        end_of_turn: bool = False,
    ) -> ThinkerWindowStep:
        return ThinkerWindowStep(
            sampled_token_id=sampled_token_id,
            feed_token_ids=feed_token_ids,
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
