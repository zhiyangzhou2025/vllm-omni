# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 output projection for Qwen3-Omni-Duplex.

Stage 1 does not exist yet, so a window's text is the final output. Audio is
produced by Stage 2 and is not invented here.

Every window also carries the model's verdict on the *user*: who is speaking
(``activity``) or what they said (``transcript``). That verdict is relayed as
is. No segmentation policy beyond the one threshold the model's own jitter
forces, and no synthesised VAD.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    closed_window_count,
    window_decision,
    window_text_role,
    window_token_ids,
    window_user_activity,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import UNIT_DURATION_MS

# Consecutive silent windows that end a user utterance. One silent window is
# jitter, not a boundary: in the 54-window reference run the silent runs were
# 7, 3 and 1 windows long, and the 1-window run sits inside a single sentence.
# Both the ``activity`` and the ``transcript`` checkpoint produced that same
# run-length profile, so one threshold serves both. 2 windows is 960 ms.
USER_SPEECH_SILENCE_WINDOWS = 2


@dataclass
class _UserSpeech:
    """One request's user-side utterance, tracked a window at a time."""

    item_id: str | None = None
    silent_windows: int = 0
    parts: list[str] = field(default_factory=list)
    window_parts: list[str] = field(default_factory=list)


class Qwen3OmniDuplexDataPlane(DuplexDataPlane):
    def __init__(self) -> None:
        self.terminal: set[str] = set()
        self._user_speech: dict[str, _UserSpeech] = {}

    def begin_request(self, request_id: str) -> None:
        # Called once per append, so once per window. The utterance state spans
        # windows and must survive; only ``close_session`` ends it.
        self.terminal.discard(request_id)

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id in self.terminal

    def mark_terminal(self, request_id: str) -> None:
        self.terminal.add(request_id)

    def close_stream(self, request_id: str) -> None:
        return None

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        del active_request_id
        self.terminal = {
            request_id
            for request_id in self.terminal
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }
        self._user_speech = {
            request_id: speech
            for request_id, speech in self._user_speech.items()
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }

    def project(self, result: object, *, context: object | None = None) -> Iterable[dict[str, object]]:
        if not isinstance(result, dict):
            return
        # ``context`` is ``DuplexDataPlaneContext``, a dataclass, so its fields
        # are attributes. Reading it as a mapping pinned every window to turn 0,
        # and the session then dropped every model turn after the first as the
        # late output of a turn it had already completed.
        model_turn_id = getattr(context, "active_response_turn_id", None)
        if model_turn_id is None:
            model_turn_id = getattr(context, "turn_id", 0)
        response_format = getattr(context, "response_format", "pcm16")
        for output in result.get("data_plane_outputs", []):
            request_id = output.request_id
            if self.is_terminal(request_id):
                continue
            completion = next(iter(getattr(output, "outputs", ()) or ()), None)
            token_ids = window_token_ids(completion)
            text = getattr(completion, "text", "") if getattr(output, "stage_id", 0) == 0 else ""
            if not isinstance(text, str):
                text = ""
            # A ``transcript`` window transcribes the user before it decides what
            # to do. Those tokens are the user's words, not the assistant's, and
            # they belong to the input stream rather than to this response.
            role = window_text_role(token_ids)
            speech = self._user_speech.setdefault(request_id, _UserSpeech())
            if text and role == "user":
                speech.window_parts.append(text)
            if role != "assistant":
                text = ""
            # ``decide_output`` never runs here: the runner calls it only for
            # stages below the final one, and Stage 0 is the only stage.
            decision = window_decision(token_ids)
            user_speech_events = () if decision is None else _close_user_window(speech, token_ids)
            yield {
                "supported": True,
                "data_plane_request_id": request_id,
                "text": text,
                "audio_data": "",
                "audio_format": response_format,
                "audio_duration_ms": 0,
                "audio_text_mark": False,
                "text_requires_complete_audio": False,
                "audio_complete": False,
                "sample_rate_hz": 24000,
                "history_audio_pcm": None,
                "history_audio_supported": False,
                "is_listen": decision == "listen",
                "model_listen": decision == "listen",
                "end_of_turn": decision == "turn_end",
                "user_speech_events": list(user_speech_events),
                "model_turn_id": model_turn_id,
                "runtime_impl": "qwen3_omni_duplex_stage0",
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
            }


def _close_user_window(speech: _UserSpeech, token_ids: Sequence[int]) -> list[dict[str, object]]:
    """Turn one closed window's user-side verdict into input-stream events.

    The verdict is read from wherever the checkpoint puts it: the ``idle`` /
    ``non_idle`` pair in ``activity`` mode, otherwise whether the window
    transcribed anything. A ``none`` checkpoint has neither, so it reports
    nothing at all.
    """

    index = closed_window_count(token_ids) - 1
    window_text = "".join(speech.window_parts)
    speech.window_parts.clear()
    activity = window_user_activity(token_ids)
    speaking = activity == "non_idle" if activity is not None else bool(window_text)
    events: list[dict[str, object]] = []
    if speaking:
        if speech.item_id is None:
            speech.item_id = f"item_{uuid4().hex}"
            events.append({"kind": "started", "item_id": speech.item_id, "audio_start_ms": index * UNIT_DURATION_MS})
        speech.silent_windows = 0
        if window_text:
            speech.parts.append(window_text)
            events.append({"kind": "delta", "item_id": speech.item_id, "delta": window_text})
        return events
    if speech.item_id is None:
        return events
    speech.silent_windows += 1
    if speech.silent_windows < USER_SPEECH_SILENCE_WINDOWS:
        return events
    # The utterance ended with the last window that carried it, not with the
    # silence that proved it was over.
    audio_end_ms = (index - speech.silent_windows + 1) * UNIT_DURATION_MS
    events.append({"kind": "stopped", "item_id": speech.item_id, "audio_end_ms": audio_end_ms})
    events.append({"kind": "completed", "item_id": speech.item_id, "transcript": "".join(speech.parts)})
    speech.item_id = None
    speech.silent_windows = 0
    speech.parts.clear()
    return events
