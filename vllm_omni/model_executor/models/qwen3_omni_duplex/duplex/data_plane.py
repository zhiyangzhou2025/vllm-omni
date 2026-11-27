# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Output projection for Qwen3-Omni-Duplex.

Stage 0's text is the final text. When a later stage produces audio, that audio
is projected here too: a speak window's text waits while an earlier turn's audio
is still in flight, and the turn ends with its last audio chunk.

Every window also carries the model's verdict on the *user*: who is speaking
(``activity``) or what they said (``transcript``). That verdict is relayed as
is. No segmentation policy beyond the one threshold the model's own jitter
forces, and no synthesised VAD.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    closed_window_count,
    window_actions,
    window_decision,
    window_text_role,
    window_token_ids,
    window_user_activity,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import UNIT_DURATION_MS
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

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


@dataclass
class _TurnAudio:
    """One request's turns, so text and audio of different turns do not cross.

    ``s_turn`` is the Stage 0 turn being projected, ``done_turn`` the first turn
    whose audio has all been sent. A later turn's text is held until then.
    ``windows`` pairs each speak window with the turn it belongs to.
    """

    s_turn: int = 0
    done_turn: int = 0
    held: dict[int, str] = field(default_factory=dict)
    template: dict[str, object] | None = None
    windows: deque[tuple[int, bool]] = field(default_factory=deque)


SYSTEM2_LAUNCH = "system2_launch"
SYSTEM2_CANCEL = "system2_cancel"
# The projected result that carries one System 2 call. It is its own result
# with no text, because the session publishes a recognized call and drops the
# rest of that result.
FUNCTION_CALL_KEY = "qwen3_duplex_function_call"


@dataclass
class System2Generation:
    """One request's System 2 generation, as the client sees it.

    ``generation_id`` names the generation the model launched and has neither
    cancelled nor received a final result for. ``awaiting`` is the one launch
    call whose result is still owed. ``relaunch`` asks the next closed window
    to publish the call for the generation's next result.
    """

    generation_id: str | None = None
    awaiting: str | None = None
    result_index: int = 0
    relaunch: bool = False


def _new_call_id() -> str:
    return f"call_{uuid4().hex}"


class Qwen3OmniDuplexDataPlane(DuplexDataPlane):
    def __init__(self, encode_audio: Any = None) -> None:
        self.terminal: set[str] = set()
        self._user_speech: dict[str, _UserSpeech] = {}
        self.system2: dict[str, System2Generation] = {}
        # Every published call names the request it belongs to until it settles.
        self.call_requests: dict[str, str] = {}
        # Launch calls still owed a result whose generation has ended. Their
        # result settles the call and is dropped.
        self.abandoned_calls: set[str] = set()
        self._encode_audio = encode_audio
        # Set by the plugin once the deployment's stages are known. None means
        # this deployment has no audio stage, so Stage 0 behaves as before.
        self._audio_stage: int | None = None
        self._turns: dict[str, _TurnAudio] = {}
        self.on_close_session: Any = None

    def set_audio_stage(self, stage_id: int | None) -> None:
        self._audio_stage = stage_id

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
        if self.on_close_session is not None:
            self.on_close_session(session_id)
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
        self._turns = {
            request_id: turns
            for request_id, turns in self._turns.items()
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }
        self.system2 = {
            request_id: generation
            for request_id, generation in self.system2.items()
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }
        self.call_requests = {
            call_id: request_id
            for call_id, request_id in self.call_requests.items()
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }
        self.abandoned_calls &= self.call_requests.keys()

    def _abandon(self, generation: System2Generation) -> None:
        if generation.awaiting is not None:
            self.abandoned_calls.add(generation.awaiting)
        generation.generation_id = generation.awaiting = None

    def _system2_calls(self, request_id: str, actions: Sequence[int]) -> list[dict[str, str]]:
        """The calls one closed window publishes, in the order the model acted."""
        generation = self.system2.setdefault(request_id, System2Generation())
        calls: list[dict[str, str]] = []
        for token_id in actions:
            if token_id == DUPLEX_TOKENS.cancel and generation.generation_id is not None:
                calls.append(self._publish(request_id, SYSTEM2_CANCEL, {"generation_id": generation.generation_id}))
                self._abandon(generation)
                generation.relaunch = False
            elif token_id == DUPLEX_TOKENS.launch:
                self._abandon(generation)
                call_id = _new_call_id()
                generation.generation_id = generation.awaiting = call_id
                generation.result_index = 0
                generation.relaunch = False
                calls.append(
                    self._publish(
                        request_id, SYSTEM2_LAUNCH, {"generation_id": call_id, "result_index": 0}, call_id=call_id
                    )
                )
        if generation.relaunch and generation.generation_id is not None:
            generation.relaunch = False
            generation.result_index += 1
            generation.awaiting = _new_call_id()
            calls.append(
                self._publish(
                    request_id,
                    SYSTEM2_LAUNCH,
                    {"generation_id": generation.generation_id, "result_index": generation.result_index},
                    call_id=generation.awaiting,
                )
            )
        return calls

    def _publish(
        self, request_id: str, name: str, arguments: dict[str, object], *, call_id: str | None = None
    ) -> dict[str, str]:
        call_id = call_id or _new_call_id()
        self.call_requests[call_id] = request_id
        return {"call_id": call_id, "name": name, "arguments": json.dumps(arguments)}

    def settled_call(self, call_id: str) -> None:
        """The client answered, or can no longer answer, this call."""
        self.abandoned_calls.discard(call_id)
        request_id = self.call_requests.pop(call_id, None)
        generation = self.system2.get(request_id) if request_id is not None else None
        if generation is not None and generation.awaiting == call_id:
            generation.awaiting = None

    def submitted_result(self, request_id: str, generation_id: str, *, final: bool) -> None:
        """A result reached the Thinker: end its generation or ask for the next result."""
        generation = self.system2.get(request_id)
        if generation is None or generation.generation_id != generation_id:
            return
        if final:
            generation.generation_id = generation.awaiting = None
            generation.relaunch = False
        elif generation.awaiting is None:
            generation.relaunch = True

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
        outputs = list(result.get("data_plane_outputs", []))
        audio_stage = self._audio_stage
        projects_audio = audio_stage is not None and "audio" in (getattr(context, "modalities", ()) or ())
        if projects_audio:
            yield from self._project_with_audio(
                result,
                outputs,
                audio_stage=audio_stage,
                context=context,
                model_turn_id=model_turn_id,
                response_format=response_format,
            )
            return
        for output in outputs:
            # Only Stage 0's text is projected here. The audio stage's output is
            # audio, and a session that asked for none gets none of it.
            if getattr(output, "stage_id", 0) != 0:
                continue
            yield from self._project_stage0(output, model_turn_id=model_turn_id, response_format=response_format)

    def _project_stage0(
        self,
        output: object,
        *,
        model_turn_id: object,
        response_format: object,
    ) -> Iterable[dict[str, object]]:
        request_id = output.request_id
        if self.is_terminal(request_id):
            return
            yield  # keeps this a generator when the request is already terminal
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
        projected: dict[str, object] = {
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
        yield projected
        if decision is None:
            return
        # The window's text and audio go out first; each call follows on its own.
        for call in self._system2_calls(request_id, window_actions(token_ids)):
            yield {
                **projected,
                "text": "",
                "is_listen": False,
                "model_listen": False,
                "end_of_turn": False,
                "user_speech_events": [],
                FUNCTION_CALL_KEY: call,
            }

    def _project_with_audio(
        self,
        result: dict[str, object],
        outputs: list[object],
        *,
        audio_stage: int,
        context: object,
        model_turn_id: object,
        response_format: object,
    ) -> Iterable[dict[str, object]]:
        """Stage 0 text now, Stage 2 audio with the turn it belongs to.

        One session has one active response, so a later turn's text is held
        while an earlier turn's audio is still going out, and flushed once that
        turn ends. ``end_of_turn`` moves from the text to the turn's last audio.
        """

        stage0 = [output for output in outputs if getattr(output, "stage_id", 0) != audio_stage]
        audio = [output for output in outputs if getattr(output, "stage_id", 0) == audio_stage]
        for output in stage0:
            request_id = output.request_id
            turns = self._turns.setdefault(request_id, _TurnAudio())
            if turns.done_turn in turns.held and turns.template is not None:
                yield {
                    **turns.template,
                    "text": turns.held.pop(turns.done_turn),
                    "is_listen": False,
                    "model_listen": False,
                    "end_of_turn": False,
                    "user_speech_events": [],
                    "model_turn_id": model_turn_id,
                }
            for projected in self._project_stage0(output, model_turn_id=model_turn_id, response_format=response_format):
                if FUNCTION_CALL_KEY in projected:
                    yield projected
                    continue
                in_flight = turns.done_turn < turns.s_turn
                turns.template = dict(projected)
                if projected["text"] and in_flight:
                    turns.held[turns.s_turn] = turns.held.get(turns.s_turn, "") + str(projected["text"])
                    projected["text"] = ""
                projected["end_of_turn"] = False
                decision = window_decision(window_token_ids(next(iter(getattr(output, "outputs", ()) or ()), None)))
                if decision in ("speak", "turn_end"):
                    turns.windows.append((turns.s_turn, decision == "turn_end"))
                    if decision == "turn_end":
                        turns.s_turn += 1
                elif (
                    decision == "listen" and in_flight and getattr(context, "active_response_turn_id", None) is not None
                ):
                    projected["is_listen"] = False
                    projected["model_listen"] = False
                yield projected
        for output in audio:
            request_id = output.request_id
            turns = self._turns.setdefault(request_id, _TurnAudio())
            window_turn, final = turns.windows.popleft()
            if window_turn in turns.held and turns.done_turn >= window_turn and turns.template is not None:
                yield {
                    **turns.template,
                    "text": turns.held.pop(window_turn),
                    "is_listen": False,
                    "model_listen": False,
                    "end_of_turn": False,
                    "user_speech_events": [],
                    "model_turn_id": model_turn_id,
                }
            if final:
                turns.done_turn = window_turn + 1
            samples = _audio_samples(output)
            encoded = ""
            if samples.size and self._encode_audio is not None:
                encoded = self._encode_audio(samples, 24000, response_format, getattr(context, "speed", None))
            if self.is_terminal(request_id):
                continue
            yield {
                "supported": True,
                "data_plane_request_id": request_id,
                "text": "",
                "audio_data": encoded,
                "audio_format": response_format,
                "audio_duration_ms": samples.size * 1000 / 24000 if encoded else 0,
                "audio_text_mark": False,
                "text_requires_complete_audio": False,
                "audio_complete": False,
                "sample_rate_hz": 24000,
                "history_audio_pcm": None,
                "history_audio_supported": False,
                "is_listen": False,
                "model_listen": False,
                "end_of_turn": final,
                "user_speech_events": [],
                "model_turn_id": model_turn_id,
                "runtime_impl": "qwen3_omni_duplex_stage0",
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
            }
        del result


def _audio_samples(output: object) -> np.ndarray:
    """The Stage 2 output's audio, as one float32 array."""

    completion = next(iter(getattr(output, "outputs", ()) or ()), None)
    multimodal = getattr(output, "multimodal_output", None)
    if not isinstance(multimodal, dict) or "audio" not in multimodal:
        multimodal = getattr(completion, "multimodal_output", None) or {}
    audio = multimodal.get("audio") if hasattr(multimodal, "get") else None
    if audio is None:
        return np.empty(0, dtype=np.float32)
    chunks = audio if isinstance(audio, (list, tuple)) else [audio]
    return np.concatenate(
        [
            np.asarray(
                chunk.detach().float().cpu().numpy() if hasattr(chunk, "detach") else chunk, dtype=np.float32
            ).reshape(-1)
            for chunk in chunks
        ]
    )


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
