# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 output projection for Qwen3-Omni-Duplex.

Stage 1 does not exist yet, so a window's text is the final output. Audio is
produced by Stage 2 and is not invented here.
"""

from __future__ import annotations

from collections.abc import Iterable

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    window_decision,
    window_text_is_assistant,
    window_token_ids,
)


class Qwen3OmniDuplexDataPlane(DuplexDataPlane):
    def __init__(self) -> None:
        self.terminal: set[str] = set()

    def begin_request(self, request_id: str) -> None:
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
            # A ``transcript`` window transcribes the user before it decides what
            # to do. Those tokens are the user's words, not the assistant's.
            if not isinstance(text, str) or not window_text_is_assistant(token_ids):
                text = ""
            # ``decide_output`` never runs here: the runner calls it only for
            # stages below the final one, and Stage 0 is the only stage.
            decision = window_decision(token_ids)
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
                "model_turn_id": model_turn_id,
                "runtime_impl": "qwen3_omni_duplex_stage0",
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
            }
