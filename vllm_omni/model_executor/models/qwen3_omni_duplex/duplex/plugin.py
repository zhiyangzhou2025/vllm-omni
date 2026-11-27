# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duplex plugin for the Qwen3-Omni-Duplex Stage 0 Thinker."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, DuplexRuntimeConfigError
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.data_plane import (
    FUNCTION_CALL_KEY,
    SYSTEM2_CANCEL,
    SYSTEM2_LAUNCH,
    Qwen3OmniDuplexDataPlane,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    MAX_GENERATED_TOKENS_PER_WINDOW,
    window_decision,
    window_token_ids,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.session import (
    Qwen3OmniDuplexSessionState,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    INPUT_SAMPLE_RATE_HZ,
    INPUT_SAMPLES_PER_UNIT,
    TOOL_BLOCK_END,
    TOOL_BLOCK_NEWLINE,
    TOOL_BLOCK_START,
    UNIT_DURATION_MS,
    build_audio_window_prompt,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.tool_calls import ToolCallOrigin, ToolCallSettlement

logger = init_logger(__name__)

PLUGIN_ID = "qwen3_omni_duplex"
# A System 2 result enters the Thinker's context in the reference runtime's form.
TOOL_BLOCK_TEMPLATE = "<|im_start|>tool\n{content}<|im_end|>\n"
# The accepted result the next append carries, and the private prompt key that
# names it once that append is planned.
SYSTEM2_PENDING_KEY = "system2_pending_result"
_SYSTEM2_SUBMITTED_KEY = "_qwen3_duplex_system2_submitted"
# The session's system prompt, wrapped in the chat template. A client that sets
# no instructions gets the reference runtime's prompt.
DEFAULT_INSTRUCTIONS = "Streaming Omni Conversation."
SYSTEM_PREFIX_TEMPLATE = "<|im_start|>system\n{instructions}<|im_end|>\n"
# ``<|unit_end|>`` ends every window, for listen and speak alike. Neither
# ``<|listen|>`` nor ``<|turn_eos|>`` stops the request: the protocol samples
# one of them and then constrains the next step to ``<|unit_end|>``.
WINDOW_STOP_TOKEN_IDS = (DUPLEX_TOKENS.unit_end,)


def _modalities(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(item).lower() for item in values)


def _modality_set(values: object) -> frozenset[str]:
    return frozenset(_modalities(values))


def _effective_instructions(value: object) -> str:
    # An omitted field, null, and an empty string all mean the default sentence.
    if isinstance(value, str) and value:
        return value
    return DEFAULT_INSTRUCTIONS


class Qwen3OmniDuplexPlugin(DuplexModelPlugin):
    """Serve Stage 0 as one resumable request of exact audio windows."""

    plugin_id = PLUGIN_ID
    silence_continuation_samples = INPUT_SAMPLES_PER_UNIT
    silence_continuation_sample_rate_hz = INPUT_SAMPLE_RATE_HZ
    # Stage 0's text reaches the client even when a later stage is the final one.
    projects_intermediate_outputs = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # The data plane holds per-session state, so each plugin instance owns one.
        encode_audio = args[0] if args else kwargs.get("encode_audio")
        self.data_plane = Qwen3OmniDuplexDataPlane(encode_audio)
        self.data_plane.on_close_session = self._close_session
        self._request_session_keys: dict[str, str] = {}
        # The system prefix is staged until the stage accepts the first append, so
        # a rejected append does not consume it.
        self._pending_prefix: dict[str, tuple[int, ...]] = {}
        self._submitted_prefix: dict[str, tuple[int, ...]] = {}
        # A speak turn's output modalities, snapshotted when its first window
        # closes. ``None`` means no turn is open. The live session value can
        # change while the turn's later windows are still being generated, so
        # those windows read this snapshot rather than the attribute below.
        self._speak_turn_modalities: dict[str, tuple[str, ...]] = {}
        # Kept so a session update before the first window can re-encode the prefix.
        self._instructions_model_config: object | None = None
        # The System 2 result each planned append carries, until the stage accepts it.
        self._pending_system2: dict[str, Mapping[str, object]] = {}
        self._tokenizers: dict[str, object] = {}

    def configure_pipeline(self, pipeline_config: object) -> None:
        stages = getattr(pipeline_config, "stages", ()) or ()
        code2wav = next((stage.stage_id for stage in stages if getattr(stage, "model_stage", None) == "code2wav"), None)
        self.data_plane.set_audio_stage(code2wav)

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        # One append is one window. Decoding runs until the window closes, which
        # the protocol tokens signal, or until the window budget runs out.
        # The window budget belongs to Stage 0. A later stage keeps its own
        # max_tokens; only the output kind is shared, because the session is one
        # resumable request and CUMULATIVE would repeat every window's output.
        configured = []
        for index, default in enumerate(defaults):
            if not isinstance(default, SamplingParams):
                configured.append(default)
                continue
            params = default.clone()
            params.output_kind = RequestOutputKind.DELTA
            if index == 0:
                params.max_tokens = MAX_GENERATED_TOKENS_PER_WINDOW
                params.stop_token_ids = list(WINDOW_STOP_TOKEN_IDS)
            configured.append(params)
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        decode_pcm_f32le_payload(
            payload,
            sample_rate_hz=INPUT_SAMPLE_RATE_HZ,
            exact_samples=INPUT_SAMPLES_PER_UNIT,
            model="Qwen3-Omni-Duplex",
        )
        # Only the first append carries tokens of its own: the system prefix.
        # Later appends close the previous window with ``<|unit_end|>``, which
        # ``build_audio_window_prompt`` reserves from the sequence number.
        self._request_session_keys[request_id] = str(runtime_config.get("duplex_session_key", request_id))
        prefix = () if request_id in self._submitted_prefix else self._system_prefix_token_ids(runtime_config)
        # An accepted System 2 result enters ahead of this window. It stays in
        # the runtime until the stage accepts this append.
        result = runtime_config.get(SYSTEM2_PENDING_KEY)
        tool_token_ids: list[int] = []
        if isinstance(result, Mapping):
            tool_token_ids = [int(token_id) for token_id in result["token_ids"]]
            payload = {**dict(payload), "system2_result": "final" if result["final"] else "partial"}  # type: ignore[call-overload]
            self._pending_system2[request_id] = result
        else:
            self._pending_system2.pop(request_id, None)
        prompt = build_audio_window_prompt(
            request_id=request_id,
            fence=fence,
            session_config=session_config,
            runtime_config={key: value for key, value in runtime_config.items() if key != SYSTEM2_PENDING_KEY},
            seq=seq,
            turn_seq=turn_seq,
            payload=payload,
            final=final,
            prefix_token_ids=prefix,
            tool_token_ids=tool_token_ids,
        )
        if isinstance(result, Mapping):
            prompt[_SYSTEM2_SUBMITTED_KEY] = result["call_id"]
        self._pending_prefix[request_id] = prefix
        return DuplexAppendPlan(prompt=prompt)

    def runtime_patch_for_submitted_prompt(self, prompt: dict[str, object]) -> Mapping[str, object] | None:
        call_id = prompt.pop(_SYSTEM2_SUBMITTED_KEY, None)
        return None if call_id is None else {_SYSTEM2_SUBMITTED_KEY: call_id}

    def reconcile_submitted_runtime(
        self, current: Mapping[str, object], updated: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Drop the submitted result only; one accepted while the append was planned stays."""
        reconciled = dict(updated)
        call_id = reconciled.pop(_SYSTEM2_SUBMITTED_KEY, None)
        if call_id is None:
            return None
        pending = current.get(SYSTEM2_PENDING_KEY)
        if isinstance(pending, Mapping) and pending.get("call_id") == call_id:
            reconciled.pop(SYSTEM2_PENDING_KEY, None)
        return reconciled

    def parse_function_call(self, model_result: Mapping[str, object]) -> dict[str, str] | None:
        call = model_result.get(FUNCTION_CALL_KEY)
        return dict(call) if isinstance(call, Mapping) else None

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
        *,
        history_len: int | None = None,
    ) -> dict[str, object] | None:
        """Render a launch result as the ToolBlock the next window carries.

        A cancel acknowledgement only settles its call, and so does any output
        for a launch whose generation ended while its result was owed. Any
        other launch result must answer the call its generation still awaits,
        and its ``output`` is ``{"content": str, "final": bool}``; ``content``
        enters verbatim.
        """
        del config, history_len
        call_id, name = item.get("call_id"), item.get("name")
        if name == SYSTEM2_CANCEL or call_id in self.data_plane.abandoned_calls:
            return None
        if name != SYSTEM2_LAUNCH or not isinstance(call_id, str):
            raise DuplexRuntimeConfigError(
                f"Qwen3-Omni-Duplex has no System 2 call {call_id!r}", code="invalid_function_call_output"
            )
        request_id = self.data_plane.call_requests.get(call_id)
        generation = self.data_plane.system2.get(request_id) if request_id is not None else None
        if generation is None or generation.awaiting != call_id:
            raise DuplexRuntimeConfigError(
                f"System 2 call {call_id} is not the one its generation awaits", code="invalid_function_call_output"
            )
        try:
            output = json.loads(item.get("output"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            output = None
        if (
            not isinstance(output, dict)
            or not isinstance(output.get("content"), str)
            or not isinstance(output.get("final"), bool)
        ):
            raise DuplexRuntimeConfigError(
                'a System 2 result is the JSON object {"content": str, "final": bool}',
                code="invalid_function_call_output",
            )
        token_ids = self._encode_tool_block(output["content"])
        if current.get(SYSTEM2_PENDING_KEY) is not None:
            logger.warning("System 2 result %s replaces one the Thinker has not read", call_id)
        return {
            **current,
            SYSTEM2_PENDING_KEY: {
                "call_id": call_id,
                "generation_id": generation.generation_id,
                "token_ids": token_ids,
                "final": output["final"],
            },
        }

    def on_tool_call_settled(self, settlement: ToolCallSettlement) -> None:
        self.data_plane.settled_call(settlement.origin.call_id)

    def on_tool_calls_closed(self, origins: tuple[ToolCallOrigin, ...]) -> None:
        for origin in origins:
            self.data_plane.settled_call(origin.call_id)

    def retire_pending_function_output(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object] | None:
        del config
        if current.get(SYSTEM2_PENDING_KEY) is None:
            return None
        return {key: value for key, value in current.items() if key != SYSTEM2_PENDING_KEY}

    def _encode_tool_block(self, content: str) -> list[int]:
        tokenizer = self._tokenizer(self._instructions_model_config)
        if tokenizer is None:
            raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 cannot encode a System 2 result")
        text = TOOL_BLOCK_TEMPLATE.format(content=content)
        token_ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]  # type: ignore[attr-defined]
        special = {int(token_id) for token_id in tokenizer.added_tokens_decoder}  # type: ignore[attr-defined]
        special.update(int(token_id) for token_id in tokenizer.all_special_ids)  # type: ignore[attr-defined]
        # Only the wrapper may be special: a special token in the content would
        # read as protocol, and a stray <|im_end|> would end the block early.
        if (
            token_ids[:1] != [TOOL_BLOCK_START]
            or token_ids[-2:] != [TOOL_BLOCK_END, TOOL_BLOCK_NEWLINE]
            or any(token_id in special for token_id in token_ids[1:-2])
        ):
            raise DuplexRuntimeConfigError(
                "a System 2 result's content must not contain special tokens", code="invalid_function_call_output"
            )
        return token_ids

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        del segment_token_ids, segment_output_metadata
        if not segment_finished:
            return None
        # A stage past Stage 0 has nothing to decide: its output is audio, and the
        # orchestrator forwards it on its own.
        if stage_id > 0:
            return None
        # The segment snapshot holds only the step that closed the window, so
        # the decision comes from the output's own cumulative token history.
        decision = window_decision(window_token_ids(output))
        if decision is None:
            return None
        native_decision = "listen" if decision == "listen" else "speak"
        request_id = getattr(output, "request_id", "")
        session_key = self._request_session_keys.get(request_id, request_id)
        locked = self._speak_turn_modalities.get(session_key)
        # The first window of a speak turn freezes the modalities the Talker
        # will see. A text-only turn never starts the Talker; a later window of
        # an audio turn cannot be dropped without leaving the Talker mid-turn.
        if decision != "listen" and locked is None:
            bound = getattr(self, "output_modalities", None)
            if bound is not None:
                locked = self._speak_turn_modalities[session_key] = _modalities(bound)
        wants_audio = locked is None or "audio" in locked
        if decision == "turn_end":
            self._speak_turn_modalities.pop(session_key, None)
        # A speak window still has its audio to produce, so it goes to the next
        # stage. A listen window is finished here, and so is a speak window when
        # Stage 0 is the last stage or the turn was opened without audio.
        if stage_id < final_stage_id and decision != "listen" and wants_audio:
            return None
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata={
                "duplex_direct_response": True,
                "duplex_native_decision": native_decision,
                "model_listen": native_decision == "listen",
            },
        )

    def commit_append_plan(self, request_id: str) -> None:
        prefix = self._pending_prefix.pop(request_id, None)
        if prefix is not None:
            self._submitted_prefix[request_id] = prefix
        result = self._pending_system2.pop(request_id, None)
        if result is not None:
            self.data_plane.submitted_result(request_id, str(result["generation_id"]), final=bool(result["final"]))

    def _close_session(self, session_id: str) -> None:
        from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session

        for request_id in list(self._request_session_keys):
            if duplex_resource_request_belongs_to_session(request_id, session_id):
                key = self._request_session_keys.pop(request_id)
                self._speak_turn_modalities.pop(key, None)
                self._pending_prefix.pop(request_id, None)
                self._submitted_prefix.pop(request_id, None)
                self._pending_system2.pop(request_id, None)

    def create_session_state(self) -> Qwen3OmniDuplexSessionState:
        return Qwen3OmniDuplexSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        supports_multi_session = max_sessions > 1
        return DuplexCapabilities(
            supports_model_native_turn_policy=True,
            supports_external_turn_signal=False,
            supports_client_commit=False,
            supports_barge_in=False,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_turn_commit_only=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=False,
            required_input_modalities=frozenset({"audio"}),
            optional_input_modalities=frozenset(),
            supports_independent_io_streams=False,
            supports_realtime_endpoint=True,
            supports_multi_session=supports_multi_session,
            supports_multi_session_same_replica=supports_multi_session,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_audio_truncate=False,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            adapter_patterns=["scheduler_data_plane"],
            signal_sources=["model_native"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=UNIT_DURATION_MS,
            target_barge_in_latency_ms=None,
        )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: object | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 does not take reference audio")
        # Record the sentence actually encoded. Later updates compare against
        # this, not against a missing field.
        self._instructions_model_config = model_config
        effective = _effective_instructions(config.instructions)
        config.instructions = effective
        prefix = self._encode_system_prefix(model_config, effective)
        if prefix is None:
            raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 cannot encode the system prompt")
        return {
            "model_type": PLUGIN_ID,
            "instructions": effective,
            "system_prefix_token_ids": list(prefix),
            "duplex_session_key": uuid4().hex,
        }

    def drops_locked_instruction_updates(self) -> bool:
        return True

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        updated = dict(current)
        requested = _effective_instructions(config.instructions)
        current_effective = _effective_instructions(updated.get("instructions"))
        if requested != current_effective:
            # The prefix is already in the KV cache. Drop the new sentence and
            # keep adopting the rest of this update.
            session_key = current.get("duplex_session_key")
            if any(self._request_session_keys.get(request_id) == session_key for request_id in self._submitted_prefix):
                config.instructions = current_effective
            else:
                prefix = self._encode_system_prefix(self._instructions_model_config, requested)
                if prefix is None:
                    raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 cannot encode the system prompt")
                config.instructions = requested
                updated["instructions"] = requested
                updated["system_prefix_token_ids"] = list(prefix)
        else:
            config.instructions = current_effective
            if updated.get("instructions") != current_effective:
                updated["instructions"] = current_effective
        # A speak turn already snapshotted its modalities. Drop a different set
        # and keep the other fields. The same set in another order is not a change.
        locked = self._speak_turn_modalities.get(str(current.get("duplex_session_key", "")))
        if locked is not None and _modality_set(config.modalities) != _modality_set(locked):
            config.modalities = list(locked)
        return updated

    def _tokenizer(self, model_config: object | None) -> object | None:
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            return None
        if model_path not in self._tokenizers:
            from transformers import AutoTokenizer

            try:
                self._tokenizers[model_path] = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, local_files_only=True
                )
            except (OSError, ValueError):
                return None
        return self._tokenizers[model_path]

    def _encode_system_prefix(self, model_config: object | None, instructions: str) -> tuple[int, ...] | None:
        tokenizer = self._tokenizer(model_config)
        if tokenizer is None:
            return None
        try:
            token_ids = tokenizer.encode(  # type: ignore[attr-defined]
                SYSTEM_PREFIX_TEMPLATE.format(instructions=instructions), add_special_tokens=False
            )
        except ValueError:
            return None
        if not token_ids:
            return None
        return tuple(int(token_id) for token_id in token_ids)

    def _system_prefix_token_ids(self, runtime_config: Mapping[str, object]) -> tuple[int, ...]:
        raw = runtime_config.get("system_prefix_token_ids")
        if not isinstance(raw, (list, tuple)) or not raw:
            raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 has no encoded system prompt")
        try:
            return tuple(int(token_id) for token_id in raw)
        except (TypeError, ValueError) as exc:
            raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 system prompt is not encoded") from exc
