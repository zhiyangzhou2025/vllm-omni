# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duplex plugin for the Qwen3-Omni-Duplex Stage 0 Thinker."""

from __future__ import annotations

from collections.abc import Mapping

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
    INPUT_SAMPLES_PER_UNIT,
    build_audio_window_prompt,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

PLUGIN_ID = "qwen3_omni_duplex"
INPUT_SAMPLE_RATE_HZ = 16000
# One 0.48-second window. The runner's silence continuation uses the same unit.
UNIT_DURATION_MS = 480
# The session's system prompt, wrapped in the chat template. A client that sets
# no instructions gets the reference runtime's prompt.
DEFAULT_INSTRUCTIONS = "Streaming Omni Conversation."
SYSTEM_PREFIX_TEMPLATE = "<|im_start|>system\n{instructions}<|im_end|>\n"
# ``<|unit_end|>`` ends every window, for listen and speak alike. Neither
# ``<|listen|>`` nor ``<|turn_eos|>`` stops the request: the protocol samples
# one of them and then constrains the next step to ``<|unit_end|>``.
WINDOW_STOP_TOKEN_IDS = (DUPLEX_TOKENS.unit_end,)


class Qwen3OmniDuplexPlugin(DuplexModelPlugin):
    """Serve Stage 0 as one resumable request of exact audio windows."""

    plugin_id = PLUGIN_ID
    silence_continuation_samples = INPUT_SAMPLES_PER_UNIT
    silence_continuation_sample_rate_hz = INPUT_SAMPLE_RATE_HZ
    data_plane = Qwen3OmniDuplexDataPlane()

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # The system prefix is staged until the stage accepts the first append, so
        # a rejected append does not consume it.
        self._pending_prefix: dict[str, tuple[int, ...]] = {}
        self._submitted_prefix: dict[str, tuple[int, ...]] = {}

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        # One append is one window. Decoding runs until the window closes, which
        # the protocol tokens signal, or until the window budget runs out.
        configured = []
        for default in defaults:
            if not isinstance(default, SamplingParams):
                configured.append(default)
                continue
            params = default.clone()
            params.max_tokens = MAX_GENERATED_TOKENS_PER_WINDOW
            params.stop_token_ids = list(WINDOW_STOP_TOKEN_IDS)
            # The session is one resumable request, so a window boundary never
            # resets the output state. CUMULATIVE would hand the data plane the
            # whole session's text again on every decode step.
            params.output_kind = RequestOutputKind.DELTA
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
        prefix = () if request_id in self._submitted_prefix else self._system_prefix_token_ids(runtime_config)
        prompt = build_audio_window_prompt(
            request_id=request_id,
            fence=fence,
            session_config=session_config,
            runtime_config=runtime_config,
            seq=seq,
            turn_seq=turn_seq,
            payload=payload,
            final=final,
            prefix_token_ids=prefix,
        )
        self._pending_prefix[request_id] = prefix
        return DuplexAppendPlan(prompt=prompt)

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
        del stage_id, final_stage_id, segment_token_ids, segment_output_metadata
        if not segment_finished:
            return None
        # The segment snapshot holds only the step that closed the window, so
        # the decision comes from the output's own cumulative token history.
        decision = window_decision(window_token_ids(output))
        if decision is None:
            return None
        native_decision = "listen" if decision == "listen" else "speak"
        # No downstream stage exists yet, so both decisions close the segment
        # here. The sampled token is what the client sees.
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
        instructions = config.instructions if isinstance(config.instructions, str) else ""
        prefix = self._encode_system_prefix(model_config, instructions or DEFAULT_INSTRUCTIONS)
        if prefix is None:
            raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 cannot encode the system prompt")
        return {
            "model_type": PLUGIN_ID,
            "instructions": instructions or DEFAULT_INSTRUCTIONS,
            "system_prefix_token_ids": list(prefix),
        }

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        # The prefix is already in the KV cache, so a later session update
        # cannot change it.
        if config.instructions != current.get("instructions"):
            raise DuplexRuntimeConfigError(
                "instructions cannot be changed after the session is created",
                code="instructions_update_unsupported",
            )
        return dict(current)

    def _encode_system_prefix(self, model_config: object | None, instructions: str) -> tuple[int, ...] | None:
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            return None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            token_ids = tokenizer.encode(
                SYSTEM_PREFIX_TEMPLATE.format(instructions=instructions), add_special_tokens=False
            )
        except (OSError, ValueError):
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
