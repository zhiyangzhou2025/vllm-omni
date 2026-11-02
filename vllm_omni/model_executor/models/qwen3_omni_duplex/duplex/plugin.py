# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duplex plugin for the Qwen3-Omni-Duplex Stage 0 Thinker."""

from __future__ import annotations

from collections.abc import Mapping

from vllm.sampling_params import SamplingParams

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


class Qwen3OmniDuplexPlugin(DuplexModelPlugin):
    """Serve Stage 0 as one resumable request of exact audio windows."""

    plugin_id = PLUGIN_ID
    silence_continuation_samples = INPUT_SAMPLES_PER_UNIT
    silence_continuation_sample_rate_hz = INPUT_SAMPLE_RATE_HZ
    data_plane = Qwen3OmniDuplexDataPlane()

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        # The window prefill samples nothing. One decode step decides listen or
        # speak; the runner resubmits the same request for the next step.
        return tuple(
            default.clone() if isinstance(default, SamplingParams) else default for default in defaults
        )

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
        return DuplexAppendPlan(
            prompt=build_audio_window_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload=payload,
                final=final,
            )
        )

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
        del stage_id, final_stage_id, segment_output_metadata, output
        if not segment_finished or not segment_token_ids:
            return None
        decision = segment_token_ids[-1]
        if decision == DUPLEX_TOKENS.listen:
            native_decision = "listen"
        elif decision == DUPLEX_TOKENS.speak:
            native_decision = "speak"
        else:
            return None
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
        del model_config
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("Qwen3-Omni-Duplex Stage 0 does not take reference audio")
        return {"model_type": PLUGIN_ID, "asr_enabled": False}

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        del config
        return dict(current)
