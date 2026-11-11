# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent serving topologies for Qwen3-Omni-Duplex stages.

Each pipeline is selected only by its deploy YAML. The checkpoint still reports
the Qwen3-Omni-MoE architecture, so these entries must not participate in
automatic model detection.

Every topology here is numbered from zero whatever the module's number in the
three-stage deployment: the orchestrator indexes stage pools by ``stage_id`` and
rejects a topology whose ids are not zero-based and contiguous
(``engine/stage_runtime.py`` ``_build_logical_stage_init_plans``). A deploy YAML
may still name the module's own number under ``stages:`` -- that list selects
which stage's settings to override -- but only when a ``base_config`` supplies a
topology that reaches that number.

The Talker is not served alone, and there is no topology here that does. The
runner invokes a stage's ``postprocess`` only while some request still has a
downstream stage to feed, so a lone Talker loses ``talker_postprocess`` and the
code predictor expands every frame against a zero ``last_talker_hidden``.
``qwen3_omni_duplex_talker_code2wav`` is the smallest topology that keeps that
handoff intact. Code2Wav is likewise never served alone: it has nothing to
consume without the Talker ahead of it.
"""

from typing import Any, Literal

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import AUDIO_TOKENS_PER_UNIT

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_omni"
# Stage 2 reads codes out of the checkpoint's own Code2Wav module and needs
# nothing the duplex protocol adds.
_ARCH = "Qwen3OmniMoeForConditionalGeneration"
# The duplex subclass serves both of the stages that speak the window protocol:
# Stage 0 replaces audio-window placeholders and constrains sampling to the
# protocol, and Stage 1 reads its rows back off the Thinker prompt.
_DUPLEX_ARCH = "Qwen3OmniDuplexForConditionalGeneration"
# ``codec_eos``. Shared with the non-duplex Talker stage.
_CODEC_EOS_TOKEN_ID = 2150


def _stage(
    *,
    model_stage: str,
    execution_type: StageExecutionType,
    final_output_type: str | None,
    engine_output_type: str,
    owns_tokenizer: bool,
    requires_multimodal_data: bool,
    hf_config_name: str,
    stage_id: int = 0,
    input_sources: tuple[int, ...] = (),
    model_arch: str | None = None,
    sampling_constraints: dict[str, Any] | None = None,
    custom_process_next_stage_input_func: str | None = None,
    requires_full_payload_input: bool = False,
    recompute_preemption: Literal["allow", "fail"] = "allow",
) -> StagePipelineConfig:
    return StagePipelineConfig(
        stage_id=stage_id,
        model_stage=model_stage,
        execution_type=execution_type,
        input_sources=input_sources,
        # A stage returns to the client exactly when it names a modality to
        # return, so the two cannot drift apart.
        final_output=final_output_type is not None,
        final_output_type=final_output_type,
        owns_tokenizer=owns_tokenizer,
        requires_multimodal_data=requires_multimodal_data,
        hf_config_name=hf_config_name,
        engine_output_type=engine_output_type,
        model_arch=model_arch,
        sampling_constraints={"detokenize": owns_tokenizer, **(sampling_constraints or {})},
        custom_process_next_stage_input_func=custom_process_next_stage_input_func,
        requires_full_payload_input=requires_full_payload_input,
        recompute_preemption=recompute_preemption,
    )


QWEN3_OMNI_DUPLEX_THINKER_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_duplex_thinker",
    model_arch=_DUPLEX_ARCH,
    duplex_plugin="vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin.Qwen3OmniDuplexPlugin",
    stages=(
        _stage(
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            final_output_type="text",
            engine_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
        ),
    ),
)

QWEN3_OMNI_DUPLEX_TALKER_CODE2WAV_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_duplex_talker_code2wav",
    model_arch=_DUPLEX_ARCH,
    stages=(
        # The Talker hands its codes to Code2Wav instead of returning latents,
        # which is also what keeps its own cross-step state alive: the runner
        # only calls a stage's ``postprocess`` while some request still has a
        # downstream stage to feed (``gpu_ar_model_runner.py``
        # ``_resolve_pooler_payload_req_ids``). Served alone, the Talker's
        # ``talker_postprocess`` never runs and the code predictor reads a zero
        # ``last_talker_hidden`` on every frame.
        _stage(
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            final_output_type=None,
            engine_output_type="latent",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            hf_config_name="talker_config",
            sampling_constraints={
                # Only the turn's final window may sample codec_eos; the Talker
                # branch masks it out of every earlier one.
                "stop_token_ids": [_CODEC_EOS_TOKEN_ID],
                # One window is worth this many codec frames, and one frame
                # costs one decode step. Exhausting the budget does not end a
                # resumable request -- the scheduler returns it to
                # WAITING_FOR_STREAMING_REQ regardless of why it stopped -- so
                # this makes the turn pause once per window and resume on the
                # next one, the same mechanism Stage 0 uses per append.
                "max_tokens": AUDIO_TOKENS_PER_UNIT,
            },
            # Whole-turn handoff only. The official topology also declares
            # ``talker2code2wav_async_chunk`` for streamed 25-frame chunks; that
            # edge reads ``meta.codec_frame_valid`` / ``meta.first_audio``,
            # which the duplex window protocol has not been shown to emit, so it
            # is left unwired rather than declared untested.
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            # The Talker conditions on the Thinker's projected hidden state, not
            # on the tokens alone, so it waits for the whole upstream payload.
            requires_full_payload_input=True,
            # A decode step feeds the code predictor's summed embedding into
            # PagedAttention while only the layer-0 codec id enters the token
            # stream. Replaying the prompt after a preemption cannot rebuild
            # layers 1..15, so fail the turn instead of resuming it mid-window.
            recompute_preemption="fail",
        ),
        _stage(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output_type="audio",
            engine_output_type="audio",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            hf_config_name="thinker_config",
            # Code2Wav reads codes out of the checkpoint's own module and needs
            # nothing the duplex protocol adds, so this one stage overrides the
            # pipeline-wide duplex class back to the official one.
            model_arch=_ARCH,
            sampling_constraints={"detokenize": True},
            requires_full_payload_input=True,
        ),
    ),
)
