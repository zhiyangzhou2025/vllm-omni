# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent serving topologies for Qwen3-Omni-Duplex stages.

Each pipeline is selected only by its deploy YAML. The checkpoint still reports
the Qwen3-Omni-MoE architecture, so these entries must not participate in
automatic model detection.
"""

from typing import Any, Literal

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

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
    stage_id: int,
    model_stage: str,
    execution_type: StageExecutionType,
    final_output_type: str,
    engine_output_type: str,
    owns_tokenizer: bool,
    requires_multimodal_data: bool,
    hf_config_name: str,
    sampling_constraints: dict[str, Any] | None = None,
    requires_full_payload_input: bool = False,
    recompute_preemption: Literal["allow", "fail"] = "allow",
) -> StagePipelineConfig:
    return StagePipelineConfig(
        stage_id=stage_id,
        model_stage=model_stage,
        execution_type=execution_type,
        input_sources=(),
        final_output=True,
        final_output_type=final_output_type,
        owns_tokenizer=owns_tokenizer,
        requires_multimodal_data=requires_multimodal_data,
        hf_config_name=hf_config_name,
        engine_output_type=engine_output_type,
        sampling_constraints={"detokenize": owns_tokenizer, **(sampling_constraints or {})},
        requires_full_payload_input=requires_full_payload_input,
        recompute_preemption=recompute_preemption,
    )


QWEN3_OMNI_DUPLEX_THINKER_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_duplex_thinker",
    model_arch=_DUPLEX_ARCH,
    duplex_plugin="vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin.Qwen3OmniDuplexPlugin",
    stages=(
        _stage(
            stage_id=0,
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

QWEN3_OMNI_DUPLEX_TALKER_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_duplex_talker",
    model_arch=_DUPLEX_ARCH,
    stages=(
        _stage(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            final_output_type="latent",
            engine_output_type="latent",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            hf_config_name="talker_config",
            # Only the turn's final window may sample codec_eos; the Talker
            # branch masks it out of every earlier one.
            sampling_constraints={"stop_token_ids": [_CODEC_EOS_TOKEN_ID]},
            # The Talker conditions on the Thinker's projected hidden state, not
            # on the tokens alone, so it waits for the whole upstream payload.
            requires_full_payload_input=True,
            # A decode step feeds the code predictor's summed embedding into
            # PagedAttention while only the layer-0 codec id enters the token
            # stream. Replaying the prompt after a preemption cannot rebuild
            # layers 1..15, so fail the turn instead of resuming it mid-window.
            recompute_preemption="fail",
        ),
    ),
)

QWEN3_OMNI_DUPLEX_CODE2WAV_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_duplex_code2wav",
    model_arch=_ARCH,
    stages=(
        _stage(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            final_output_type="audio",
            engine_output_type="audio",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            hf_config_name="thinker_config",
        ),
    ),
)
