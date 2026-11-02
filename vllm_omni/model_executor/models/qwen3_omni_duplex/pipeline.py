# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent serving topologies for Qwen3-Omni-Duplex stages.

Each pipeline is selected only by its deploy YAML. The checkpoint still reports
the Qwen3-Omni-MoE architecture, so these entries must not participate in
automatic model detection.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_ARCH = "Qwen3OmniMoeForConditionalGeneration"


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
        sampling_constraints={"detokenize": owns_tokenizer},
    )


QWEN3_OMNI_DUPLEX_THINKER_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_duplex_thinker",
    model_arch=_ARCH,
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
    model_arch=_ARCH,
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
