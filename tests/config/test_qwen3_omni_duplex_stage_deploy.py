# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the independent Qwen3-Omni-Duplex stage deployments."""

from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_STAGES = (
    ("qwen3_omni_duplex_stage0.yaml", "qwen3_omni_duplex_thinker", "thinker", "text", "Qwen3OmniDuplexForConditionalGeneration"),
    ("qwen3_omni_duplex_stage1.yaml", "qwen3_omni_duplex_talker", "talker", "latent", "Qwen3OmniDuplexForConditionalGeneration"),
    ("qwen3_omni_duplex_stage2.yaml", "qwen3_omni_duplex_code2wav", "code2wav", "audio", "Qwen3OmniMoeForConditionalGeneration"),
)


@pytest.mark.parametrize(("deploy_name", "pipeline_name", "model_stage", "output_type", "model_arch"), _STAGES)
def test_independent_stage_deploy_resolves_one_stage(
    deploy_name: str,
    pipeline_name: str,
    model_stage: str,
    output_type: str,
    model_arch: str,
) -> None:
    deploy_path = Path(get_deploy_config_path(deploy_name))

    pipeline = StageConfigFactory.get_pipeline_config(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        trust_remote_code=True,
        deploy_config_path=str(deploy_path),
        user_deploy_config=load_deploy_config(deploy_path),
    )
    deploy = load_deploy_config(deploy_path)
    stages = merge_pipeline_deploy(pipeline, deploy)

    assert pipeline.model_type == pipeline_name
    assert deploy.pipeline == pipeline_name
    assert deploy.session_mode == "duplex"
    # Each topology serves one module alone, so that module is stage 0 whatever
    # its number in the eventual three-stage deployment.
    assert [stage.stage_id for stage in stages] == [0]
    assert stages[0].model_stage == model_stage
    assert stages[0].final_output is True
    assert stages[0].final_output_type == output_type
    assert stages[0].input_sources == []
    assert stages[0].yaml_engine_args["model_arch"] == model_arch
