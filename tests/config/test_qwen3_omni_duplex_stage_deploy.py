# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the independent Qwen3-Omni-Duplex stage deployments."""

from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _resolve(deploy_name: str):
    deploy_path = Path(get_deploy_config_path(deploy_name))

    pipeline = StageConfigFactory.get_pipeline_config(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        trust_remote_code=True,
        deploy_config_path=str(deploy_path),
        user_deploy_config=load_deploy_config(deploy_path),
    )
    deploy = load_deploy_config(deploy_path)
    return pipeline, deploy, merge_pipeline_deploy(pipeline, deploy)


def test_thinker_deploy_resolves_one_stage() -> None:
    pipeline, deploy, stages = _resolve("qwen3_omni_duplex_stage0.yaml")

    assert pipeline.model_type == "qwen3_omni_duplex_thinker"
    assert deploy.pipeline == "qwen3_omni_duplex_thinker"
    assert deploy.session_mode == "duplex"
    # The topology serves the Thinker alone, so it is stage 0 whatever its
    # number in the eventual three-stage deployment.
    assert [stage.stage_id for stage in stages] == [0]
    assert stages[0].model_stage == "thinker"
    assert stages[0].final_output is True
    assert stages[0].final_output_type == "text"
    assert stages[0].input_sources == []
    assert stages[0].yaml_engine_args["model_arch"] == "Qwen3OmniDuplexForConditionalGeneration"


def test_talker_code2wav_deploy_resolves_the_pair() -> None:
    pipeline, deploy, stages = _resolve("qwen3_omni_duplex_stage12.yaml")

    assert pipeline.model_type == "qwen3_omni_duplex_talker_code2wav"
    assert deploy.pipeline == "qwen3_omni_duplex_talker_code2wav"
    assert deploy.session_mode == "duplex"
    # Renumbered from zero: the Talker heads the pair whatever its number in the
    # eventual three-stage deployment.
    assert [stage.stage_id for stage in stages] == [0, 1]
    assert [stage.model_stage for stage in stages] == ["talker", "code2wav"]
    # The Talker hands its codes down rather than returning them, which is what
    # keeps its own postprocess -- and so the code predictor's input -- alive.
    assert stages[0].final_output is False
    assert stages[0].input_sources == []
    assert stages[1].input_sources == [0]
    assert stages[1].final_output is True
    assert stages[1].final_output_type == "audio"
    assert stages[0].yaml_engine_args["model_arch"] == "Qwen3OmniDuplexForConditionalGeneration"
    assert stages[1].yaml_engine_args["model_arch"] == "Qwen3OmniMoeForConditionalGeneration"
    # Codes stream down one frame per step, which keeps the prefill's all-zero
    # rows out of Code2Wav.
    assert [stage.yaml_engine_args["async_chunk"] for stage in stages] == [True, True]
    assert stages[0].yaml_engine_args["custom_process_next_stage_input_func"] == (
        "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk"
    )
