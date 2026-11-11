# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.talker_row import CODEC_FRAMES_PER_WINDOW
from vllm_omni.model_executor.models.qwen3_omni_duplex.pipeline import (
    QWEN3_OMNI_DUPLEX_CODE2WAV_PIPELINE,
    QWEN3_OMNI_DUPLEX_TALKER_CODE2WAV_PIPELINE,
    QWEN3_OMNI_DUPLEX_TALKER_PIPELINE,
    QWEN3_OMNI_DUPLEX_THINKER_PIPELINE,
)

CODEC_EOS_TOKEN_ID = 2150
PIPELINES = {
    "qwen3_omni_duplex_stage0.yaml": QWEN3_OMNI_DUPLEX_THINKER_PIPELINE,
    "qwen3_omni_duplex_stage1.yaml": QWEN3_OMNI_DUPLEX_TALKER_PIPELINE,
    "qwen3_omni_duplex_stage2.yaml": QWEN3_OMNI_DUPLEX_CODE2WAV_PIPELINE,
    "qwen3_omni_duplex_stage12.yaml": QWEN3_OMNI_DUPLEX_TALKER_CODE2WAV_PIPELINE,
}
DEPLOY_DIR = Path(__file__).parents[4] / "vllm_omni" / "deploy"
# The Talker is served through the Talker+Code2Wav pair, so its own settings are
# asserted on the stage that topology carries. One factory builds the Talker
# stage for both topologies; ``test_the_two_talker_stages_differ_only_in_the_handoff``
# is what keeps that true.
TALKER_STAGE = QWEN3_OMNI_DUPLEX_TALKER_CODE2WAV_PIPELINE.stages[0]


@pytest.mark.parametrize("name", PIPELINES, ids=lambda n: n.removesuffix(".yaml"))
def test_each_debugging_topology_numbers_its_stages_from_zero(name: str) -> None:
    pipeline = PIPELINES[name]
    deploy = yaml.safe_load((DEPLOY_DIR / name).read_text())

    # The orchestrator indexes stage pools by stage_id and refuses a topology
    # whose ids are not zero-based and contiguous, so a pipeline that serves one
    # module on its own puts that module at position 0 -- whatever its number in
    # the eventual three-stage deployment.
    assert [stage.stage_id for stage in pipeline.stages] == list(range(len(pipeline.stages)))
    # And the deploy YAML overrides stages by id, so naming a number the
    # topology does not reach would silently drop every setting under it.
    assert deploy["pipeline"] == pipeline.model_type
    assert [int(stage["stage_id"]) for stage in deploy["stages"]] == list(range(len(pipeline.stages)))


def test_both_window_protocol_stages_are_served_by_the_duplex_class() -> None:
    # The Thinker writes the window rows and the Talker reads them back, so one
    # class carries both halves of the protocol. Code2Wav only sees codes.
    assert QWEN3_OMNI_DUPLEX_THINKER_PIPELINE.model_arch == "Qwen3OmniDuplexForConditionalGeneration"
    assert QWEN3_OMNI_DUPLEX_TALKER_PIPELINE.model_arch == "Qwen3OmniDuplexForConditionalGeneration"
    assert QWEN3_OMNI_DUPLEX_CODE2WAV_PIPELINE.model_arch == "Qwen3OmniMoeForConditionalGeneration"


def test_the_talker_stops_on_codec_eos_and_keeps_its_codes_undetokenized() -> None:
    assert TALKER_STAGE.sampling_constraints["stop_token_ids"] == [CODEC_EOS_TOKEN_ID]
    # Codec ids are not text; detokenizing them yields whatever shares the id.
    assert TALKER_STAGE.sampling_constraints["detokenize"] is False


def test_the_talker_spends_one_decode_step_per_codec_frame() -> None:
    # A resumable turn is not ended by exhausting max_tokens, so the budget
    # paces the turn one window at a time instead of capping it.
    assert TALKER_STAGE.sampling_constraints["max_tokens"] == CODEC_FRAMES_PER_WINDOW


def test_a_preempted_talker_turn_fails_instead_of_replaying() -> None:
    # Only the layer-0 codec id enters the token stream, so replaying the
    # prompt cannot rebuild the residual layers a resumed turn would need.
    assert TALKER_STAGE.recompute_preemption == "fail"
    # The Talker conditions on the Thinker's projected hidden state.
    assert TALKER_STAGE.requires_full_payload_input is True


def test_the_talker_hands_its_codes_down_instead_of_returning_them() -> None:
    talker, code2wav = QWEN3_OMNI_DUPLEX_TALKER_CODE2WAV_PIPELINE.stages

    # The runner calls a stage's postprocess only while some request still has a
    # downstream stage to feed, so the Talker's cross-step handoff --
    # talker_postprocess writing hidden_states["last"] for the code predictor --
    # survives only when it is not the final stage.
    assert talker.final_output is False
    assert talker.final_output_type is None
    assert code2wav.input_sources == (talker.stage_id,)
    assert code2wav.final_output is True
    assert code2wav.final_output_type == "audio"
    # Whole-turn handoff: the async-chunk edge is deliberately not declared.
    assert talker.custom_process_next_stage_input_func == (
        "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_full_payload"
    )
    assert talker.async_chunk_process_next_stage_input_func is None
    # Code2Wav needs nothing the duplex protocol adds, so this one stage
    # overrides the pipeline-wide duplex class back to the official one.
    assert QWEN3_OMNI_DUPLEX_TALKER_CODE2WAV_PIPELINE.model_arch == "Qwen3OmniDuplexForConditionalGeneration"
    assert code2wav.model_arch == "Qwen3OmniMoeForConditionalGeneration"


def test_the_two_talker_stages_differ_only_in_the_handoff() -> None:
    (alone,) = QWEN3_OMNI_DUPLEX_TALKER_PIPELINE.stages

    differing = {"final_output", "final_output_type", "custom_process_next_stage_input_func"}
    for field in fields(TALKER_STAGE):
        if field.name in differing:
            continue
        assert getattr(alone, field.name) == getattr(TALKER_STAGE, field.name), field.name


def test_the_thinker_stage_keeps_its_own_settings() -> None:
    (stage,) = QWEN3_OMNI_DUPLEX_THINKER_PIPELINE.stages

    assert stage.sampling_constraints == {"detokenize": True}
    assert stage.recompute_preemption == "allow"
    assert stage.requires_full_payload_input is False
