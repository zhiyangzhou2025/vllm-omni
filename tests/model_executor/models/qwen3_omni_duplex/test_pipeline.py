# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.model_executor.models.qwen3_omni_duplex.pipeline import (
    QWEN3_OMNI_DUPLEX_CODE2WAV_PIPELINE,
    QWEN3_OMNI_DUPLEX_TALKER_PIPELINE,
    QWEN3_OMNI_DUPLEX_THINKER_PIPELINE,
)

CODEC_EOS_TOKEN_ID = 2150


def test_both_window_protocol_stages_are_served_by_the_duplex_class() -> None:
    # The Thinker writes the window rows and the Talker reads them back, so one
    # class carries both halves of the protocol. Code2Wav only sees codes.
    assert QWEN3_OMNI_DUPLEX_THINKER_PIPELINE.model_arch == "Qwen3OmniDuplexForConditionalGeneration"
    assert QWEN3_OMNI_DUPLEX_TALKER_PIPELINE.model_arch == "Qwen3OmniDuplexForConditionalGeneration"
    assert QWEN3_OMNI_DUPLEX_CODE2WAV_PIPELINE.model_arch == "Qwen3OmniMoeForConditionalGeneration"


def test_the_talker_stops_on_codec_eos_and_keeps_its_codes_undetokenized() -> None:
    (stage,) = QWEN3_OMNI_DUPLEX_TALKER_PIPELINE.stages

    assert stage.sampling_constraints["stop_token_ids"] == [CODEC_EOS_TOKEN_ID]
    # Codec ids are not text; detokenizing them yields whatever shares the id.
    assert stage.sampling_constraints["detokenize"] is False


def test_a_preempted_talker_turn_fails_instead_of_replaying() -> None:
    (stage,) = QWEN3_OMNI_DUPLEX_TALKER_PIPELINE.stages

    # Only the layer-0 codec id enters the token stream, so replaying the
    # prompt cannot rebuild the residual layers a resumed turn would need.
    assert stage.recompute_preemption == "fail"
    # The Talker conditions on the Thinker's projected hidden state.
    assert stage.requires_full_payload_input is True


def test_the_thinker_stage_keeps_its_own_settings() -> None:
    (stage,) = QWEN3_OMNI_DUPLEX_THINKER_PIPELINE.stages

    assert stage.sampling_constraints == {"detokenize": True}
    assert stage.recompute_preemption == "allow"
    assert stage.requires_full_payload_input is False
