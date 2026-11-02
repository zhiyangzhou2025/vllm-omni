# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the Qwen3-Omni-Duplex Stage 0 duplex plugin."""

import numpy as np
import pybase64 as base64
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import load_duplex_plugin, validate_duplex_plugin_sampling
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin import (
    INPUT_SAMPLE_RATE_HZ,
    Qwen3OmniDuplexPlugin,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.pipeline import (
    QWEN3_OMNI_DUPLEX_THINKER_PIPELINE,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_WINDOW_PROMPT_TOKENS,
    INPUT_SAMPLES_PER_UNIT,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PLUGIN_PATH = "vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin.Qwen3OmniDuplexPlugin"


def _pcm(samples: int) -> dict[str, object]:
    raw = np.zeros(samples, dtype="<f4").tobytes()
    return {
        "audio": base64.b64encode(raw).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": INPUT_SAMPLE_RATE_HZ,
    }


def _plan(plugin: Qwen3OmniDuplexPlugin, payload: object):
    return plugin.plan_append(
        request_id="req",
        fence=DuplexFence(session_id="sess"),
        session_config={},
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload=payload,
        final=False,
        sampling_params=SamplingParams(),
    )


def test_pipeline_declares_the_plugin() -> None:
    assert QWEN3_OMNI_DUPLEX_THINKER_PIPELINE.duplex_plugin == _PLUGIN_PATH
    plugin = load_duplex_plugin(_PLUGIN_PATH, lambda *_args: None)
    assert isinstance(plugin, Qwen3OmniDuplexPlugin)
    validate_duplex_plugin_sampling(plugin, sampling_defaults=(SamplingParams(),))


def test_plan_append_reserves_one_exact_window() -> None:
    plan = _plan(Qwen3OmniDuplexPlugin(lambda *_args: None), _pcm(INPUT_SAMPLES_PER_UNIT))

    prompt = plan.prompt
    assert prompt["prompt_token_ids"] == [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS
    duplex = prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["data_plane"] is True
    assert duplex["model_type"] == "qwen3_omni_duplex"
    assert duplex["scheduler_token_budget"] == AUDIO_WINDOW_PROMPT_TOKENS


@pytest.mark.parametrize("samples", [0, INPUT_SAMPLES_PER_UNIT - 1, INPUT_SAMPLES_PER_UNIT + 1])
def test_plan_append_rejects_any_other_length(samples: int) -> None:
    with pytest.raises(ValueError):
        _plan(Qwen3OmniDuplexPlugin(lambda *_args: None), _pcm(samples))


def test_decide_output_closes_listen_and_speak_on_this_stage() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)

    listen = plugin.decide_output(
        stage_id=0,
        final_stage_id=0,
        segment_finished=True,
        segment_token_ids=(DUPLEX_TOKENS.listen,),
        segment_output_metadata={},
        output=None,
    )
    speak = plugin.decide_output(
        stage_id=0,
        final_stage_id=0,
        segment_finished=True,
        segment_token_ids=(DUPLEX_TOKENS.speak,),
        segment_output_metadata={},
        output=None,
    )
    unfinished = plugin.decide_output(
        stage_id=0,
        final_stage_id=0,
        segment_finished=False,
        segment_token_ids=(DUPLEX_TOKENS.listen,),
        segment_output_metadata={},
        output=None,
    )

    assert listen is not None and listen.metadata["duplex_native_decision"] == "listen"
    assert listen.metadata["model_listen"] is True
    assert speak is not None and speak.metadata["duplex_native_decision"] == "speak"
    assert unfinished is None


def test_capabilities_are_resumable_and_audio_only() -> None:
    capabilities = Qwen3OmniDuplexPlugin(lambda *_args: None).capabilities(max_sessions=1)

    assert capabilities.supports_core_resumable_request is True
    assert capabilities.supports_realtime_endpoint is True
    assert capabilities.required_input_modalities == frozenset({"audio"})
    assert capabilities.chunk_period_ms == 480
    assert capabilities.supports_barge_in is False


def test_silence_unit_is_one_window() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)

    payload = plugin.silence_unit_payload()

    assert payload["sample_rate_hz"] == INPUT_SAMPLE_RATE_HZ
    plan = _plan(plugin, payload)
    assert len(plan.prompt["prompt_token_ids"]) == AUDIO_WINDOW_PROMPT_TOKENS


@pytest.mark.asyncio
async def test_runtime_config_rejects_reference_audio() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)

    config = await plugin.prepare_runtime_config(DuplexSessionConfig(), model_config=None)
    assert config == {"model_type": "qwen3_omni_duplex", "asr_enabled": False}

    with pytest.raises(Exception, match="reference audio"):
        await plugin.prepare_runtime_config(DuplexSessionConfig(ref_audio="clip.wav"), model_config=None)
