# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the Qwen3-Omni-Duplex Stage 0 duplex plugin."""

from types import SimpleNamespace

import numpy as np
import pybase64 as base64
import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import load_duplex_plugin, validate_duplex_plugin_sampling
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin import (
    INPUT_SAMPLE_RATE_HZ,
    WINDOW_STOP_TOKEN_IDS,
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
# ``<|im_start|>system\nStreaming Omni Conversation.<|im_end|>\n``, encoded by the
# checkpoint tokenizer. Tests pass it in; encoding itself needs the checkpoint.
_DEFAULT_PREFIX = (151644, 8948, 198, 76509, 85225, 50830, 13, 151645, 198)


def _pcm(samples: int) -> dict[str, object]:
    raw = np.zeros(samples, dtype="<f4").tobytes()
    return {
        "audio": base64.b64encode(raw).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": INPUT_SAMPLE_RATE_HZ,
    }


def _plan(
    plugin: Qwen3OmniDuplexPlugin,
    payload: object,
    prefix: tuple[int, ...] = _DEFAULT_PREFIX,
    seq: int = 1,
):
    return plugin.plan_append(
        request_id="req",
        fence=DuplexFence(session_id="sess"),
        session_config={},
        runtime_config={"system_prefix_token_ids": list(prefix)},
        seq=seq,
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
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)

    plan = _plan(plugin, _pcm(INPUT_SAMPLES_PER_UNIT))

    prompt = plan.prompt
    assert prompt["prompt_token_ids"] == [
        *_DEFAULT_PREFIX,
        *([DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS),
    ]
    duplex = prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["data_plane"] is True
    assert duplex["model_type"] == "qwen3_omni_duplex"
    assert duplex["scheduler_token_budget"] == len(prompt["prompt_token_ids"])
    # The prefix is staged, not spent, until the stage accepts the append.
    assert plugin._submitted_prefix == {}
    plugin.commit_append_plan("req")
    assert plugin._submitted_prefix["req"] == _DEFAULT_PREFIX


@pytest.mark.parametrize("samples", [0, INPUT_SAMPLES_PER_UNIT - 1, INPUT_SAMPLES_PER_UNIT + 1])
def test_plan_append_rejects_any_other_length(samples: int) -> None:
    with pytest.raises(ValueError):
        _plan(Qwen3OmniDuplexPlugin(lambda *_args: None), _pcm(samples))


def _stage_output(token_ids: tuple[int, ...]) -> SimpleNamespace:
    # DELTA output: the step reports one token, the window lives in the
    # cumulative history the Omni output processor attaches.
    completion = SimpleNamespace(token_ids=token_ids[-1:], cumulative_token_ids=token_ids)
    return SimpleNamespace(outputs=[completion])


def test_decide_output_closes_listen_and_speak_on_this_stage() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)

    def decide(token_ids: tuple[int, ...], *, finished: bool = True) -> object:
        return plugin.decide_output(
            stage_id=0,
            final_stage_id=0,
            segment_finished=finished,
            # The orchestrator snapshots only the closing step's token.
            segment_token_ids=token_ids[-1:],
            segment_output_metadata={},
            output=_stage_output(token_ids),
        )

    listen = decide((DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end))
    speak = decide((DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end))
    unfinished = decide((DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end), finished=False)

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
    assert plan.prompt["prompt_token_ids"][-AUDIO_WINDOW_PROMPT_TOKENS:] == [DUPLEX_TOKENS.unit_start] * (
        AUDIO_WINDOW_PROMPT_TOKENS
    )


@pytest.mark.asyncio
async def test_runtime_config_encodes_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    monkeypatch.setattr(plugin, "_encode_system_prefix", lambda model_config, instructions: (7, 8, 9))

    config = await plugin.prepare_runtime_config(
        DuplexSessionConfig(instructions="Answer briefly."), model_config=object()
    )

    assert config["instructions"] == "Answer briefly."
    assert config["system_prefix_token_ids"] == [7, 8, 9]
    plan = _plan(plugin, _pcm(INPUT_SAMPLES_PER_UNIT), prefix=tuple(config["system_prefix_token_ids"]))
    assert plan.prompt["prompt_token_ids"][:3] == [7, 8, 9]


@pytest.mark.asyncio
async def test_runtime_config_rejects_reference_audio_and_a_changed_prompt() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)

    with pytest.raises(Exception, match="reference audio"):
        await plugin.prepare_runtime_config(DuplexSessionConfig(ref_audio="clip.wav"), model_config=None)
    with pytest.raises(Exception, match="instructions cannot be changed"):
        plugin.runtime_config_for_update(
            DuplexSessionConfig(instructions="new"), {"instructions": "Streaming Omni Conversation."}
        )


def test_later_appends_reserve_the_unit_end_slot() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    first = _plan(plugin, _pcm(INPUT_SAMPLES_PER_UNIT))
    plugin.commit_append_plan("req")

    # The slot is reserved from the sequence number, and the token in it is the
    # literal the scheduler dropped, so nothing has to be carried over.
    second = _plan(plugin, plugin.silence_unit_payload(), seq=2)

    window = [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS
    assert first.prompt["prompt_token_ids"] == [*_DEFAULT_PREFIX, *window]
    # Every window stops on unit_end, so the one slot the previous window leaves
    # behind always holds that token.
    assert second.prompt["prompt_token_ids"] == [DUPLEX_TOKENS.unit_end, *window]
    assert second.prompt["model_intermediate_buffer"]["duplex"]["scheduler_token_budget"] == 1 + len(window)


def test_window_sampling_stops_on_the_protocol_tokens() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)

    (params,) = plugin.configure_sampling_params(runtime_config={}, defaults=(SamplingParams(temperature=0.0),))

    assert params.max_tokens == 16
    assert params.stop_token_ids == list(WINDOW_STOP_TOKEN_IDS)
    assert params.stop_token_ids == [DUPLEX_TOKENS.unit_end]
    # Neither listen nor turn_eos stops the request: the protocol samples one of
    # them and then constrains the next step to unit_end, which is the stop.
    assert DUPLEX_TOKENS.listen not in params.stop_token_ids
    assert DUPLEX_TOKENS.turn_eos not in params.stop_token_ids
    assert DUPLEX_TOKENS.chunk_eos not in params.stop_token_ids
    assert params.temperature == 0.0
    # The whole session is one resumable request, so cumulative output would
    # resend every earlier window's text on every step.
    assert params.output_kind == RequestOutputKind.DELTA
