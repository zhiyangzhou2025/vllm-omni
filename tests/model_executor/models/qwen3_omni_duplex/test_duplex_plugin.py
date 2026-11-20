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
    DEFAULT_INSTRUCTIONS,
    INPUT_SAMPLE_RATE_HZ,
    WINDOW_STOP_TOKEN_IDS,
    Qwen3OmniDuplexPlugin,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    closed_window_count,
    window_text_role,
    window_user_activity,
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
    return SimpleNamespace(request_id="req", outputs=[completion])


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

    # A deployment with a stage after Stage 0 forwards the speak window and still
    # closes the listen window here.
    forwarded = plugin.decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(DUPLEX_TOKENS.unit_end,),
        segment_output_metadata={},
        output=_stage_output((DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end)),
    )
    kept = plugin.decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(DUPLEX_TOKENS.unit_end,),
        segment_output_metadata={},
        output=_stage_output((DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end)),
    )
    assert forwarded is None
    assert kept is not None and kept.metadata["duplex_native_decision"] == "listen"


def _decide(plugin: Qwen3OmniDuplexPlugin, token_ids: tuple[int, ...], *, modalities: tuple[str, ...] | None) -> object:
    if modalities is None:
        plugin.__dict__.pop("output_modalities", None)
    else:
        plugin.output_modalities = modalities
    return plugin.decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(token_ids[-1],),
        segment_output_metadata={},
        output=_stage_output(token_ids),
    )


def test_text_only_speak_turn_is_not_forwarded_and_stays_locked() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    speak = (DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end)
    later = (DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end, DUPLEX_TOKENS.speak, 101, DUPLEX_TOKENS.unit_end)

    first = _decide(plugin, speak, modalities=("text",))
    # A live change during the turn does not reopen the Talker. The snapshot
    # taken when the turn opened is what the later window is judged against.
    second = _decide(plugin, later, modalities=("text", "audio"))

    assert first is not None and first.metadata["duplex_native_decision"] == "speak"
    assert second is not None and second.metadata["duplex_native_decision"] == "speak"
    assert plugin._speak_turn_modalities.get("req") == ("text",)
    # A different set is dropped. The same set in another order is kept.
    changed = DuplexSessionConfig(modalities=["text", "audio"], temperature=0.2)
    assert plugin.runtime_config_for_update(changed, {"instructions": None, "duplex_session_key": "req"})["instructions"] == DEFAULT_INSTRUCTIONS
    assert changed.modalities == ["text"]
    assert changed.temperature == 0.2
    reordered = DuplexSessionConfig(modalities=["audio", "text"])
    plugin._speak_turn_modalities["req"] = ("text", "audio")
    plugin.runtime_config_for_update(reordered, {"instructions": DEFAULT_INSTRUCTIONS, "duplex_session_key": "req"})
    assert reordered.modalities == ["audio", "text"]


def test_speak_turn_lock_clears_on_turn_eos_and_the_next_turn_can_enable_audio() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    ending = (DUPLEX_TOKENS.speak, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end)
    nxt = (*ending, DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end)

    closed = _decide(plugin, ending, modalities=("text",))
    assert closed is not None
    assert not plugin._speak_turn_modalities
    assert plugin.runtime_config_for_update(DuplexSessionConfig(modalities=["audio"]), {"instructions": None, "duplex_session_key": "req"})[
        "instructions"
    ] == DEFAULT_INSTRUCTIONS
    forwarded = _decide(plugin, nxt, modalities=("text", "audio"))
    assert forwarded is None
    assert plugin._speak_turn_modalities.get("req") == ("text", "audio")


def test_listen_window_does_not_open_or_clear_the_speak_turn_lock() -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    listen = (DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end)

    assert _decide(plugin, listen, modalities=("text",)) is not None
    assert not plugin._speak_turn_modalities

    speak = (DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end)
    assert _decide(plugin, speak, modalities=("text", "audio")) is None
    mid_listen = (*speak, DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end)
    assert _decide(plugin, mid_listen, modalities=("text",)) is not None
    assert plugin._speak_turn_modalities.get("req") == ("text", "audio")


def test_interleaved_sessions_keep_turn_modalities_and_instructions_separate() -> None:
    from vllm_omni.engine.duplex.contracts import duplex_resource_request_id

    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    speak = (DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end)
    ending = (*speak, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end)
    requests = [duplex_resource_request_id(DuplexFence(session_id=name), "stage0") for name in ("one", "two")]
    for request_id, key in zip(requests, ("key-one", "key-two")):
        plugin._request_session_keys[request_id] = key
        plugin._submitted_prefix[request_id] = _DEFAULT_PREFIX

    def decide(request_id: str, tokens: tuple[int, ...], modalities: tuple[str, ...]) -> object:
        plugin.output_modalities = modalities
        output = _stage_output(tokens)
        output.request_id = request_id
        return plugin.decide_output(
            stage_id=0, final_stage_id=2, segment_finished=True,
            segment_token_ids=tokens[-1:], segment_output_metadata={}, output=output,
        )

    assert decide(requests[0], speak, ("text",)) is not None
    assert decide(requests[1], speak, ("text", "audio")) is None
    assert plugin._speak_turn_modalities == {"key-one": ("text",), "key-two": ("text", "audio")}
    updated = DuplexSessionConfig(modalities=["audio"])
    plugin.runtime_config_for_update(updated, {"instructions": None, "duplex_session_key": "key-one"})
    assert updated.modalities == ["text"]
    updated = DuplexSessionConfig(modalities=["text"])
    plugin.runtime_config_for_update(updated, {"instructions": None, "duplex_session_key": "key-two"})
    assert updated.modalities == ["text", "audio"]
    plugin._submitted_prefix.pop(requests[1])
    plugin._encode_system_prefix = lambda _model, _instructions: (101, 102)
    changed = DuplexSessionConfig(instructions="new prompt")
    runtime = plugin.runtime_config_for_update(
        changed, {"instructions": None, "duplex_session_key": "key-two"}
    )
    assert changed.instructions == "new prompt"
    assert runtime["system_prefix_token_ids"] == [101, 102]
    locked = DuplexSessionConfig(instructions="new prompt")
    plugin.runtime_config_for_update(locked, {"instructions": None, "duplex_session_key": "key-one"})
    assert locked.instructions == DEFAULT_INSTRUCTIONS
    plugin._submitted_prefix[requests[1]] = _DEFAULT_PREFIX
    assert decide(requests[0], ending, ("text",)) is not None
    assert decide(requests[1], speak, ("text", "audio")) is None
    plugin.data_plane.close_session("one")
    assert requests[0] not in plugin._submitted_prefix
    assert requests[1] in plugin._submitted_prefix
    assert plugin._speak_turn_modalities == {"key-two": ("text", "audio")}


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
    omitted = DuplexSessionConfig()
    omitted_runtime = await plugin.prepare_runtime_config(omitted, model_config=object())
    assert omitted.instructions == DEFAULT_INSTRUCTIONS
    assert omitted_runtime["instructions"] == DEFAULT_INSTRUCTIONS
    plan = _plan(plugin, _pcm(INPUT_SAMPLES_PER_UNIT), prefix=tuple(config["system_prefix_token_ids"]))
    assert plan.prompt["prompt_token_ids"][:3] == [7, 8, 9]


@pytest.mark.asyncio
async def test_runtime_config_rejects_reference_audio_and_keeps_a_locked_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    monkeypatch.setattr(plugin, "_encode_system_prefix", lambda model_config, instructions: (1, 2, 3))

    with pytest.raises(Exception, match="reference audio"):
        await plugin.prepare_runtime_config(DuplexSessionConfig(ref_audio="clip.wav"), model_config=None)

    # Before the prefix is submitted, a real change is re-encoded.
    pending = DuplexSessionConfig(instructions="new", modalities=["text"])
    reencoded = plugin.runtime_config_for_update(pending, {"instructions": DEFAULT_INSTRUCTIONS})
    assert pending.instructions == "new"
    assert reencoded["instructions"] == "new"
    assert reencoded["system_prefix_token_ids"] == [1, 2, 3]

    # After it is in the KV cache, the new sentence is dropped and modalities stay.
    plugin._submitted_prefix["req"] = (1, 2, 3)
    locked = DuplexSessionConfig(instructions="newer", modalities=["text"])
    kept = plugin.runtime_config_for_update(locked, {"instructions": "new", "system_prefix_token_ids": [1, 2, 3]})
    assert locked.instructions == "new"
    assert locked.modalities == ["text"]
    assert kept["instructions"] == "new"
    assert kept["system_prefix_token_ids"] == [1, 2, 3]

    # Resending the sentence in use, or omitting it, is not a change.
    echo = DuplexSessionConfig(instructions=None, modalities=["audio"])
    plugin.runtime_config_for_update(echo, {"instructions": "new"})
    assert echo.instructions == "new"
    assert echo.modalities == ["audio"]


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

    # The window budget is Stage 0's. A later stage keeps the budget its pipeline
    # declared and only shares the output kind.
    stage0, stage1 = plugin.configure_sampling_params(
        runtime_config={}, defaults=(SamplingParams(temperature=0.0), SamplingParams(max_tokens=6))
    )
    assert stage1.max_tokens == 6
    assert not stage1.stop_token_ids
    assert stage1.output_kind == RequestOutputKind.DELTA


def test_a_window_is_dated_by_the_unit_ends_before_it() -> None:
    # Window k carries the audio of [k * 480 ms, (k + 1) * 480 ms), so the
    # count of closed windows is also the index of the one being decoded.
    assert closed_window_count([]) == 0
    assert closed_window_count([DUPLEX_TOKENS.speak, 100]) == 0
    assert closed_window_count([DUPLEX_TOKENS.speak, DUPLEX_TOKENS.unit_end]) == 1
    assert closed_window_count([DUPLEX_TOKENS.unit_end, DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end]) == 2


def test_text_belongs_to_whichever_phase_opened_it() -> None:
    assert window_text_role([DUPLEX_TOKENS.asr, 100]) == "user"
    assert window_text_role([DUPLEX_TOKENS.asr, 100, DUPLEX_TOKENS.asr_eos, DUPLEX_TOKENS.speak, 101]) == "assistant"
    # A protocol token is nobody's words.
    assert window_text_role([DUPLEX_TOKENS.asr, 100, DUPLEX_TOKENS.asr_eos]) is None
    assert window_text_role([]) is None


def test_only_an_activity_window_carries_a_user_activity_verdict() -> None:
    # The activity phase forces idle/non_idle before anything else, so the
    # verdict is the window's opening token.
    assert window_user_activity([DUPLEX_TOKENS.non_idle, DUPLEX_TOKENS.speak]) == "non_idle"
    assert window_user_activity([DUPLEX_TOKENS.idle, DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end]) == "idle"
    # transcript opens on asr, none opens on the control token.
    assert window_user_activity([DUPLEX_TOKENS.asr, 100]) is None
    assert window_user_activity([DUPLEX_TOKENS.speak, 100]) is None


def test_an_earlier_window_does_not_lend_its_verdict_to_the_next() -> None:
    closed = [DUPLEX_TOKENS.non_idle, DUPLEX_TOKENS.speak, DUPLEX_TOKENS.unit_end]

    # The closing step still reads its own window, not the empty slice after it.
    assert window_user_activity(closed) == "non_idle"
    assert window_user_activity([*closed, DUPLEX_TOKENS.idle]) == "idle"
