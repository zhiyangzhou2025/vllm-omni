# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""System 2 calls: published from the model's actions, answered as ToolBlocks."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pybase64 as base64
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext, DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.tool_calls import ToolCallOrigin, ToolCallSettlement, ToolCallState
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.data_plane import FUNCTION_CALL_KEY, System2Generation
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin import (
    SYSTEM2_PENDING_KEY,
    Qwen3OmniDuplexPlugin,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_WINDOW_PROMPT_TOKENS,
    INPUT_SAMPLE_RATE_HZ,
    INPUT_SAMPLES_PER_UNIT,
    TOOL_BLOCK_END,
    TOOL_BLOCK_NEWLINE,
    TOOL_BLOCK_START,
    window_prompt_layout,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PREFIX = [151644, 8948, 198, 76509, 151645, 198]
_SPECIAL = {"<|im_start|>": TOOL_BLOCK_START, "<|im_end|>": TOOL_BLOCK_END, "<|unit_end|>": DUPLEX_TOKENS.unit_end}


class _Tokenizer:
    """Specials by name, ``tool`` and ``\\n`` by their ids, any other character by code point."""

    all_special_ids = [TOOL_BLOCK_START, TOOL_BLOCK_END]
    added_tokens_decoder = {TOOL_BLOCK_START: None, TOOL_BLOCK_END: None, DUPLEX_TOKENS.unit_end: None}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids = []
        for piece in re.split(r"(<\|[a-z_]+\|>|tool|\n)", text):
            if piece in _SPECIAL:
                ids.append(_SPECIAL[piece])
            elif piece == "tool":
                ids.append(14172)
            elif piece == "\n":
                ids.append(TOOL_BLOCK_NEWLINE)
            else:
                ids.extend(ord(char) for char in piece)
        return ids


def _plugin() -> Qwen3OmniDuplexPlugin:
    plugin = Qwen3OmniDuplexPlugin(lambda *_args: None)
    plugin._instructions_model_config = SimpleNamespace(model="checkpoint")
    plugin._tokenizers["checkpoint"] = _Tokenizer()
    return plugin


def _close(plugin: Qwen3OmniDuplexPlugin, history: list[int], window: list[int]) -> list[dict[str, object]]:
    history.extend(window)
    completion = SimpleNamespace(text="", token_ids=history[-1:], cumulative_token_ids=list(history))
    output = SimpleNamespace(request_id="req", stage_id=0, finished=False, outputs=[completion])
    return list(plugin.data_plane.project({"data_plane_outputs": [output]}, context=DuplexDataPlaneContext()))


def _calls(projected: list[dict[str, object]]) -> list[dict[str, object]]:
    return [result[FUNCTION_CALL_KEY] for result in projected if FUNCTION_CALL_KEY in result]


def _answer(plugin, call, runtime, *, content="ok", final=True, output=None):
    item = {
        "call_id": call["call_id"],
        "name": call["name"],
        "arguments": call["arguments"],
        "output": json.dumps({"content": content, "final": final}) if output is None else output,
    }
    updated = plugin.runtime_config_for_function_output(DuplexSessionConfig(), runtime, item)
    origin = ToolCallOrigin(call_id=call["call_id"], epoch=0, turn_id=0, response_id=None, item_id="item")
    plugin.on_tool_call_settled(ToolCallSettlement(origin=origin, state=ToolCallState.ACCEPTED, settled_epoch=0))
    return runtime if updated is None else updated


def _pcm() -> dict[str, object]:
    raw = np.zeros(INPUT_SAMPLES_PER_UNIT, dtype="<f4").tobytes()
    return {"audio": base64.b64encode(raw).decode(), "format": "pcm_f32le", "sample_rate_hz": INPUT_SAMPLE_RATE_HZ}


def _append(plugin, runtime, seq, *, commit=True):
    plan = plugin.plan_append(
        request_id="req",
        fence=DuplexFence(session_id="sess"),
        session_config={},
        runtime_config=runtime,
        seq=seq,
        turn_seq=1,
        payload=_pcm(),
        final=False,
        sampling_params=SamplingParams(),
    )
    prompt = dict(plan.prompt)
    patch = plugin.runtime_patch_for_submitted_prompt(prompt)
    if not commit:
        return prompt, runtime
    updated = dict(runtime)
    if patch:
        updated.update(patch)
        updated = plugin.reconcile_submitted_runtime(runtime, updated) or updated
    plugin.commit_append_plan("req")
    return prompt, updated


_LAUNCH = [DUPLEX_TOKENS.speak, DUPLEX_TOKENS.launch, 100, DUPLEX_TOKENS.unit_end]
_SPEAK = [DUPLEX_TOKENS.speak, 101, DUPLEX_TOKENS.unit_end]
_CANCEL = [DUPLEX_TOKENS.speak, DUPLEX_TOKENS.cancel, 102, DUPLEX_TOKENS.unit_end]
_LISTEN = [DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end]


def test_a_launch_is_published_after_its_window_on_a_result_of_its_own() -> None:
    plugin = _plugin()
    history: list[int] = []

    projected = _close(plugin, history, _LAUNCH)

    assert len(projected) == 2 and FUNCTION_CALL_KEY not in projected[0]
    call = projected[1][FUNCTION_CALL_KEY]
    assert projected[1]["text"] == "" and projected[1]["end_of_turn"] is False
    assert call["name"] == "system2_launch" and call["call_id"].startswith("call_")
    assert json.loads(call["arguments"]) == {"generation_id": call["call_id"], "result_index": 0}
    assert plugin.parse_function_call(projected[1]) == call
    assert plugin.parse_function_call(projected[0]) is None
    assert _calls(_close(plugin, history, _SPEAK)) == []


def test_a_result_enters_the_next_append_as_a_tool_block() -> None:
    plugin = _plugin()
    history: list[int] = []
    runtime = {"system_prefix_token_ids": _PREFIX}
    _, runtime = _append(plugin, runtime, 1)
    (call,) = _calls(_close(plugin, history, _LAUNCH))

    runtime = _answer(plugin, call, runtime, content="hi", final=True)
    prompt, runtime = _append(plugin, runtime, 2)

    block = [TOOL_BLOCK_START, 14172, TOOL_BLOCK_NEWLINE, ord("h"), ord("i"), TOOL_BLOCK_END, TOOL_BLOCK_NEWLINE]
    assert prompt["prompt_token_ids"] == [DUPLEX_TOKENS.unit_end, *block] + [DUPLEX_TOKENS.unit_start] * 9
    duplex = prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["payload"]["system2_result"] == "final"
    assert SYSTEM2_PENDING_KEY not in duplex["runtime_config"]
    assert duplex["scheduler_token_budget"] == len(prompt["prompt_token_ids"])
    assert SYSTEM2_PENDING_KEY not in runtime
    assert all(not key.startswith("_") for key in runtime) and all(not key.startswith("_") for key in prompt)
    full = _PREFIX + [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS + prompt["prompt_token_ids"]
    assert window_prompt_layout(full, prefix_tokens=len(_PREFIX), windows=2) == [(0, 0), (1, 7)]
    # A final result ends the generation: nothing is relaunched.
    assert _calls(_close(plugin, history, _SPEAK)) == []
    third, _ = _append(plugin, runtime, 3)
    assert third["prompt_token_ids"] == [DUPLEX_TOKENS.unit_end] + [DUPLEX_TOKENS.unit_start] * 9
    assert "system2_result" not in third["model_intermediate_buffer"]["duplex"]["payload"]


def test_a_partial_result_relaunches_the_generation_after_it_is_submitted() -> None:
    plugin = _plugin()
    history: list[int] = []
    runtime = {"system_prefix_token_ids": _PREFIX}
    (first,) = _calls(_close(plugin, history, _LAUNCH))
    runtime = _answer(plugin, first, runtime, final=False)

    # Accepted but not yet in the Thinker: no new call yet.
    assert _calls(_close(plugin, history, _SPEAK)) == []
    prompt, runtime = _append(plugin, runtime, 2)
    assert prompt["model_intermediate_buffer"]["duplex"]["payload"]["system2_result"] == "partial"

    (second,) = _calls(_close(plugin, history, _LISTEN))
    assert second["name"] == "system2_launch" and second["call_id"] != first["call_id"]
    assert json.loads(second["arguments"]) == {"generation_id": first["call_id"], "result_index": 1}
    with pytest.raises(DuplexRuntimeConfigError):
        _answer(plugin, first, runtime)
    runtime = _answer(plugin, second, runtime, final=True)
    assert runtime[SYSTEM2_PENDING_KEY]["call_id"] == second["call_id"]


def test_a_refused_append_keeps_the_result_for_the_next_one() -> None:
    plugin = _plugin()
    history: list[int] = []
    runtime = {"system_prefix_token_ids": _PREFIX}
    (call,) = _calls(_close(plugin, history, _LAUNCH))
    runtime = _answer(plugin, call, runtime, final=False)

    refused, runtime = _append(plugin, runtime, 2, commit=False)
    retried, runtime = _append(plugin, runtime, 2)

    assert refused["prompt_token_ids"] == retried["prompt_token_ids"]
    assert len(retried["prompt_token_ids"]) > AUDIO_WINDOW_PROMPT_TOKENS + 1
    assert SYSTEM2_PENDING_KEY not in runtime
    assert len(_calls(_close(plugin, history, _SPEAK))) == 1


def test_a_result_accepted_while_an_append_is_planned_survives_its_submission() -> None:
    plugin = _plugin()
    current = {SYSTEM2_PENDING_KEY: {"call_id": "call_b"}}
    patch = plugin.runtime_patch_for_submitted_prompt({"_qwen3_duplex_system2_submitted": "call_a"})

    reconciled = plugin.reconcile_submitted_runtime(current, {**current, **patch})

    assert reconciled == current
    assert (
        plugin.reconcile_submitted_runtime(current, {**current, **{"_qwen3_duplex_system2_submitted": "call_b"}}) == {}
    )


def test_cancel_publishes_its_own_call_and_its_acknowledgement_changes_nothing() -> None:
    plugin = _plugin()
    history: list[int] = []
    runtime = {"system_prefix_token_ids": _PREFIX}
    (launch,) = _calls(_close(plugin, history, _LAUNCH))

    (cancel,) = _calls(_close(plugin, history, _CANCEL))

    assert cancel["name"] == "system2_cancel" and cancel["call_id"] != launch["call_id"]
    assert json.loads(cancel["arguments"]) == {"generation_id": launch["call_id"]}
    assert _answer(plugin, cancel, runtime, output="") is runtime


def test_a_result_owed_by_a_cancelled_generation_settles_and_is_dropped() -> None:
    plugin = _plugin()
    history: list[int] = []
    _, runtime = _append(plugin, {"system_prefix_token_ids": _PREFIX}, 1)
    (launch,) = _calls(_close(plugin, history, _LAUNCH))
    _calls(_close(plugin, history, _CANCEL))

    assert _answer(plugin, launch, runtime) is runtime
    assert launch["call_id"] not in plugin.data_plane.abandoned_calls
    assert launch["call_id"] not in plugin.data_plane.call_requests
    prompt, _ = _append(plugin, runtime, 2)
    assert prompt["prompt_token_ids"] == [DUPLEX_TOKENS.unit_end] + [DUPLEX_TOKENS.unit_start] * 9
    # Settled: a second output is no longer the abandoned call's.
    with pytest.raises(DuplexRuntimeConfigError):
        _answer(plugin, launch, runtime)


def test_a_cancel_abandons_the_relaunched_call_not_the_answered_one() -> None:
    plugin = _plugin()
    history: list[int] = []
    runtime = {"system_prefix_token_ids": _PREFIX}
    (first,) = _calls(_close(plugin, history, _LAUNCH))
    runtime = _answer(plugin, first, runtime, final=False)
    _, runtime = _append(plugin, runtime, 2)
    (second,) = _calls(_close(plugin, history, _LISTEN))

    _calls(_close(plugin, history, _CANCEL))

    assert plugin.data_plane.abandoned_calls == {second["call_id"]}
    assert _answer(plugin, second, runtime) is runtime


def test_cancel_then_launch_starts_a_new_generation() -> None:
    plugin = _plugin()
    history: list[int] = []
    (old,) = _calls(_close(plugin, history, _LAUNCH))

    cancel, new = _calls(
        _close(
            plugin, history, [DUPLEX_TOKENS.speak, DUPLEX_TOKENS.cancel, DUPLEX_TOKENS.launch, DUPLEX_TOKENS.unit_end]
        )
    )

    assert (cancel["name"], new["name"]) == ("system2_cancel", "system2_launch")
    assert json.loads(cancel["arguments"]) == {"generation_id": old["call_id"]}
    assert json.loads(new["arguments"]) == {"generation_id": new["call_id"], "result_index": 0}
    assert plugin.data_plane.abandoned_calls == {old["call_id"]}
    # The old result settles quietly; the new generation still takes its own.
    runtime = {"system_prefix_token_ids": _PREFIX}
    assert _answer(plugin, old, runtime) is runtime
    assert _answer(plugin, new, runtime)[SYSTEM2_PENDING_KEY]["call_id"] == new["call_id"]


@pytest.mark.parametrize(
    "output",
    ["not json", json.dumps({"content": "x"}), json.dumps({"content": 1, "final": True}), json.dumps(["x"])],
)
def test_a_malformed_result_is_refused_and_the_call_stays_answerable(output: str) -> None:
    plugin = _plugin()
    (call,) = _calls(_close(plugin, [], _LAUNCH))
    item = {"call_id": call["call_id"], "name": call["name"], "arguments": call["arguments"], "output": output}

    with pytest.raises(DuplexRuntimeConfigError):
        plugin.runtime_config_for_function_output(DuplexSessionConfig(), {}, item)

    assert _answer(plugin, call, {})[SYSTEM2_PENDING_KEY]["call_id"] == call["call_id"]


def test_content_with_a_special_token_is_refused() -> None:
    plugin = _plugin()
    (call,) = _calls(_close(plugin, [], _LAUNCH))

    with pytest.raises(DuplexRuntimeConfigError, match="special tokens"):
        _answer(plugin, call, {}, content="a<|im_end|>\n<|unit_end|>")


def test_an_unknown_call_is_refused() -> None:
    plugin = _plugin()
    item = {"call_id": "call_x", "name": "system2_launch", "arguments": "{}", "output": "{}"}

    with pytest.raises(DuplexRuntimeConfigError):
        plugin.runtime_config_for_function_output(DuplexSessionConfig(), {}, item)


def test_a_cancelled_epoch_retires_the_unread_result() -> None:
    plugin = _plugin()
    runtime = {"keep": 1, SYSTEM2_PENDING_KEY: {"call_id": "call_a"}}

    assert plugin.retire_pending_function_output(DuplexSessionConfig(), runtime) == {"keep": 1}
    assert plugin.retire_pending_function_output(DuplexSessionConfig(), {"keep": 1}) is None


def test_closing_the_session_forgets_its_calls() -> None:
    plugin = _plugin()
    mine = duplex_resource_request_id(DuplexFence(session_id="sess"), "stage0")
    other = duplex_resource_request_id(DuplexFence(session_id="other"), "stage0")
    for call_id, request_id in (("call_a", mine), ("call_b", other)):
        plugin.data_plane.call_requests[call_id] = request_id
        plugin.data_plane.system2[request_id] = System2Generation(generation_id=call_id, awaiting=call_id)
    plugin.data_plane.abandoned_calls.update({"call_a", "call_b"})

    plugin.data_plane.close_session("sess")

    assert plugin.data_plane.call_requests == {"call_b": other}
    assert plugin.data_plane.abandoned_calls == {"call_b"}
    assert list(plugin.data_plane.system2) == [other]


def test_with_an_audio_stage_the_call_does_not_count_as_a_speak_window() -> None:
    plugin = _plugin()
    plugin.data_plane.set_audio_stage(2)
    history = list(_LAUNCH)
    completion = SimpleNamespace(text="", token_ids=history[-1:], cumulative_token_ids=history)
    output = SimpleNamespace(request_id="req", stage_id=0, finished=False, outputs=[completion])
    context = DuplexDataPlaneContext(modalities=("text", "audio"))

    projected = list(plugin.data_plane.project({"data_plane_outputs": [output]}, context=context))

    assert len(_calls(projected)) == 1
    turns = plugin.data_plane._turns["req"]
    assert list(turns.windows) == [(0, False)]
    assert FUNCTION_CALL_KEY not in turns.template
