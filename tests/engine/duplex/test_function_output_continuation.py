# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""An accepted ``function_call_output`` resumes the response only when it must.

A turn-taking model ends its answer to wait for the result, so the session
continues the response. A model that streams units of its own carries the
result in its next unit, and a continuation would submit a unit the client
never sent.
"""

import pytest

from tests.engine.duplex.test_session_runner import append_audio, close_harness, open_harness
from tests.engine.duplex.test_session_runner_nemotron_voicechat import error_codes, output_item
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin
from vllm_omni.engine.duplex.session.tool_calls import ToolCallState
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CALL_ID = "call_continue"


async def _accept_one_result(h, monkeypatch, *, continues: bool) -> list[dict[str, object]]:
    monkeypatch.setattr(h.runner.plugin, "function_output_continues_response", lambda: continues)
    resumed: list[dict[str, object]] = []

    async def record(**kwargs: object) -> None:
        resumed.append(kwargs)

    monkeypatch.setattr(h.runner.model, "maybe_continue_response", record)
    assert error_codes(await h.run(append_audio())) == []
    await h.runner.model._expose_function_call({"call_id": CALL_ID, "name": "lookup", "arguments": "{}"})
    await h.settle()
    events = await h.run(commands.CreateItem(item=output_item(CALL_ID)))
    assert error_codes(events) == []
    assert h.runner.tool_calls.record(CALL_ID).state is ToolCallState.ACCEPTED
    return resumed


def test_a_turn_taking_plugin_keeps_the_default() -> None:
    assert DuplexModelPlugin.function_output_continues_response(object()) is True
    assert Qwen3OmniDuplexPlugin.function_output_continues_response(object()) is False


@pytest.mark.asyncio
async def test_a_turn_taking_plugin_resumes_the_response_after_a_result(monkeypatch) -> None:
    h = await open_harness()
    try:
        resumed = await _accept_one_result(h, monkeypatch, continues=True)

        assert len(resumed) == 1 and resumed[0]["expected_epoch"] == h.session.epoch
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_streaming_plugin_answers_a_result_without_a_continuation(monkeypatch) -> None:
    h = await open_harness()
    try:
        assert await _accept_one_result(h, monkeypatch, continues=False) == []
    finally:
        await close_harness(h)
