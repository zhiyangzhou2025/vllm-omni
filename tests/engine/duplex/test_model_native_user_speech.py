# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A native-duplex model hears the user while it answers; the server relays that."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    note_input_append,
    project_internal_event,
)
from vllm_omni.engine.duplex.session.model_channel import ModelChannel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _channel() -> tuple[ModelChannel, list[dict[str, object]], list[dict[str, object]]]:
    """A channel whose emitter and listen branch only record what reaches them."""

    async def _unused(*args, **kwargs):
        del args, kwargs

    emitted: list[dict[str, object]] = []
    listened: list[dict[str, object]] = []
    session = SimpleNamespace(
        session_id="s",
        epoch=0,
        turn_id=1,
        history=[],
        active_request_id=None,
        is_draining_request=lambda request_id: False,
    )
    channel = ModelChannel(
        SimpleNamespace(
            session=session,
            plugin=SimpleNamespace(data_plane=SimpleNamespace(is_terminal=lambda request_id: False)),
            run=SimpleNamespace(closing=False),
        ),
        SimpleNamespace(emit=emitted.append, auto_responds=lambda: False),
        close_from_runtime=_unused,
        schedule_silence_continuation=_unused,
        abort_request=_unused,
    )

    async def _on_model_listen(model_result, **kwargs):
        del kwargs
        listened.append(model_result)
        return None, False

    channel._on_model_listen = _on_model_listen  # type: ignore[method-assign]
    return channel, emitted, listened


async def _send(model_result: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    channel, emitted, listened = _channel()
    await channel._send_one_model_output_event(model_result)
    return emitted, listened


@pytest.mark.asyncio
async def test_each_verdict_about_the_user_becomes_one_standard_fact() -> None:
    emitted, _ = await _send(
        {
            "user_speech_events": [
                {"kind": "started", "item_id": "item_u", "audio_start_ms": 480},
                {"kind": "delta", "item_id": "item_u", "delta": "你好"},
                {"kind": "stopped", "item_id": "item_u", "audio_end_ms": 1440},
                {"kind": "completed", "item_id": "item_u", "transcript": "你好"},
            ]
        }
    )

    assert [event["type"] for event in emitted] == [
        "input_audio_buffer.speech_started",
        "input.transcript_delta",
        "input_audio_buffer.speech_stopped",
        "input.committed",
    ]
    assert emitted[0]["audio_start_ms"] == 480
    assert emitted[1]["delta"] == "你好"
    assert emitted[2]["audio_end_ms"] == 1440
    assert emitted[3]["transcript"] == "你好"
    assert {event.get("item_id", event.get("realtime_item_id")) for event in emitted} == {"item_u"}


@pytest.mark.asyncio
async def test_a_listen_window_still_reports_the_user_it_transcribed() -> None:
    # The listen branch returns, so relaying the user has to come first.
    emitted, listened = await _send(
        {
            "is_listen": True,
            "user_speech_events": [{"kind": "delta", "item_id": "item_u", "delta": "你好"}],
        }
    )

    assert [event["type"] for event in emitted] == ["input.transcript_delta"]
    assert len(listened) == 1


@pytest.mark.asyncio
async def test_a_window_with_no_verdict_about_the_user_says_nothing() -> None:
    emitted, _ = await _send({"user_speech_events": []})

    assert emitted == []


def test_the_delta_projects_to_the_openai_streaming_transcript_event() -> None:
    (event,) = project_internal_event(
        RealtimeProjectionState(session_id="s"),
        {"type": "input.transcript_delta", "realtime_item_id": "item_u", "delta": "你好"},
    )

    assert event.wire_type == "conversation.item.input_audio_transcription.delta"
    assert event.item_id == "item_u"
    assert event.delta == "你好"


def test_an_overlapping_utterance_is_chained_where_it_began() -> None:
    # The user spoke over the assistant, so their item is only committed once
    # the assistant's item exists. Chaining it there would claim they waited.
    state = RealtimeProjectionState(session_id="s")
    project_internal_event(state, {"type": "input_audio_buffer.speech_started", "item_id": "item_u"})
    state.last_conversation_item_id = "item_assistant"
    events = project_internal_event(
        state,
        {"type": "input.committed", "realtime_item_id": "item_u", "is_speech": True, "message": {"role": "user"}},
    )

    chained = {event.wire_type: event.previous_item_id for event in events}
    assert chained["conversation.item.added"] is None
    assert chained["conversation.item.created"] is None
    assert chained["conversation.item.done"] is None


def test_the_model_is_the_only_judge_of_when_the_user_started() -> None:
    # Without this, the first append latches ``input_speech_started`` and mints
    # an item id, so the model's own verdict is swallowed as a duplicate.
    state = RealtimeProjectionState(session_id="s")
    events = note_input_append(state, {"audio": "AAAA"}, model_native_input_speech=True)

    assert [event.wire_type for event in events] == []
    assert state.input_speech_started is False
