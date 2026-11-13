# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 projects a finished window from the one token the engine received."""

from types import SimpleNamespace

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.data_plane import (
    Qwen3OmniDuplexDataPlane,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


def _result(token_ids: list[int], *, finished: bool = False, text: str = "") -> dict[str, object]:
    # The session runs on DELTA output, so the projected step carries only the
    # token it sampled; the window lives in the cumulative view.
    completion = SimpleNamespace(text=text, token_ids=token_ids[-1:], cumulative_token_ids=token_ids)
    output = SimpleNamespace(request_id="req", stage_id=0, finished=finished, outputs=[completion])
    return {"data_plane_outputs": [output]}


def test_listen_window_projects_as_listen() -> None:
    (projected,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end]))

    assert projected["is_listen"] is True
    assert projected["model_listen"] is True
    assert projected["end_of_turn"] is False


def test_turn_end_window_projects_as_end_of_turn() -> None:
    # The protocol samples turn_eos, then constrains unit_end as the stop, so
    # the turn-end window's last two tokens are that pair.
    (projected,) = Qwen3OmniDuplexDataPlane().project(
        _result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end])
    )

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is True


def test_mid_turn_speak_window_projects_as_neither() -> None:
    # unit_end closes the window but not the turn.
    (projected,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end]))

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is False


def test_a_step_inside_a_window_decides_nothing() -> None:
    # Only the step that samples unit_end closes a window.
    (projected,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.speak, 100]))

    assert projected["is_listen"] is False
    assert projected["end_of_turn"] is False


def test_earlier_windows_do_not_disturb_the_decision() -> None:
    # One resumable request holds every window, so the cumulative view grows
    # across windows; only the last two tokens decide.
    (projected,) = Qwen3OmniDuplexDataPlane().project(
        _result(
            [
                DUPLEX_TOKENS.listen,
                DUPLEX_TOKENS.unit_end,
                DUPLEX_TOKENS.speak,
                100,
                DUPLEX_TOKENS.turn_eos,
                DUPLEX_TOKENS.unit_end,
            ]
        )
    )

    assert projected["end_of_turn"] is True


def test_an_asr_token_is_not_reported_as_the_assistant_speaking() -> None:
    # ``transcript`` windows transcribe the user between asr and asr_eos. The
    # tokens detokenize like any other text, so projecting them verbatim put the
    # user's words inside the assistant's transcript, e.g. a turn reading
    # "<user question> <assistant answer>" as one assistant utterance.
    (user,) = Qwen3OmniDuplexDataPlane().project(_result([DUPLEX_TOKENS.asr, 100], text="what the user said"))
    (assistant,) = Qwen3OmniDuplexDataPlane().project(
        _result(
            [DUPLEX_TOKENS.asr, 100, DUPLEX_TOKENS.asr_eos, DUPLEX_TOKENS.speak, 101],
            text="what the model answers",
        )
    )

    assert user["text"] == ""
    assert assistant["text"] == "what the model answers"


def _transcript_window(text: str) -> list[tuple[list[int], str]]:
    """The steps of one ``transcript`` window; an empty window still pairs asr."""
    steps: list[tuple[list[int], str]] = [([DUPLEX_TOKENS.asr], "")]
    if text:
        steps.append(([100], text))
    steps.extend([([DUPLEX_TOKENS.asr_eos], ""), ([DUPLEX_TOKENS.speak], ""), ([DUPLEX_TOKENS.unit_end], "")])
    return steps


def _activity_window(*, speaking: bool) -> list[tuple[list[int], str]]:
    opener = DUPLEX_TOKENS.non_idle if speaking else DUPLEX_TOKENS.idle
    return [([opener], ""), ([DUPLEX_TOKENS.speak], ""), ([DUPLEX_TOKENS.unit_end], "")]


def _run(windows: list[list[tuple[list[int], str]]]) -> list[dict[str, object]]:
    """Feed whole windows through one data plane and collect the user events.

    One window is one append, and the session opens every append with
    ``begin_request`` on the same resumable request id. Clearing the utterance
    there restarted it every 480 ms: each window became its own item with one
    delta, and no utterance ever ended.
    """
    plane = Qwen3OmniDuplexDataPlane()
    cumulative: list[int] = []
    events: list[dict[str, object]] = []
    for window in windows:
        plane.begin_request("req")
        for tokens, text in window:
            cumulative.extend(tokens)
            (projected,) = plane.project(_result(list(cumulative), text=text))
            events.extend(projected["user_speech_events"])
    return events


def test_a_transcribing_window_opens_the_user_item_and_streams_its_words() -> None:
    events = _run([_transcript_window("hello")])

    assert events == [
        {"kind": "started", "item_id": events[0]["item_id"], "audio_start_ms": 0},
        {"kind": "delta", "item_id": events[0]["item_id"], "delta": "hello"},
    ]


def test_one_silent_window_is_jitter_and_does_not_end_the_utterance() -> None:
    # Per-window ASR lags the audio, so a single empty window inside a sentence
    # is the transcriber catching up, not the user stopping.
    events = _run([_transcript_window("one"), _transcript_window(""), _transcript_window("two")])

    assert [event["kind"] for event in events] == ["started", "delta", "delta"]
    assert len({event["item_id"] for event in events}) == 1


def test_two_silent_windows_end_the_utterance_at_its_last_word() -> None:
    events = _run([_transcript_window("hi"), _transcript_window(""), _transcript_window("")])

    assert [event["kind"] for event in events] == ["started", "delta", "stopped", "completed"]
    # The utterance ended with window 0, not with the silence that proved it.
    assert events[2]["audio_end_ms"] == 480
    assert events[3]["transcript"] == "hi"


def test_a_second_utterance_is_a_new_item_dated_after_the_first() -> None:
    events = _run(
        [
            _transcript_window("first"),
            _transcript_window(""),
            _transcript_window(""),
            _transcript_window(""),
            _transcript_window("second"),
        ]
    )

    starts = [event for event in events if event["kind"] == "started"]
    assert [event["audio_start_ms"] for event in starts] == [0, 1920]
    assert starts[0]["item_id"] != starts[1]["item_id"]
    assert [event["transcript"] for event in events if event["kind"] == "completed"] == ["first"]


def test_an_activity_window_reports_the_user_without_transcribing_them() -> None:
    events = _run(
        [
            _activity_window(speaking=True),
            _activity_window(speaking=False),
            _activity_window(speaking=False),
        ]
    )

    assert [event["kind"] for event in events] == ["started", "stopped", "completed"]
    assert events[1]["audio_end_ms"] == 480
    assert events[2]["transcript"] == ""


def test_a_checkpoint_without_a_user_signal_reports_nothing() -> None:
    # ``none`` windows open straight on the control token: no verdict to relay.
    events = _run([[([DUPLEX_TOKENS.speak], ""), ([100], "answer"), ([DUPLEX_TOKENS.unit_end], "")]] * 3)

    assert events == []


def test_a_deployment_with_audio_moves_the_turn_end_to_the_last_chunk() -> None:
    import numpy as np
    import torch

    encoded: list[int] = []
    plane = Qwen3OmniDuplexDataPlane(lambda samples, *_args: encoded.append(int(samples.size)) or "pcm")
    plane.set_audio_stage(2)
    context = DuplexDataPlaneContext(modalities=("audio",), active_response_turn_id=0, turn_id=0)

    def audio_output() -> SimpleNamespace:
        return SimpleNamespace(
            request_id="req",
            stage_id=2,
            finished=True,
            outputs=[SimpleNamespace()],
            multimodal_output={"audio": [torch.zeros(4)]},
        )

    # DELTA output: the text rides the step that sampled it, and the window
    # closes on the step that samples unit_end, which is where the turn's audio
    # chunk is accounted for. The text goes out at once; the turn is not over.
    spoken = next(iter(plane.project(_result([DUPLEX_TOKENS.speak, 100], text="hi"), context=context)))
    assert spoken["text"] == "hi"
    closed = next(
        iter(
            plane.project(
                _result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end]), context=context
            )
        )
    )
    assert closed["end_of_turn"] is False

    # The turn's audio chunk carries the end, and the samples reach the encoder.
    chunk = next(iter(plane.project({"data_plane_outputs": [audio_output()]}, context=context)))
    assert chunk["audio_data"] == "pcm"
    assert chunk["end_of_turn"] is True
    assert encoded == [4]
    assert chunk["audio_duration_ms"] == 4 * 1000 / 24000

    # A text-only session never sees the audio stage's output.
    text_only = list(Qwen3OmniDuplexDataPlane().project({"data_plane_outputs": [audio_output()]}))
    assert text_only == []
    del np


def test_the_window_is_tagged_with_the_session_model_turn() -> None:
    # The context is a dataclass, not a mapping. Reading it as a mapping tagged
    # every window with turn 0, and the session dropped each model turn after
    # the first because 0 was behind the turn it had already completed.
    window = _result([DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.unit_end])

    (inside,) = Qwen3OmniDuplexDataPlane().project(
        window, context=DuplexDataPlaneContext(turn_id=7, active_response_turn_id=5, response_format="pcm16")
    )
    (between,) = Qwen3OmniDuplexDataPlane().project(
        window, context=DuplexDataPlaneContext(turn_id=7, active_response_turn_id=None)
    )

    # A response is open, so the window belongs to the turn that response serves.
    assert inside["model_turn_id"] == 5
    assert inside["audio_format"] == "pcm16"
    # No response is open, so the window opens the session's current turn.
    assert between["model_turn_id"] == 7
