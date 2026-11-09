#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drive Qwen3-Omni-Duplex Stage 0 one exact audio window at a time.

Every append is one 0.48-second window, paced like a live microphone. The
script prints what went into each window and what the session reported back
inside that window's slot, then one line per user utterance and per model turn
in the order they opened.

The protocol carries no window index, so a window's output is taken to be the
events that arrived between its append and the next one. Stage 0 keeps up with
realtime pacing, so the slots line up; at ``--pace 0`` the appends outrun the
model and every event lands in the drain bucket instead.

Start the server first, then run this against it::

    vllm serve <v3-checkpoint> --omni \
        --deploy-config vllm_omni/deploy/qwen3_omni_duplex_stage0.yaml \
        --trust-remote-code --served-model-name qwen3-omni-duplex \
        --host 127.0.0.1 --port 8091

    python examples/online_serving/qwen3_omni_duplex/stage0_window_client.py \
        --wav tests/assets/minicpmo_4_5/soft_interrupt_16k.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.clients.duplex import AudioFormat, DuplexClient, SessionConfig  # noqa: E402
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.plugin import (  # noqa: E402
    INPUT_SAMPLE_RATE_HZ,
    UNIT_DURATION_MS,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (  # noqa: E402
    INPUT_SAMPLES_PER_UNIT,
)

# Types the per-window line already reports, plus the bookkeeping the session
# emits around every response. Anything else is surfaced so a new event type
# cannot pass unnoticed.
_SUMMARIZED_TYPES = frozenset(
    {
        "response.listen",
        "response.created",
        "response.done",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
        "error",
        "conversation.item.added",
        "conversation.item.created",
        "conversation.item.done",
        "response.output_item.added",
        "response.output_item.done",
        "response.output_audio.done",
        "response.speak",
        "rate_limits.updated",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
    }
)


def load_windows(path: Path) -> np.ndarray:
    """Slice a WAV into exact 0.48-second windows, zero-padding the tail."""

    with wave.open(str(path)) as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
            raise SystemExit(f"{path}: need a mono 16-bit uncompressed PCM WAV")
        if handle.getframerate() != INPUT_SAMPLE_RATE_HZ:
            raise SystemExit(
                f"{path}: {handle.getframerate()} Hz, but the window protocol is fixed at "
                f"{INPUT_SAMPLE_RATE_HZ} Hz. Convert first:\n"
                f"  ffmpeg -i {path} -ac 1 -ar {INPUT_SAMPLE_RATE_HZ} -sample_fmt s16 converted.wav"
            )
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    pad = (-len(samples)) % INPUT_SAMPLES_PER_UNIT
    if pad:
        samples = np.concatenate([samples, np.zeros(pad, dtype=np.float32)])
    return samples.reshape(-1, INPUT_SAMPLES_PER_UNIT)


def describe_slot(events: list[dict]) -> str:
    """One line for everything the session reported inside a window's slot."""

    notes: list[str] = []
    # What the model reported about the *user* inside this window.
    for event in events:
        kind = event.get("type")
        if kind == "input_audio_buffer.speech_started":
            notes.append(f"user-start@{event.get('audio_start_ms')}ms")
        elif kind == "conversation.item.input_audio_transcription.delta":
            notes.append(f'user "{event.get("delta")}"')
        elif kind == "input_audio_buffer.speech_stopped":
            notes.append(f"user-stop@{event.get('audio_end_ms')}ms")
        elif kind == "conversation.item.input_audio_transcription.completed":
            notes.append(f'user-done "{event.get("transcript")}"')
    listens = [event for event in events if event.get("type") == "response.listen"]
    if listens:
        reasons = sorted({str((event.get("response") or {}).get("metadata", {}).get("reason")) for event in listens})
        notes.append(f"listen({','.join(reasons)})" + (f" x{len(listens)}" if len(listens) > 1 else ""))
    if any(event.get("type") == "response.created" for event in events):
        notes.append("turn-start")
    text = "".join(
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "response.output_audio_transcript.delta"
    )
    if text:
        notes.append(f'speak "{text}"')
    for event in events:
        if event.get("type") == "response.done":
            response = event.get("response") or {}
            reason = (response.get("status_details") or {}).get("reason") or response.get("status")
            notes.append(f"turn-done({reason})")
        if event.get("type") == "error":
            notes.append(f"ERROR {json.dumps(event, ensure_ascii=False)[:200]}")
    extra = sorted({str(event.get("type")) for event in events} - _SUMMARIZED_TYPES)
    if extra:
        notes.append("+" + ",".join(extra))
    return "  ".join(notes) if notes else "-"


def conversation_timeline(events: list[dict], arrivals: list[float]) -> list[str]:
    """One line per user utterance and per model turn, ordered by when it opened.

    The two sides share no clock. The user's milliseconds are audio time the
    model reported; a response is only placed by when its events reached the
    client. Arrival orders both, and printing both ends of each one shows where
    the user and the model were speaking at the same time.
    """

    entries: dict[tuple[str, str], dict] = {}

    def entry(side: str, key: str, at: float) -> dict:
        return entries.setdefault((side, key), {"opened": at, "closed": at, "span": "", "deltas": [], "text": None})

    for index, event in enumerate(events):
        kind = event.get("type")
        at = arrivals[index]
        item_id = str(event.get("item_id") or "")
        response_id = str(event.get("response_id") or "")
        if kind == "input_audio_buffer.speech_started":
            entry("user", item_id, at)["span"] = f"[{event.get('audio_start_ms')},"
        elif kind == "conversation.item.input_audio_transcription.delta":
            entry("user", item_id, at)["deltas"].append(str(event.get("delta") or ""))
        elif kind == "input_audio_buffer.speech_stopped":
            stopped = entry("user", item_id, at)
            stopped["span"] += f"{event.get('audio_end_ms')}]ms"
            stopped["closed"] = at
        elif kind == "conversation.item.input_audio_transcription.completed":
            transcribed = entry("user", item_id, at)
            transcribed["text"] = str(event.get("transcript") or "")
            transcribed["closed"] = at
        elif kind == "response.created":
            entry("turn", response_id, at)
        elif kind == "response.output_audio_transcript.done":
            entry("turn", response_id, at)["text"] = str(event.get("transcript") or "")
        elif kind == "response.done":
            entry("turn", response_id, at)["closed"] = at

    lines: list[str] = []
    for (side, key), record in sorted(entries.items(), key=lambda pair: pair[1]["opened"]):
        when = f"{record['opened']:6.2f}s →{record['closed']:6.2f}s"
        if side != "user":
            lines.append(f"{when}  turn {key[-8:]:<14} {'':<16} {(record['text'] or '')!r}")
            continue
        deltas = "".join(record["deltas"])
        text = deltas if record["text"] is None else record["text"]
        # The deltas a client streamed must add up to the transcript it is then
        # told the utterance was.
        mismatch = "" if record["text"] in (None, deltas) else f"  (deltas={deltas!r})"
        lines.append(f"{when}  user {key[:14]:<14} {record['span']:<16} {text!r}{mismatch}")
    return lines


def user_speech_order_violations(events: list[dict]) -> list[str]:
    """Check the contract the four user-side events owe each other."""

    violations: list[str] = []
    seen_start: set[str] = set()
    seen_stop: set[str] = set()
    last_start_ms = -1
    for event in events:
        kind = event.get("type")
        item_id = str(event.get("item_id") or "")
        if kind == "input_audio_buffer.speech_started":
            seen_start.add(item_id)
            start_ms = int(event.get("audio_start_ms") or 0)
            if start_ms < last_start_ms:
                violations.append(f"{item_id}: started at {start_ms}ms, behind {last_start_ms}ms")
            last_start_ms = start_ms
        elif kind == "conversation.item.input_audio_transcription.delta" and item_id not in seen_start:
            violations.append(f"{item_id}: delta before started")
        elif kind == "input_audio_buffer.speech_stopped":
            seen_stop.add(item_id)
        elif kind == "conversation.item.input_audio_transcription.completed" and item_id not in seen_stop:
            violations.append(f"{item_id}: completed before stopped")
    return violations


async def run(args: argparse.Namespace) -> int:
    windows = load_windows(Path(args.wav))
    print(
        f"wav={args.wav}\n"
        f"windows={len(windows)} samples_each={INPUT_SAMPLES_PER_UNIT} "
        f"rate={INPUT_SAMPLE_RATE_HZ} unit={UNIT_DURATION_MS}ms "
        f"audio={len(windows) * UNIT_DURATION_MS / 1000:.2f}s pace={args.pace}s",
        flush=True,
    )

    config = SessionConfig(
        modalities=("text",),
        input_audio=AudioFormat("pcm_f32le", INPUT_SAMPLE_RATE_HZ),
        output_audio=AudioFormat("pcm16", 24_000),
        auto_response=True,
        instructions=args.instructions,
    )
    events: list[dict] = []
    # When each event reached the client. The only clock the user side and the
    # model side have in common, so it is what orders the timeline.
    arrivals: list[float] = []
    started = time.monotonic()

    async with DuplexClient(
        args.url, model=args.model, config=config, reconnect=None, heartbeat_interval_s=None
    ) as client:
        print(f"session={client.session_id}", flush=True)

        async def collect() -> None:
            async for event in client.events():
                arrivals.append(time.monotonic() - started)
                events.append(event.raw)
                if args.verbose:
                    print(f"    {time.monotonic() - started:7.2f}s {json.dumps(event.raw, ensure_ascii=False)}")

        collector = asyncio.create_task(collect())
        for index, window in enumerate(windows):
            mark = len(events)
            await client.append_audio(window.astype("<f4").tobytes())
            if args.pace > 0:
                await asyncio.sleep(args.pace)
            print(
                f"win {index:3d}  +{time.monotonic() - started:6.2f}s  "
                f"in rms={float(np.sqrt(np.mean(window**2))):.4f} peak={float(np.abs(window).max()):.4f}  "
                f"out {describe_slot(events[mark:])}",
                flush=True,
            )

        mark = len(events)
        print(f"appends done at {time.monotonic() - started:.2f}s, draining {args.drain}s...", flush=True)
        await asyncio.sleep(args.drain)
        print(f"drain  +{time.monotonic() - started:6.2f}s  out {describe_slot(events[mark:])}", flush=True)

        collector.cancel()
        try:
            await collector
        except asyncio.CancelledError:
            pass

    errors = [event for event in events if event.get("type") == "error"]
    print("\n--- summary ---")
    print(f"elapsed={time.monotonic() - started:.2f}s events={len(events)}")
    print(json.dumps(dict(sorted(Counter(str(event.get("type")) for event in events).items())), indent=2))
    print("\n--- timeline: opened → closed, seconds since connect ---")
    for line in conversation_timeline(events, arrivals):
        print(line)
    violations = user_speech_order_violations(events)
    print(f"user_speech_order={'ok' if not violations else violations}")
    if errors:
        print(f"errors={json.dumps(errors, ensure_ascii=False)[:2000]}")
    if args.jsonl:
        Path(args.jsonl).write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8"
        )
        print(f"events_log={args.jsonl}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8091/v1/realtime?duplex=1")
    parser.add_argument("--model", default="qwen3-omni-duplex")
    parser.add_argument("--wav", default=str(REPO_ROOT / "tests/assets/minicpmo_4_5/soft_interrupt_16k.wav"))
    parser.add_argument(
        "--pace",
        type=float,
        default=UNIT_DURATION_MS / 1000,
        help="seconds to wait after each append; 0 sends every window at once",
    )
    parser.add_argument("--drain", type=float, default=30.0, help="seconds to keep reading after the last append")
    parser.add_argument("--instructions", default=None, help="session system prompt; unset uses the server default")
    parser.add_argument("--jsonl", default=None, help="write every received event to this file")
    parser.add_argument("--verbose", action="store_true", help="print every event as it arrives")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
