#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drive the three-stage Qwen3-Omni-Duplex service one audio window at a time.

Thinker (Stage 0) decides listen/speak per 0.48-second window and streams the
text. Talker (Stage 1) and Code2Wav (Stage 2) turn every speak window into
audio, which arrives as ``response.output_audio.delta``. This script appends
the windows of a WAV like a live microphone and prints, per window slot, what
the session reported back: the user-side events, listen/turn decisions, text,
and how many milliseconds of audio arrived. After the drain it prints a
timeline with per-turn audio length and first-audio latency, and writes each
turn's audio and the whole session's audio as WAV files.

The protocol carries no window index, so a window's output is taken to be the
events that arrived between its append and the next one. Audio trails its text
by the Talker and Code2Wav latency, so a turn's audio can spill into the next
slots or the drain.

Start the server first, then run this against it::

    vllm serve <v3-checkpoint> --omni \
        --deploy-config vllm_omni/deploy/qwen3_omni_duplex.yaml \
        --trust-remote-code --served-model-name qwen3-omni-duplex \
        --host 127.0.0.1 --port 8093

    python examples/online_serving/qwen3_omni_duplex/duplex_window_client.py \
        --wav tests/assets/minicpmo_4_5/soft_interrupt_16k.wav --out-dir /tmp/duplex_out

A scripted ``session.update`` exercises the modalities and instructions rules.
For example, turn speech off from window 6 on::

    ... --update-at 6 --update-json '{"modalities": ["text"]}'
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
import pybase64 as base64

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

OUTPUT_SAMPLE_RATE_HZ = 24_000

# Types the per-window line already reports, plus the bookkeeping the session
# emits around every response. Anything else is surfaced so a new event type
# cannot pass unnoticed.
_SUMMARIZED_TYPES = frozenset(
    {
        "response.listen",
        "response.created",
        "response.done",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
        "response.content_part.added",
        "response.content_part.done",
        "error",
        "session.updated",
        "conversation.item.added",
        "conversation.item.created",
        "conversation.item.done",
        "response.output_item.added",
        "response.output_item.done",
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


def audio_samples(event: dict) -> np.ndarray:
    """The int16 samples one ``response.output_audio.delta`` carries."""

    fmt = str(event.get("format") or "pcm16")
    if fmt != "pcm16":
        raise SystemExit(f"output audio arrived as {fmt!r}; this client asks for and decodes pcm16 only")
    rate = event.get("sample_rate_hz")
    if rate not in (None, OUTPUT_SAMPLE_RATE_HZ):
        raise SystemExit(f"output audio arrived at {rate} Hz, expected {OUTPUT_SAMPLE_RATE_HZ}")
    return np.frombuffer(base64.b64decode(str(event.get("delta") or "")), dtype="<i2")


def audio_ms(events: list[dict]) -> float:
    samples = sum(len(audio_samples(event)) for event in events if event.get("type") == "response.output_audio.delta")
    return samples * 1000 / OUTPUT_SAMPLE_RATE_HZ


def describe_slot(events: list[dict]) -> str:
    """One line for everything the session reported inside a window's slot."""

    notes: list[str] = []
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
    deltas = sum(1 for event in events if event.get("type") == "response.output_audio.delta")
    if deltas:
        notes.append(f"audio {audio_ms(events):.0f}ms/{deltas}")
    if any(event.get("type") == "response.output_audio.done" for event in events):
        notes.append("audio-done")
    for event in events:
        if event.get("type") == "response.done":
            response = event.get("response") or {}
            reason = (response.get("status_details") or {}).get("reason") or response.get("status")
            notes.append(f"turn-done({reason})")
        if event.get("type") == "session.updated":
            session = event.get("session") or {}
            notes.append(
                f"session.updated(modalities={session.get('modalities')}, "
                f"instructions={session.get('instructions')!r})"
            )
        if event.get("type") == "error":
            notes.append(f"ERROR {json.dumps(event, ensure_ascii=False)[:200]}")
    extra = sorted({str(event.get("type")) for event in events} - _SUMMARIZED_TYPES)
    if extra:
        notes.append("+" + ",".join(extra))
    return "  ".join(notes) if notes else "-"


def conversation_timeline(events: list[dict], arrivals: list[float]) -> list[str]:
    """One line per user utterance and per model turn, ordered by when it opened.

    The user's milliseconds are audio time the model reported; a turn is placed
    by when its events reached the client. For a turn, ``first-audio`` is the
    wait from ``response.created`` to its first audio delta.
    """

    entries: dict[tuple[str, str], dict] = {}

    def entry(side: str, key: str, at: float) -> dict:
        return entries.setdefault(
            (side, key),
            {"opened": at, "closed": at, "span": "", "deltas": [], "text": None, "audio": [], "first_audio": None},
        )

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
        elif kind == "response.output_audio.delta":
            turn = entry("turn", response_id, at)
            turn["audio"].append(event)
            if turn["first_audio"] is None:
                turn["first_audio"] = at
            turn["closed"] = max(turn["closed"], at)
        elif kind == "response.done":
            turn = entry("turn", response_id, at)
            turn["closed"] = max(turn["closed"], at)

    lines: list[str] = []
    for (side, key), record in sorted(entries.items(), key=lambda pair: pair[1]["opened"]):
        when = f"{record['opened']:6.2f}s →{record['closed']:6.2f}s"
        if side != "user":
            seconds = audio_ms(record["audio"]) / 1000
            first = "-" if record["first_audio"] is None else f"{record['first_audio'] - record['opened']:.2f}s"
            sound = f"audio {seconds:5.2f}s/{len(record['audio']):<3d} first-audio {first:<6}"
            lines.append(f"{when}  turn {key[-8:]:<14} {sound} {(record['text'] or '')!r}")
            continue
        deltas = "".join(record["deltas"])
        text = deltas if record["text"] is None else record["text"]
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


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(OUTPUT_SAMPLE_RATE_HZ)
        handle.writeframes(samples.astype("<i2").tobytes())


def write_audio(out_dir: Path, events: list[dict]) -> list[str]:
    """One WAV per turn in the order turns opened, plus the session's audio back to back."""

    out_dir.mkdir(parents=True, exist_ok=True)
    turns: dict[str, list[np.ndarray]] = {}
    for event in events:
        if event.get("type") == "response.created":
            turns.setdefault(str((event.get("response") or {}).get("id") or event.get("response_id") or ""), [])
        elif event.get("type") == "response.output_audio.delta":
            turns.setdefault(str(event.get("response_id") or ""), []).append(audio_samples(event))
    written: list[str] = []
    everything: list[np.ndarray] = []
    for index, chunks in enumerate(chunks for chunks in turns.values() if chunks):
        samples = np.concatenate(chunks)
        everything.append(samples)
        path = out_dir / f"turn{index}.wav"
        write_wav(path, samples)
        written.append(f"{path} {len(samples) / OUTPUT_SAMPLE_RATE_HZ:.2f}s")
    if everything:
        path = out_dir / "session.wav"
        write_wav(path, np.concatenate(everything))
        written.append(str(path))
    return written


async def run(args: argparse.Namespace) -> int:
    windows = load_windows(Path(args.wav))
    update = json.loads(args.update_json) if args.update_json else None
    if (update is None) != (args.update_at is None):
        raise SystemExit("--update-at and --update-json go together")
    print(
        f"wav={args.wav}\n"
        f"windows={len(windows)} samples_each={INPUT_SAMPLES_PER_UNIT} "
        f"rate={INPUT_SAMPLE_RATE_HZ} unit={UNIT_DURATION_MS}ms "
        f"audio={len(windows) * UNIT_DURATION_MS / 1000:.2f}s pace={args.pace}s modalities={args.modalities}",
        flush=True,
    )

    config = SessionConfig(
        modalities=tuple(args.modalities),
        input_audio=AudioFormat("pcm_f32le", INPUT_SAMPLE_RATE_HZ),
        output_audio=AudioFormat("pcm16", OUTPUT_SAMPLE_RATE_HZ),
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
                    raw = dict(event.raw)
                    if raw.get("type") == "response.output_audio.delta":
                        raw["delta"] = f"<{len(str(raw.get('delta') or ''))} b64 chars>"
                    print(f"    {time.monotonic() - started:7.2f}s {json.dumps(raw, ensure_ascii=False)}")

        collector = asyncio.create_task(collect())
        for index, window in enumerate(windows):
            mark = len(events)
            await client.append_audio(window.astype("<f4").tobytes())
            if index == args.update_at:
                # The reply lands in this window's slot as session.updated or error.
                await client.send({"type": "session.update", "session": update})
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
    print(f"elapsed={time.monotonic() - started:.2f}s events={len(events)} audio={audio_ms(events) / 1000:.2f}s")
    print(json.dumps(dict(sorted(Counter(str(event.get("type")) for event in events).items())), indent=2))
    print("\n--- timeline: opened → closed, seconds since connect ---")
    for line in conversation_timeline(events, arrivals):
        print(line)
    violations = user_speech_order_violations(events)
    print(f"user_speech_order={'ok' if not violations else violations}")
    if errors:
        print(f"errors={json.dumps(errors, ensure_ascii=False)[:2000]}")
    if args.out_dir:
        for line in write_audio(Path(args.out_dir), events):
            print(f"wav {line}")
    if args.jsonl:
        Path(args.jsonl).write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8"
        )
        print(f"events_log={args.jsonl}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8093/v1/realtime?duplex=1")
    parser.add_argument("--model", default="qwen3-omni-duplex")
    parser.add_argument("--wav", default=str(REPO_ROOT / "tests/assets/minicpmo_4_5/soft_interrupt_16k.wav"))
    parser.add_argument(
        "--pace",
        type=float,
        default=UNIT_DURATION_MS / 1000,
        help="seconds to wait after each append; 0 sends every window at once",
    )
    parser.add_argument("--drain", type=float, default=30.0, help="seconds to keep reading after the last append")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["text", "audio"],
        choices=["text", "audio"],
        help="output modalities at session open; text alone keeps the Talker and Code2Wav idle",
    )
    parser.add_argument("--instructions", default=None, help="session system prompt; unset uses the server default")
    parser.add_argument("--update-at", type=int, default=None, help="window index after whose append to send --update-json")
    parser.add_argument("--update-json", default=None, help='session.update body, e.g. \'{"modalities": ["text"]}\'')
    parser.add_argument("--out-dir", default=None, help="write turn<N>.wav per turn and session.wav here")
    parser.add_argument("--jsonl", default=None, help="write every received event to this file")
    parser.add_argument("--verbose", action="store_true", help="print every event as it arrives")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
