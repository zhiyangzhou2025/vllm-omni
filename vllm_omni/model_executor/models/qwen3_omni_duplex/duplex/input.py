# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PCM buffer that emits one exact 0.48-second window per append."""

from __future__ import annotations

from dataclasses import dataclass, field

import pybase64 as base64

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import INPUT_SAMPLES_PER_UNIT

_BYTES_PER_SAMPLE = 4
WINDOW_BYTES = INPUT_SAMPLES_PER_UNIT * _BYTES_PER_SAMPLE


@dataclass
class WindowReservation(PcmAppendReservation):
    owner: Qwen3OmniDuplexPcmBuffer
    operation_id: str
    payload: dict[str, object]
    raw: bytes
    _active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self.raw)

    def commit(self) -> None:
        if self._active:
            self._active = False
            self.owner.reservations.remove(self)

    def rollback(self) -> None:
        if self._active:
            self.owner.buffer[:0] = self.raw
            self.commit()


@dataclass
class Qwen3OmniDuplexPcmBuffer(PcmAppendBuffer):
    """Hold incoming PCM and release it only in exact 7680-sample windows."""

    buffer: bytearray = field(default_factory=bytearray)
    reservations: list[WindowReservation] = field(default_factory=list)

    @property
    def pending_byte_count(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        for reservation in self.reservations:
            reservation._active = False
        self.reservations.clear()
        self.buffer.clear()

    def clear_force_listen(self) -> None:
        return None

    def has_pending(self) -> bool:
        return bool(self.buffer)

    def has_reserved(self) -> bool:
        return bool(self.reservations)

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> PcmAppendReservation | None:
        del chunk_period_ms
        if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != 16000:
            raise ValueError("Qwen3-Omni-Duplex requires normalized 16 kHz pcm_f32le")
        if payload.get("video_frames"):
            raise ValueError("Qwen3-Omni-Duplex Stage 0 does not accept video frames")
        raw = base64.b64decode(payload.get("audio", ""), validate=True)
        if len(raw) % _BYTES_PER_SAMPLE:
            raise ValueError("Invalid float32 PCM length")
        self.buffer.extend(raw)
        if not allow_emit or len(self.buffer) < WINDOW_BYTES:
            return None
        window = bytes(self.buffer[:WINDOW_BYTES])
        del self.buffer[:WINDOW_BYTES]
        reservation = WindowReservation(
            self,
            operation_id,
            {
                "audio": base64.b64encode(window).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            },
            window,
        )
        self.reservations.append(reservation)
        return reservation

    def prepare_commit(self, *, operation_id: str, chunk_period_ms: int) -> PcmAppendReservation | None:
        del operation_id, chunk_period_ms
        if self.buffer:
            raise ValueError(
                f"Qwen3-Omni-Duplex commit leaves {len(self.buffer)} bytes, not a whole {WINDOW_BYTES}-byte window"
            )
        return None

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        del chunk_period_ms
        if self.buffer:
            raise ValueError("Qwen3-Omni-Duplex cannot flush a partial audio window")
        return None
