# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Row contract between the Stage 0 Thinker and the Qwen3-Omni-Duplex Talker.

The Thinker's prompt already is the context the Talker conditions on. Each
append reserves the previous window's ``<|unit_end|>`` and then nine rows for
the window itself, so the prompt reads

    prefix, window_0, unit_end, window_1, unit_end, ..., window_k

and the Talker prefills exactly those rows once per turn. What it needs on top
of them is which rows carry acoustic content: those are projected from the
Thinker hidden state at ``accept_hidden_layer``, every other row from the
Thinker input embedding. The non-duplex Talker path derives that from
``audio_token_id``, which a duplex window never contains, so the split has to
come from the window layout instead.

It comes from ``<|unit_end|>`` and the fixed window length rather than from the
window's own token IDs, because the engine reserves a window as nine
``<|unit_start|>`` placeholders and only the worker substitutes the audio
boundary IDs into them. ``<|unit_end|>`` is the one row that reads the same on
both sides of that substitution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_TOKENS_PER_UNIT,
    AUDIO_WINDOW_PROMPT_TOKENS,
    UNIT_START_TOKENS,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

# The acoustic rows of one window, counted from the window's first row. The
# window opens with ``<|unit_start|>`` and then an audio-start boundary.
AUDIO_ROW_OFFSETS = tuple(range(UNIT_START_TOKENS + 1, UNIT_START_TOKENS + 1 + AUDIO_TOKENS_PER_UNIT))
# One 0.48-second window is worth this many codec frames at 12.5 Hz.
CODEC_FRAMES_PER_WINDOW = AUDIO_TOKENS_PER_UNIT
# ``[tts_pad, tts_pad, tts_pad, tts_pad, tts_bos, first_text]`` summed with six
# codec special rows, and the whole of what a turn adds to the context rows. The
# non-duplex Talker spends nine rows here: it splits the Thinker prompt into
# ChatML segments and keeps the three ``<|im_start|>assistant\n`` rows it split
# on. A window run has no such header to keep.
BOOTSTRAP_ROWS = 6


@dataclass(frozen=True, slots=True)
class FrameBudget:
    """What one window of speech costs the Talker."""

    sampling_steps: int
    """Talker decode steps this window is scheduled for."""
    allow_codec_eos: bool
    """Whether the window may sample ``codec_eos`` and end the turn."""


def audio_row_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Mark the rows of a Thinker prompt that carry acoustic content.

    Every ``<|unit_end|>`` closes the window in the rows just before it, and the
    prompt's last ``AUDIO_WINDOW_PROMPT_TOKENS`` rows are the window still open.
    Inside a window the acoustic rows are the fixed ``AUDIO_ROW_OFFSETS``, so
    each window contributes exactly ``CODEC_FRAMES_PER_WINDOW`` rows.
    """

    ids = input_ids.reshape(-1)
    length = int(ids.shape[0])
    closed = torch.nonzero(ids == DUPLEX_TOKENS.unit_end).flatten()
    ends = torch.cat([closed, torch.tensor([length], dtype=closed.dtype, device=ids.device)])
    starts = ends - AUDIO_WINDOW_PROMPT_TOKENS
    # A negative start would wrap around and silently mark rows of an earlier
    # window, so the caller hears about a prompt that is not window-shaped.
    if int(starts.min()) < 0:
        raise ValueError(f"no {AUDIO_WINDOW_PROMPT_TOKENS}-row window fits before every <|unit_end|> in {length} rows")
    offsets = torch.tensor(AUDIO_ROW_OFFSETS, dtype=starts.dtype, device=ids.device)
    mask = torch.zeros(length, dtype=torch.bool, device=ids.device)
    mask[(starts.unsqueeze(1) + offsets).flatten()] = True
    return mask


def frame_budget(*, is_final_window: bool) -> FrameBudget:
    """Decode steps and stop permission for one window of speech.

    A window is worth ``CODEC_FRAMES_PER_WINDOW`` codec frames, and the code
    predictor runs one step behind the sampler: step ``k`` predicts the codes of
    the hidden state the forward of step ``k - 1`` produced. A window therefore
    costs one decode step per frame and holds its last frame back, which the
    first step of the next window flushes out.

    The final window has no next window, so it pays one extra step to flush the
    frame it holds back. That step's token is discarded unless it is
    ``codec_eos``, which only the final window is allowed to sample.
    """

    if is_final_window:
        return FrameBudget(sampling_steps=CODEC_FRAMES_PER_WINDOW + 1, allow_codec_eos=True)
    return FrameBudget(sampling_steps=CODEC_FRAMES_PER_WINDOW, allow_codec_eos=False)


__all__ = [
    "AUDIO_ROW_OFFSETS",
    "BOOTSTRAP_ROWS",
    "CODEC_FRAMES_PER_WINDOW",
    "FrameBudget",
    "audio_row_mask",
    "frame_budget",
]
