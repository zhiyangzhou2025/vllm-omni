# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Row contract between a Stage 0 window and the Qwen3-Omni-Duplex Talker.

The Talker reads the Thinker's turn as a sequence of context atoms rather than
the ChatML turn the non-duplex Qwen3-Omni path assembles. One window of the
Thinker contributes two atoms:

``env``
    The nine Thinker rows ``unit_start, audio_start, audio x 6, audio_end``.
    Only the six audio rows carry acoustic information, so only those are
    projected from the captured Thinker hidden state; the other three are
    projected from the Thinker input embeddings.
``close``
    The single ``unit_end`` row the Thinker sampled to end the window. It has
    no acoustic content, so it is projected from its token embedding alone.

Splitting a window this way produces exactly what
``Qwen3OmniMoeTalker.project_thinker_outputs`` consumes: one boolean row mask
over the concatenated context. The mask is positional because the duplex window
fills its audio rows with ``unit_start``, not with the checkpoint's
``audio_token_id``, so a mask derived from token IDs selects nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.audio_window import (
    AUDIO_END_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_TOKENS_PER_UNIT,
    AUDIO_WINDOW_PROMPT_TOKENS,
    UNIT_START_TOKENS,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

# The acoustic rows of one window, counted from the window's first row.
AUDIO_ROW_OFFSETS = tuple(range(UNIT_START_TOKENS + 1, UNIT_START_TOKENS + 1 + AUDIO_TOKENS_PER_UNIT))
# One 0.48-second window is worth this many codec frames at 12.5 Hz.
CODEC_FRAMES_PER_WINDOW = AUDIO_TOKENS_PER_UNIT

# The token IDs ``build_audio_window`` lays a window out with.
_WINDOW_TOKEN_IDS = (
    DUPLEX_TOKENS.unit_start,
    AUDIO_START_TOKEN_ID,
    *([DUPLEX_TOKENS.unit_start] * AUDIO_TOKENS_PER_UNIT),
    AUDIO_END_TOKEN_ID,
)


@dataclass(frozen=True, slots=True)
class TalkerCtxAtom:
    """Thinker rows the Talker prefills, with the projection mask they need."""

    input_ids: torch.Tensor
    """``[rows]`` Thinker token IDs."""
    embeds: torch.Tensor
    """``[rows, thinker_hidden]`` Thinker input embeddings."""
    hidden: torch.Tensor
    """``[rows, thinker_hidden]`` Thinker hidden states at ``accept_hidden_layer``."""
    multimodal_mask: torch.Tensor
    """``[rows]`` boolean, true where the row is projected from ``hidden``."""

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1 or self.multimodal_mask.ndim != 1:
            raise ValueError("context atom token IDs and mask must be one-dimensional")
        if self.embeds.ndim != 2 or self.hidden.ndim != 2:
            raise ValueError("context atom embeddings and hidden states must be two-dimensional")
        if self.multimodal_mask.dtype is not torch.bool:
            raise ValueError("context atom multimodal mask must be a bool tensor")
        if self.hidden.shape != self.embeds.shape:
            raise ValueError("context atom embeddings and hidden states must have the same shape")
        rows = int(self.input_ids.shape[0])
        if int(self.embeds.shape[0]) != rows or int(self.multimodal_mask.shape[0]) != rows:
            raise ValueError("context atom token IDs, embeddings and mask must describe the same rows")
        devices = {self.input_ids.device, self.embeds.device, self.hidden.device, self.multimodal_mask.device}
        if len(devices) != 1:
            raise ValueError("context atom tensors must share a device")


@dataclass(frozen=True, slots=True)
class FrameBudget:
    """What one window of speech costs the Talker."""

    sampling_steps: int
    """Talker decode steps this window is scheduled for."""
    allow_codec_eos: bool
    """Whether the window may sample ``codec_eos`` and end the turn."""


def window_ctx_atoms(
    *,
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    hidden: torch.Tensor,
    unit_end_embed: torch.Tensor,
    closed: bool,
) -> tuple[TalkerCtxAtom, ...]:
    """Split one exact Stage 0 window into the atoms the Talker prefills.

    ``closed`` appends the ``unit_end`` atom the Thinker ended the window with.
    The window the Talker is answering has not been closed yet, so it
    contributes the ``env`` atom alone.
    """

    _validate_window(input_ids)
    mask = torch.zeros(AUDIO_WINDOW_PROMPT_TOKENS, dtype=torch.bool, device=input_ids.device)
    mask[list(AUDIO_ROW_OFFSETS)] = True
    env = TalkerCtxAtom(input_ids=input_ids, embeds=embeds, hidden=hidden, multimodal_mask=mask)
    if not closed:
        return (env,)
    return (env, _close_atom(unit_end_embed, device=input_ids.device))


def concat_ctx_atoms(atoms: Sequence[TalkerCtxAtom]) -> TalkerCtxAtom:
    """Join atoms into the single context the Talker prefills in one forward."""

    if not atoms:
        raise ValueError("a Talker context needs at least one atom")
    return TalkerCtxAtom(
        input_ids=torch.cat([atom.input_ids for atom in atoms]),
        embeds=torch.cat([atom.embeds for atom in atoms]),
        hidden=torch.cat([atom.hidden for atom in atoms]),
        multimodal_mask=torch.cat([atom.multimodal_mask for atom in atoms]),
    )


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


def _close_atom(unit_end_embed: torch.Tensor, *, device: torch.device) -> TalkerCtxAtom:
    if unit_end_embed.ndim not in (1, 2) or unit_end_embed.numel() != unit_end_embed.shape[-1]:
        raise ValueError("the unit_end embedding must be a single row")
    embed = unit_end_embed.reshape(1, -1)
    return TalkerCtxAtom(
        input_ids=torch.tensor([DUPLEX_TOKENS.unit_end], dtype=torch.long, device=device),
        embeds=embed,
        # The close row is never projected from a hidden state, so the Talker
        # cannot read these zeros.
        hidden=torch.zeros_like(embed),
        multimodal_mask=torch.zeros(1, dtype=torch.bool, device=device),
    )


def _validate_window(input_ids: torch.Tensor) -> None:
    if input_ids.ndim != 1 or int(input_ids.shape[0]) != AUDIO_WINDOW_PROMPT_TOKENS:
        raise ValueError(f"a Talker context window requires exactly {AUDIO_WINDOW_PROMPT_TOKENS} token IDs")
    if tuple(int(token_id) for token_id in input_ids.tolist()) != _WINDOW_TOKEN_IDS:
        raise ValueError("window token IDs do not match the Qwen3-Omni-Duplex audio window layout")


__all__ = [
    "AUDIO_ROW_OFFSETS",
    "CODEC_FRAMES_PER_WINDOW",
    "FrameBudget",
    "TalkerCtxAtom",
    "concat_ctx_atoms",
    "frame_budget",
    "window_ctx_atoms",
]
