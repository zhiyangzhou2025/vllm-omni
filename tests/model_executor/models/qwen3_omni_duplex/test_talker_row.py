# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.audio_window import (
    AUDIO_END_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.talker_row import (
    AUDIO_ROW_OFFSETS,
    CODEC_FRAMES_PER_WINDOW,
    audio_row_mask,
    frame_budget,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import AUDIO_WINDOW_PROMPT_TOKENS
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

# The checkpoint's ``thinker_config.audio_token_id``. A duplex window never
# contains it, which is why the non-duplex Talker's mask cannot be reused.
AUDIO_TOKEN_ID = 151675
# Stands in for the system prefix the first append carries.
PREFIX = (151644, 8948, 872)


def _reserved_window() -> list[int]:
    """One window as the engine reserves it: placeholders only."""

    return [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS


def _substituted_window() -> list[int]:
    """One window as the worker leaves it, with the audio boundaries in place."""

    return [
        DUPLEX_TOKENS.unit_start,
        AUDIO_START_TOKEN_ID,
        *([DUPLEX_TOKENS.unit_start] * CODEC_FRAMES_PER_WINDOW),
        AUDIO_END_TOKEN_ID,
    ]


def _prompt(window: list[int], *, windows: int) -> torch.Tensor:
    """``prefix, window_0, unit_end, window_1, ..., window_k``."""

    rows = list(PREFIX)
    for index in range(windows):
        if index:
            rows.append(DUPLEX_TOKENS.unit_end)
        rows.extend(window)
    return torch.tensor(rows, dtype=torch.long)


def _expected_rows(windows: int) -> list[int]:
    rows: list[int] = []
    for index in range(windows):
        start = len(PREFIX) + index * (AUDIO_WINDOW_PROMPT_TOKENS + 1)
        rows.extend(start + offset for offset in AUDIO_ROW_OFFSETS)
    return rows


@pytest.mark.parametrize("window", [_reserved_window(), _substituted_window()], ids=["reserved", "substituted"])
@pytest.mark.parametrize("windows", [1, 2, 54])
def test_every_window_contributes_its_acoustic_rows(window: list[int], windows: int) -> None:
    mask = audio_row_mask(_prompt(window, windows=windows))

    assert mask.nonzero().flatten().tolist() == _expected_rows(windows)
    assert int(mask.sum()) == CODEC_FRAMES_PER_WINDOW * windows


def test_the_mask_reads_the_same_on_both_sides_of_the_worker_substitution() -> None:
    # The engine reserves a window as nine ``<|unit_start|>`` placeholders and
    # only the worker writes the audio boundary IDs into it. A mask keyed on
    # those IDs would answer differently depending on which side asked.
    reserved = audio_row_mask(_prompt(_reserved_window(), windows=3))
    substituted = audio_row_mask(_prompt(_substituted_window(), windows=3))

    assert torch.equal(reserved, substituted)


def test_an_id_derived_mask_would_select_nothing() -> None:
    prompt = _prompt(_substituted_window(), windows=3)

    # The measured failure of the non-duplex Talker path: it builds its
    # multimodal mask from ``audio_token_id``, which no duplex window carries.
    assert int((prompt == AUDIO_TOKEN_ID).sum()) == 0
    assert int(audio_row_mask(prompt).sum()) == CODEC_FRAMES_PER_WINDOW * 3


def test_the_prefix_and_the_boundary_rows_are_not_acoustic() -> None:
    prompt = _prompt(_substituted_window(), windows=2)
    mask = audio_row_mask(prompt)

    assert not bool(mask[: len(PREFIX)].any())
    assert not bool(mask[prompt == AUDIO_START_TOKEN_ID].any())
    assert not bool(mask[prompt == AUDIO_END_TOKEN_ID].any())
    assert not bool(mask[prompt == DUPLEX_TOKENS.unit_end].any())


@pytest.mark.parametrize(
    "prompt",
    [
        [DUPLEX_TOKENS.unit_start] * (AUDIO_WINDOW_PROMPT_TOKENS - 1),
        # A closure with no window in front of it.
        [DUPLEX_TOKENS.unit_end, *_reserved_window()],
    ],
    ids=["short-prompt", "unanchored-closure"],
)
def test_a_prompt_that_is_not_window_shaped_is_rejected(prompt: list[int]) -> None:
    with pytest.raises(ValueError, match="window fits"):
        audio_row_mask(torch.tensor(prompt, dtype=torch.long))


def test_only_the_final_window_may_stop_the_turn() -> None:
    assert not frame_budget(is_final_window=False).allow_codec_eos
    assert frame_budget(is_final_window=False).sampling_steps == CODEC_FRAMES_PER_WINDOW
    assert frame_budget(is_final_window=True).allow_codec_eos
    assert frame_budget(is_final_window=True).sampling_steps == CODEC_FRAMES_PER_WINDOW + 1


def test_a_turn_pays_one_step_per_frame_plus_one_flush() -> None:
    for windows in (1, 2, 54):
        steps = sum(frame_budget(is_final_window=index == windows - 1).sampling_steps for index in range(windows))
        # The code predictor runs one step behind the sampler, so a turn of
        # ``windows`` windows emits ``CODEC_FRAMES_PER_WINDOW * windows`` frames
        # and spends one extra step flushing the frame the last window holds.
        assert steps == CODEC_FRAMES_PER_WINDOW * windows + 1
