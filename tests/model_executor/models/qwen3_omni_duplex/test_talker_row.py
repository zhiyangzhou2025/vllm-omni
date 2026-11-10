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
    TalkerCtxAtom,
    concat_ctx_atoms,
    frame_budget,
    window_ctx_atoms,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

HIDDEN = 4
# The checkpoint's ``thinker_config.audio_token_id``. A duplex window fills its
# audio rows with ``unit_start`` instead, so an ID-derived mask selects nothing.
AUDIO_TOKEN_ID = 151675


def _window_ids() -> torch.Tensor:
    return torch.tensor(
        [
            DUPLEX_TOKENS.unit_start,
            AUDIO_START_TOKEN_ID,
            *([DUPLEX_TOKENS.unit_start] * 6),
            AUDIO_END_TOKEN_ID,
        ],
        dtype=torch.long,
    )


def _rows(base: float, count: int = 9) -> torch.Tensor:
    return base + torch.arange(count * HIDDEN, dtype=torch.float32).reshape(count, HIDDEN)


def _atoms(base: float = 0.0, *, closed: bool = True) -> tuple[TalkerCtxAtom, ...]:
    return window_ctx_atoms(
        input_ids=_window_ids(),
        embeds=_rows(base),
        hidden=_rows(base + 1000.0),
        unit_end_embed=torch.full((HIDDEN,), -1.0),
        closed=closed,
    )


def test_closed_window_yields_an_env_atom_and_a_close_atom() -> None:
    env, close = _atoms()

    assert env.input_ids.tolist() == _window_ids().tolist()
    assert torch.equal(env.embeds, _rows(0.0))
    assert torch.equal(env.hidden, _rows(1000.0))

    assert close.input_ids.tolist() == [DUPLEX_TOKENS.unit_end]
    assert close.embeds.tolist() == [[-1.0] * HIDDEN]
    # The close row is projected from its token embedding alone, so its hidden
    # state is unreachable padding.
    assert close.hidden.tolist() == [[0.0] * HIDDEN]
    assert close.multimodal_mask.tolist() == [False]


def test_the_answered_window_has_not_been_closed_yet() -> None:
    assert len(_atoms(closed=False)) == 1


def test_the_multimodal_mask_is_positional_not_id_derived() -> None:
    env, _ = _atoms()

    assert env.multimodal_mask.nonzero().flatten().tolist() == list(AUDIO_ROW_OFFSETS)
    assert int(env.multimodal_mask.sum()) == CODEC_FRAMES_PER_WINDOW
    # The measured failure of the non-duplex Talker path: its mask is built from
    # ``audio_token_id``, which never appears in a duplex window.
    assert int((env.input_ids == AUDIO_TOKEN_ID).sum()) == 0


def test_concatenated_atoms_keep_every_window_audio_rows_aligned() -> None:
    first = _atoms(0.0)
    second = _atoms(100.0, closed=False)

    ctx = concat_ctx_atoms([*first, *second])

    assert int(ctx.input_ids.shape[0]) == 9 + 1 + 9
    assert ctx.input_ids.tolist() == [*_window_ids().tolist(), DUPLEX_TOKENS.unit_end, *_window_ids().tolist()]
    assert ctx.multimodal_mask.nonzero().flatten().tolist() == [
        *AUDIO_ROW_OFFSETS,
        *(10 + offset for offset in AUDIO_ROW_OFFSETS),
    ]
    assert torch.equal(ctx.embeds[10:], _rows(100.0))
    assert torch.equal(ctx.hidden[10:], _rows(1100.0))


def test_concatenating_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one atom"):
        concat_ctx_atoms([])


@pytest.mark.parametrize(
    ("input_ids", "match"),
    [
        (torch.zeros(8, dtype=torch.long), "exactly 9 token IDs"),
        (torch.zeros(9, dtype=torch.long), "do not match"),
    ],
)
def test_a_window_that_is_not_the_exact_protocol_layout_is_rejected(input_ids: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        window_ctx_atoms(
            input_ids=input_ids,
            embeds=_rows(0.0),
            hidden=_rows(0.0),
            unit_end_embed=torch.zeros(HIDDEN),
            closed=False,
        )


def test_an_atom_whose_rows_disagree_is_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        TalkerCtxAtom(
            input_ids=_window_ids(),
            embeds=_rows(0.0),
            hidden=_rows(0.0)[:, :2],
            multimodal_mask=torch.zeros(9, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="same rows"):
        TalkerCtxAtom(
            input_ids=_window_ids(),
            embeds=_rows(0.0, count=8),
            hidden=_rows(0.0, count=8),
            multimodal_mask=torch.zeros(9, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="bool tensor"):
        TalkerCtxAtom(
            input_ids=_window_ids(),
            embeds=_rows(0.0),
            hidden=_rows(0.0),
            multimodal_mask=torch.zeros(9, dtype=torch.long),
        )


def test_only_the_final_window_may_stop_the_turn() -> None:
    assert not frame_budget(is_final_window=False).allow_codec_eos
    assert frame_budget(is_final_window=False).sampling_steps == CODEC_FRAMES_PER_WINDOW
    assert frame_budget(is_final_window=True).allow_codec_eos
    assert frame_budget(is_final_window=True).sampling_steps == CODEC_FRAMES_PER_WINDOW + 1


def test_a_turn_pays_one_step_per_frame_plus_one_flush() -> None:
    for windows in (1, 2, 54):
        steps = sum(frame_budget(is_final_window=index == windows - 1).sampling_steps for index in range(windows))
        # The code predictor runs one step behind the sampler, so a turn of
        # ``windows`` windows emits ``6 * windows`` frames and spends one extra
        # step flushing the frame the last window holds back.
        assert steps == CODEC_FRAMES_PER_WINDOW * windows + 1
