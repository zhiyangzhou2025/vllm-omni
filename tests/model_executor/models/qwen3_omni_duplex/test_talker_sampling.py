# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 1 sampling may only end a turn on the turn's last window."""

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.qwen3_omni_duplex import (
    Qwen3OmniDuplexForConditionalGeneration,
)

# ``talker_config.codec_eos_token_id``, read from the checkpoint at runtime.
CODEC_EOS_TOKEN_ID = 2150
VOCAB = CODEC_EOS_TOKEN_ID + 10


class _Talker(Qwen3OmniDuplexForConditionalGeneration):
    def __init__(self) -> None:
        self.model_stage = "talker"
        self.talker_config = SimpleNamespace(codec_eos_token_id=CODEC_EOS_TOKEN_ID)


def _row(*, final_window: bool, row_idx: int = 0) -> SimpleNamespace:
    # The runner builds these from the request's duplex payload; only the
    # final-window flag matters to the frame budget.
    return SimpleNamespace(row_idx=row_idx, request_id="req", payload={"final_window": final_window})


def test_a_window_the_thinker_followed_with_more_audio_cannot_end_the_turn() -> None:
    logits = torch.full((1, VOCAB), 5.0)

    _Talker().prepare_duplex_sampling(logits, None, (_row(final_window=False),))

    assert logits[0, CODEC_EOS_TOKEN_ID] == float("-inf")
    # Nothing else is a protocol token on this stage, so nothing else is masked.
    assert int(torch.isfinite(logits[0]).sum()) == VOCAB - 1


def test_the_last_window_of_a_turn_keeps_its_stop_token() -> None:
    logits = torch.full((1, VOCAB), 5.0)

    _Talker().prepare_duplex_sampling(logits, None, (_row(final_window=True),))

    assert bool(torch.isfinite(logits).all())


def test_each_row_is_judged_on_its_own_window() -> None:
    logits = torch.full((2, VOCAB), 5.0)

    _Talker().prepare_duplex_sampling(
        logits,
        None,
        (_row(final_window=False, row_idx=0), _row(final_window=True, row_idx=1)),
    )

    assert logits[0, CODEC_EOS_TOKEN_ID] == float("-inf")
    assert logits[1, CODEC_EOS_TOKEN_ID] == 5.0


def test_a_row_without_a_duplex_payload_is_left_alone() -> None:
    logits = torch.full((1, VOCAB), 5.0)

    # Turn-based serving reaches the same hook with no window to pace.
    _Talker().prepare_duplex_sampling(logits, None, (SimpleNamespace(row_idx=0, request_id="req", payload=None),))

    assert bool(torch.isfinite(logits).all())


def test_the_talker_never_runs_the_thinker_window_constraint() -> None:
    stage = _Talker()
    # The Thinker's constraint reads _stage0_windows, which only the thinker
    # stage builds, and it masks every token outside the current phase.
    assert not hasattr(stage, "_stage0_windows")

    logits = torch.full((1, VOCAB), 5.0)
    stage.prepare_duplex_sampling(logits, None, (_row(final_window=True),))

    assert bool(torch.isfinite(logits).all())


def test_the_talker_leaves_the_draw_to_the_runner() -> None:
    stage = _Talker()
    # The stage module the parent would forward to. The real one is
    # Qwen3OmniMoeTalkerForConditionalGeneration, which implements no sampler,
    # so forwarding raises instead of drawing.
    stage.model = SimpleNamespace()

    # prefer_model_sampler is set so the runner calls prepare_duplex_sampling,
    # not because this stage draws. None is the runner's contract for declining,
    # and it then samples the layer-0 codec id from the masked logits itself.
    assert stage.sample(torch.full((1, VOCAB), 5.0), None) is None
