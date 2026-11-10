# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 1 prefill reads a window run, not a ChatML transcript."""

from collections.abc import Iterator
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.talker_row import BOOTSTRAP_ROWS
from vllm_omni.model_executor.models.qwen3_omni_duplex.qwen3_omni_duplex import (
    Qwen3OmniDuplexForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import AUDIO_WINDOW_PROMPT_TOKENS
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

THINKER_HIDDEN = 4
TALKER_HIDDEN = 2
# ``talker_config`` and ``config.tts_pad_token_id``, read from the checkpoint.
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157
CODEC_PAD_ID = 2148
CODEC_BOS_ID = 2149
TTS_PAD_TOKEN_ID = 151671
SPEAKER_ID = 3000
# Sentinel values the fake projections carry through unchanged, so every
# returned row says which thinker row and which channel it came from.
TTS_PAD = 10000.0
TTS_BOS = 20000.0
TTS_EOS = 30000.0

# ``<|im_start|>system``: the prefix the first append carries, and the row the
# non-duplex Talker splits the prompt on.
PREFIX = [151644, 8948]
# Two windows as the engine reserves them, with the closing <|unit_end|> the
# second append puts between them.
WINDOW = [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS
PROMPT = [*PREFIX, *WINDOW, DUPLEX_TOKENS.unit_end, *WINDOW]
# ``AUDIO_ROW_OFFSETS`` inside each of the two windows.
ACOUSTIC_ROWS = [4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19]
# The Thinker spoke three text tokens in this window, on rows 22, 23 and 24.
SPOKEN = [151679, 1111, 2222, 3333, DUPLEX_TOKENS.unit_end]
FIRST_TEXT_ROW = 22.0
QUEUED_TEXT_ROWS = [23.0, 24.0]


class _FakeTalker:
    """Projections that keep each row's provenance readable.

    ``text_projection`` keeps a thinker row's first two columns and
    ``hidden_projection`` its last two, so a projected row reads ``[n, 0]`` when
    it came from thinker row ``n``'s input embedding and ``[n, 1]`` when it came
    from that row's accepted hidden state. ``embed_input_ids`` marks the codec
    channel with ``-1``.

    Not an ``nn.Module``: the stage under test is one, and assigning a submodule
    to it without running ``nn.Module.__init__`` raises. ``_module_device`` only
    needs these two to answer CPU.
    """

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter(())

    def named_buffers(self, recurse: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(())

    def text_projection(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[..., :TALKER_HIDDEN]

    def hidden_projection(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[..., TALKER_HIDDEN:]

    def embed_input_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        codes = token_ids.to(torch.float32)
        return torch.stack([codes, torch.full_like(codes, -1.0)], dim=-1)


class _Talker(Qwen3OmniDuplexForConditionalGeneration):
    def __init__(self) -> None:
        self.model_stage = "talker"
        self.talker = _FakeTalker()
        self.talker_config = SimpleNamespace(
            codec_nothink_id=CODEC_NOTHINK_ID,
            codec_think_bos_id=CODEC_THINK_BOS_ID,
            codec_think_eos_id=CODEC_THINK_EOS_ID,
            codec_pad_id=CODEC_PAD_ID,
            codec_bos_id=CODEC_BOS_ID,
            text_config=SimpleNamespace(hidden_size=TALKER_HIDDEN),
        )
        self.config = SimpleNamespace(talker_config=self.talker_config, tts_pad_token_id=TTS_PAD_TOKEN_ID)


def _tts_row(value: float) -> torch.Tensor:
    # The payload ships these as [1, 1, thinker_hidden].
    return torch.tensor([[[value, 0.0, 0.0, 0.0]]])


def _thinker_rows(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    embed = torch.zeros(rows, THINKER_HIDDEN)
    embed[:, 0] = torch.arange(rows, dtype=torch.float32)
    hidden = torch.zeros(rows, THINKER_HIDDEN)
    hidden[:, TALKER_HIDDEN] = torch.arange(rows, dtype=torch.float32)
    hidden[:, TALKER_HIDDEN + 1] = 1.0
    return embed, hidden


def _prefill(generated: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    all_ids = [*PROMPT, *(SPOKEN if generated is None else generated)]
    # The payload drops the Thinker's terminal stop row, so the conditioning
    # tensors are one row shorter than the ids.
    embed, hidden = _thinker_rows(len(all_ids) - 1)
    prompt_ids = torch.tensor([PROMPT], dtype=torch.long)
    result_ids = torch.tensor(all_ids, dtype=torch.long)
    return _Talker()._thinker_to_talker_prefill(
        thinker_embed=embed,
        thinker_hidden=hidden,
        multimodal_mask=None,
        input_ids=prompt_ids,
        thinker_result_ids=result_ids,
        speaker_id=SPEAKER_ID,
        tts_bos_thinker=_tts_row(TTS_BOS),
        tts_eos_thinker=_tts_row(TTS_EOS),
        tts_pad_thinker=_tts_row(TTS_PAD),
        input_ids_cpu=prompt_ids,
        thinker_result_ids_cpu=result_ids,
    )


def test_the_whole_window_run_is_one_context_span() -> None:
    input_ids, embeds, _ = _prefill()

    assert int(embeds.shape[0]) == len(PROMPT) + BOOTSTRAP_ROWS
    # Splitting on <|im_start|> would keep the prefix and throw both windows
    # away, because a window run has no role header after the prefix.
    assert input_ids[: len(PROMPT)].tolist() == PROMPT
    assert embeds[: len(PROMPT), 0].tolist() == [float(row) for row in range(len(PROMPT))]


def test_only_the_window_audio_rows_take_the_accepted_hidden_state() -> None:
    _, embeds, _ = _prefill()

    context = embeds[: len(PROMPT)]
    from_hidden = torch.nonzero(context[:, 1] == 1.0).flatten().tolist()
    # The prefix and the window's own boundary rows stay on the input embedding.
    assert from_hidden == ACOUSTIC_ROWS


def test_a_turn_opens_with_six_bootstrap_rows() -> None:
    input_ids, embeds, _ = _prefill()

    bootstrap = embeds[len(PROMPT) :]
    assert int(bootstrap.shape[0]) == BOOTSTRAP_ROWS
    assert input_ids[len(PROMPT) :].tolist() == [TTS_PAD_TOKEN_ID] * BOOTSTRAP_ROWS
    # Every bootstrap row sums a text row with a codec row.
    assert bootstrap[:, 1].tolist() == [-1.0] * BOOTSTRAP_ROWS
    assert bootstrap[:, 0].tolist() == [
        TTS_PAD + CODEC_NOTHINK_ID,
        TTS_PAD + CODEC_THINK_BOS_ID,
        TTS_PAD + CODEC_THINK_EOS_ID,
        TTS_PAD + SPEAKER_ID,
        TTS_BOS + CODEC_PAD_ID,
        FIRST_TEXT_ROW + CODEC_BOS_ID,
    ]


def test_the_first_spoken_row_opens_the_turn_and_the_rest_queue_for_decode() -> None:
    _, embeds, trailing = _prefill()

    assert float(embeds[-1, 0]) == FIRST_TEXT_ROW + CODEC_BOS_ID
    assert trailing[:, 0].tolist() == QUEUED_TEXT_ROWS


def test_protocol_tokens_never_become_text_steps() -> None:
    # A listen window: the Thinker paced the window and said nothing.
    _, embeds, trailing = _prefill([DUPLEX_TOKENS.listen, DUPLEX_TOKENS.unit_end])

    assert trailing is None
    # No text to open with, so the bootstrap's last row is codec only.
    assert float(embeds[-1, 0]) == CODEC_BOS_ID


def test_a_window_the_thinker_will_follow_does_not_end_the_turn_text() -> None:
    _, _, trailing = _prefill()

    # The parent appends <|tts_eos|> to this queue unconditionally, which would
    # end the turn's text on whichever window prefilled it. The decode path
    # emits it instead, once the queue runs dry.
    assert not bool((trailing == TTS_EOS).any())
