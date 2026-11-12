# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 1 decode takes its text from the prefill's queue, whatever async_chunk says."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.qwen3_omni_duplex import (
    Qwen3OmniDuplexForConditionalGeneration,
)

HIDDEN = 2
PAD = torch.tensor([[-7.0, -7.0]])


class _Talker(Qwen3OmniDuplexForConditionalGeneration):
    def __init__(self, *, async_chunk: bool) -> None:
        self.model_stage = "talker"
        self.tts_pad_embed = PAD
        self.talker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=HIDDEN))
        # The flag the parent switches its text source on. The deploy sets it to
        # select the streamed Talker -> Code2Wav edge.
        self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk))


def _step(talker: _Talker, payload: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    embeds = torch.zeros(1, HIDDEN)
    return talker.talker_preprocess_decode(torch.zeros(1, dtype=torch.long), embeds, {}, payload)


@pytest.mark.parametrize("async_chunk", [False, True])
def test_each_step_takes_the_next_queued_text_row(async_chunk: bool) -> None:
    talker = _Talker(async_chunk=async_chunk)
    queue = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    # No upstream stage feeds the duplex Talker a live Thinker stream, so with
    # async_chunk on the parent would answer EOS and then pad here.
    _, text, update = _step(talker, {"hidden_states": {"trailing_text": queue}})
    assert text.tolist() == [[1.0, 1.0]]
    rest = update["hidden_states"]["trailing_text"]
    assert rest.tolist() == [[2.0, 2.0]]

    _, text, update = _step(talker, {"hidden_states": {"trailing_text": rest}})
    assert text.tolist() == [[2.0, 2.0]]
    assert update["hidden_states"]["trailing_text"].tolist() == PAD.tolist()


@pytest.mark.parametrize("async_chunk", [False, True])
def test_a_drained_queue_pads(async_chunk: bool) -> None:
    _, text, update = _step(_Talker(async_chunk=async_chunk), {"hidden_states": {}})

    assert text.tolist() == PAD.tolist()
    assert "hidden_states" not in update


def test_the_previous_step_hidden_state_feeds_the_code_predictor() -> None:
    last = torch.tensor([[[3.0, 4.0]]])

    hidden, _, _ = _step(_Talker(async_chunk=True), {"hidden_states": {"last": last}})
    assert hidden.tolist() == [[3.0, 4.0]]

    # Before talker_postprocess has run once there is nothing to read.
    hidden, _, _ = _step(_Talker(async_chunk=True), {"hidden_states": {}})
    assert hidden.tolist() == [[0.0, 0.0]]
