# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 sampling stays inside one Thinker window."""

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import Stage0Windows
from vllm_omni.model_executor.models.qwen3_omni_duplex.qwen3_omni_duplex import (
    Qwen3OmniDuplexForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


class _Stage(Qwen3OmniDuplexForConditionalGeneration):
    def __init__(self) -> None:
        self.model_stage = "thinker"
        self._stage0_windows = Stage0Windows()

    def _sample_with_parent(self, logits, sampling_metadata):
        return SimpleNamespace(sampled_token_ids=torch.argmax(logits, dim=-1, keepdim=True))


def _row(request_id: str, row_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(row_idx=row_idx, request_id=request_id)


def _metadata(*request_ids: str) -> SimpleNamespace:
    return SimpleNamespace(input_batch=SimpleNamespace(req_ids=list(request_ids)))


def test_control_step_can_only_sample_listen_or_speak() -> None:
    stage = _Stage()
    logits = torch.full((1, DUPLEX_TOKENS.cancel + 1), 5.0)
    logits[0, DUPLEX_TOKENS.listen] = 1.0
    logits[0, DUPLEX_TOKENS.speak] = 2.0

    stage.prepare_duplex_sampling(logits, None, (_row("req"),))
    sampled = stage.sample(logits, _metadata("req"))

    assert int(sampled.sampled_token_ids[0, 0]) == DUPLEX_TOKENS.speak
    assert torch.isneginf(logits[0, DUPLEX_TOKENS.listen - 1])


def test_listen_closes_the_window_and_reports_unit_end() -> None:
    stage = _Stage()
    logits = torch.full((1, DUPLEX_TOKENS.cancel + 1), float("-inf"))
    logits[0, DUPLEX_TOKENS.listen] = 3.0
    stage.prepare_duplex_sampling(logits, None, (_row("req"),))

    stage.sample(logits, _metadata("req"))
    replay = stage.post_sample_talker_mtp(req_ids=["req"], req_infos=[{}])

    assert replay.tolist() == [[DUPLEX_TOKENS.unit_end]]
    # The closed window is reported once, so the next step starts clean.
    assert stage.post_sample_talker_mtp(req_ids=["req"], req_infos=[{}]).tolist() == [[-1]]


def test_turn_end_reports_chunk_eos_and_unit_end() -> None:
    stage = _Stage()
    window = stage._stage0_windows.window_for("req")
    window.observe(DUPLEX_TOKENS.speak)
    window.observe(100)
    logits = torch.full((1, DUPLEX_TOKENS.cancel + 1), float("-inf"))
    logits[0, DUPLEX_TOKENS.turn_eos] = 4.0

    stage.sample(logits, _metadata("req"))
    replay = stage.post_sample_talker_mtp(req_ids=["req"], req_infos=[{}])

    # turn_eos is sampled, so the runner feeds it. The two tokens the protocol
    # appends after it still have to be replayed.
    assert replay.tolist() == [[DUPLEX_TOKENS.chunk_eos, DUPLEX_TOKENS.unit_end]]
