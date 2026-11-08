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
        self._stage0_windows = Stage0Windows("none")

    def _sample_with_parent(self, logits, sampling_metadata):
        return SimpleNamespace(sampled_token_ids=torch.argmax(logits, dim=-1, keepdim=True))


def _row(request_id: str, row_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(row_idx=row_idx, request_id=request_id)


def _metadata(*request_ids: str) -> SimpleNamespace:
    return SimpleNamespace(input_batch=SimpleNamespace(req_ids=list(request_ids)))


def test_control_step_can_only_sample_listen_or_speak() -> None:
    stage = _Stage()
    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), 5.0)
    logits[0, DUPLEX_TOKENS.listen] = 1.0
    logits[0, DUPLEX_TOKENS.speak] = 2.0

    stage.prepare_duplex_sampling(logits, None, (_row("req"),))
    sampled = stage.sample(logits, _metadata("req"))

    assert int(sampled.sampled_token_ids[0, 0]) == DUPLEX_TOKENS.speak
    assert torch.isneginf(logits[0, DUPLEX_TOKENS.listen - 1])


def test_listen_forces_unit_end_before_the_window_closes() -> None:
    stage = _Stage()
    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), float("-inf"))
    logits[0, DUPLEX_TOKENS.listen] = 3.0
    stage.prepare_duplex_sampling(logits, None, (_row("req"),))

    # The runner samples itself and reports the token through post_sample_talker_mtp.
    stage.post_sample_talker_mtp(req_ids=["req"], req_infos=[{}], input_ids=torch.tensor([[DUPLEX_TOKENS.listen]]))
    assert stage._stage0_windows.window_for("req").state.phase == "post_listen"

    # The constraint leaves exactly one legal token on the next step, and that
    # token is the request's stop token.
    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), 5.0)
    stage.prepare_duplex_sampling(logits, None, (_row("req"),))
    sampled = stage.sample(logits, _metadata("req"))

    assert int(sampled.sampled_token_ids[0, 0]) == DUPLEX_TOKENS.unit_end
    # The closed window is dropped, so the next step starts a fresh window.
    assert stage._stage0_windows.window_for("req").state.phase == "control"


def test_turn_eos_forces_unit_end_before_the_window_closes() -> None:
    stage = _Stage()
    window = stage._stage0_windows.window_for("req")
    window.observe(DUPLEX_TOKENS.speak)
    window.observe(100)

    # The runner samples turn_eos itself and reports it afterwards. The turn
    # ends here, but the window stays open until the forced unit_end.
    stage.post_sample_talker_mtp(req_ids=["req"], req_infos=[{}], input_ids=torch.tensor([[DUPLEX_TOKENS.turn_eos]]))
    assert stage._stage0_windows.window_for("req").state.phase == "post_turn_eos"

    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), 5.0)
    stage.prepare_duplex_sampling(logits, None, (_row("req"),))
    sampled = stage.sample(logits, _metadata("req"))

    assert int(sampled.sampled_token_ids[0, 0]) == DUPLEX_TOKENS.unit_end
    assert stage._stage0_windows.window_for("req").state.phase == "control"
