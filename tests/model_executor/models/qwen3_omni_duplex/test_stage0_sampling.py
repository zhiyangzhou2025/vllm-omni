# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 sampling stays inside one Thinker window."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    Stage0Windows,
    ThinkerProtocolError,
)
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
    assert int(torch.isfinite(logits[0]).sum()) == 1
    sampled = stage.sample(logits, _metadata("req"))

    assert int(sampled.sampled_token_ids[0, 0]) == DUPLEX_TOKENS.unit_end
    # The closed window is dropped, so the next step starts a fresh window.
    assert stage._stage0_windows.window_for("req").state.phase == "control"


def test_a_phase_whose_only_token_was_already_masked_fails_loudly() -> None:
    stage = _Stage()
    stage._stage0_windows.window_for("req").observe(DUPLEX_TOKENS.listen)

    # ``<|unit_end|>`` is the request's stop token, so an upstream processor can
    # mask it. Leaving the row all ``-inf`` would make the sampler draw an
    # arbitrary token that the window then rejects with no trace.
    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), 5.0)
    logits[0, DUPLEX_TOKENS.unit_end] = float("-inf")

    with pytest.raises(ThinkerProtocolError):
        stage.prepare_duplex_sampling(logits, None, (_row("req"),))


def test_text_phases_keep_text_and_drop_only_the_illegal_protocol_tokens() -> None:
    stage = _Stage()
    window = stage._stage0_windows.window_for("req")
    window.observe(DUPLEX_TOKENS.speak)

    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), 5.0)
    stage.prepare_duplex_sampling(logits, None, (_row("req"),))

    # ``<|speak|>`` opens the text phase, so ordinary tokens stay drawable and
    # only the protocol tokens the phase forbids are masked.
    assert torch.isfinite(logits[0, 100])
    assert torch.isfinite(logits[0, DUPLEX_TOKENS.unit_end])
    assert torch.isfinite(logits[0, DUPLEX_TOKENS.turn_eos])
    assert torch.isfinite(logits[0, DUPLEX_TOKENS.launch])
    assert torch.isneginf(logits[0, DUPLEX_TOKENS.chunk_eos])
    assert torch.isneginf(logits[0, DUPLEX_TOKENS.listen])
    assert torch.isneginf(logits[0, DUPLEX_TOKENS.cancel])


def _close_speak_window(stage: _Stage, *actions: int) -> None:
    window = stage._stage0_windows.window_for("req")
    for token_id in (DUPLEX_TOKENS.speak, *actions, DUPLEX_TOKENS.unit_end):
        window.observe(token_id)
    assert window.closed


def _speak_logits(stage: _Stage) -> torch.Tensor:
    stage._stage0_windows.window_for("req").observe(DUPLEX_TOKENS.speak)
    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), 5.0)
    stage.prepare_duplex_sampling(logits, None, (_row("req"),))
    return logits


def test_system2_generation_outlives_its_window_until_cancel() -> None:
    stage = _Stage()
    _close_speak_window(stage, DUPLEX_TOKENS.launch)
    # Two windows later the launched generation still runs: only cancel is legal.
    _close_speak_window(stage)
    logits = _speak_logits(stage)
    assert torch.isfinite(logits[0, DUPLEX_TOKENS.cancel])
    assert torch.isneginf(logits[0, DUPLEX_TOKENS.launch])
    stage._stage0_windows.window_for("req").observe(DUPLEX_TOKENS.cancel)
    stage._stage0_windows.window_for("req").observe(DUPLEX_TOKENS.unit_end)

    logits = _speak_logits(stage)
    assert torch.isfinite(logits[0, DUPLEX_TOKENS.launch])
    assert torch.isneginf(logits[0, DUPLEX_TOKENS.cancel])
    # Another request never sees this request's generation.
    assert not stage._stage0_windows.window_for("other").state.active_generation


def test_cancel_then_launch_keeps_a_generation_running() -> None:
    stage = _Stage()
    _close_speak_window(stage, DUPLEX_TOKENS.launch)
    _close_speak_window(stage, DUPLEX_TOKENS.cancel, DUPLEX_TOKENS.launch)

    assert stage._stage0_windows.window_for("req").state.active_generation


@pytest.mark.parametrize(("result", "active"), [("partial", True), ("final", False)])
def test_only_a_final_tool_result_ends_the_generation(result: str, active: bool) -> None:
    stage = _Stage()
    _close_speak_window(stage, DUPLEX_TOKENS.launch)
    logits = torch.full((1, DUPLEX_TOKENS.non_idle + 1), 5.0)
    row = SimpleNamespace(row_idx=0, request_id="req", payload={"system2_result": result})
    stage.prepare_duplex_sampling(logits, None, (row,))
    # The result applies when a window opens; later steps of it do not reset it.
    assert stage._stage0_windows.window_for("req").state.active_generation is active
    stage._stage0_windows.window_for("req").observe(DUPLEX_TOKENS.listen)
    stage.prepare_duplex_sampling(torch.full_like(logits, 5.0), None, (_row("req"),))
    assert stage._stage0_windows.window_for("req").state.active_generation is active


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
    assert int(torch.isfinite(logits[0]).sum()) == 1
    sampled = stage.sample(logits, _metadata("req"))

    assert int(sampled.sampled_token_ids[0, 0]) == DUPLEX_TOKENS.unit_end
    assert stage._stage0_windows.window_for("req").state.phase == "control"


def test_the_stage_0_hook_is_hidden_from_the_talker_stage() -> None:
    class _Talker(Qwen3OmniDuplexForConditionalGeneration):
        def __init__(self) -> None:
            self.model_stage = "talker"

    stage = _Talker()
    # The runner enables this path for any callable attribute, and the Stage 0
    # implementation answers with placeholder rows instead of codec codes.
    assert callable(stage.post_sample_talker_mtp)

    stage._wire_stage_hooks()

    assert stage.post_sample_talker_mtp is None
