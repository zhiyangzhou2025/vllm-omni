# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.qwen3_omni_duplex.qwen3_omni_duplex import (
    Qwen3OmniDuplexForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_WINDOW_PROMPT_TOKENS,
    build_audio_window_prompt,
    plan_history_trim,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS
from vllm_omni.model_executor.stage_input_processors.qwen3_omni_duplex import thinker2talker


def test_audio_window_reserves_exact_thinker_tokens() -> None:
    prompt = build_audio_window_prompt(
        request_id="request",
        fence=DuplexFence(session_id="session", epoch=0, turn_id=1),
        session_config={},
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload={"audio": b"pcm"},
        final=False,
    )

    assert prompt["prompt_token_ids"] == [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS
    assert len(prompt["prompt_token_ids"]) == 9
    duplex = prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["audio_tokens_per_unit"] == 6
    assert duplex["input_samples_per_unit"] == 7680
    assert duplex["scheduler_token_budget"] == 9


def _window(*body: int) -> list[int]:
    return [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS + [*body, DUPLEX_TOKENS.unit_end]


def test_history_trim_selects_closed_turn_and_aligned_prefix() -> None:
    tokens = []
    for _ in range(4):
        tokens.extend(_window())
    tokens.extend(_window(DUPLEX_TOKENS.speak, *([100] * 8), DUPLEX_TOKENS.turn_eos))
    tokens.extend(_window())
    tokens.extend(_window())

    plan = plan_history_trim(
        tokens,
        prefix_tokens=0,
        block_size=10,
        required_tokens=40,
        target_tokens=70,
    )

    assert plan is not None
    assert (plan.removed_tokens, plan.removed_windows) == (70, 6)


def test_history_trim_does_not_cut_open_turn() -> None:
    tokens = []
    for _ in range(4):
        tokens.extend(_window())
    tokens.extend(_window(DUPLEX_TOKENS.speak, *([100] * 8)))

    plan = plan_history_trim(
        tokens,
        prefix_tokens=0,
        block_size=10,
        required_tokens=45,
        target_tokens=45,
    )

    assert plan is None


def test_thinker2talker_applies_cumulative_trim_to_tokens_rows_and_prompt() -> None:
    prefix_len = 9
    first = [DUPLEX_TOKENS.speak, 100, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end]
    current = [DUPLEX_TOKENS.speak, 101, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end]
    rows = torch.arange(21 * 2, dtype=torch.float32).reshape(21, 2)
    multimodal = {
        "duplex_trimmed_tokens": torch.tensor([13]),
        "duplex_trimmed_windows": torch.tensor([1]),
        "hidden_states": {"layers": {0: rows.clone(), 24: rows.clone()}},
        "embed": {
            "tts_bos": torch.ones((1, 2)),
            "tts_eos": torch.ones((1, 2)),
            "tts_pad": torch.ones((1, 2)),
        },
    }
    output = SimpleNamespace(
        outputs=[SimpleNamespace(cumulative_token_ids=first + current, multimodal_output=multimodal)],
        multimodal_output=multimodal,
        prompt_token_ids=list(range(28)),
    )
    context = SimpleNamespace(
        bridge_states={"duplex": {"runtime_config": {"system_prefix_token_ids": list(range(prefix_len))}}}
    )

    result = thinker2talker([output], _streaming_context=context)[0]

    assert output.prompt_token_ids == list(range(9)) + list(range(19, 28))
    assert output.outputs[0].cumulative_token_ids == first + current
    assert result["model_intermediate_buffer"]["ids"]["prompt"] == list(range(9)) + list(range(19, 28))
    assert result["model_intermediate_buffer"]["hidden_states"]["output"] == rows[:18].tolist()


def test_thinker2talker_accepts_owner_compacted_windows_and_prompt() -> None:
    window = [DUPLEX_TOKENS.speak, 101, DUPLEX_TOKENS.turn_eos, DUPLEX_TOKENS.unit_end]
    rows = torch.arange(21 * 2, dtype=torch.float32).reshape(21, 2)
    multimodal = {
        "duplex_trimmed_windows": torch.tensor([1]),
        "duplex_history_window_base": torch.tensor([1]),
        "hidden_states": {"layers": {0: rows.clone(), 24: rows.clone()}},
        "embed": {"tts_bos": rows[:1], "tts_eos": rows[:1], "tts_pad": rows[:1]},
    }
    output = SimpleNamespace(
        outputs=[SimpleNamespace(cumulative_token_ids=window, multimodal_output=multimodal)],
        multimodal_output=multimodal,
        prompt_token_ids=list(range(18)),
    )
    context = SimpleNamespace(bridge_states={"duplex": {"runtime_config": {"system_prefix_token_ids": list(range(9))}}})

    result = thinker2talker([output], _streaming_context=context)[0]

    assert result["model_intermediate_buffer"]["ids"]["prompt"] == list(range(18))
    assert result["model_intermediate_buffer"]["hidden_states"]["output"] == rows[:18].tolist()


@pytest.mark.parametrize("remaining_capacity", [1, AUDIO_WINDOW_PROMPT_TOKENS + 31])
def test_history_trim_detaches_only_its_own_kv_pages_until_worker_ack(remaining_capacity: int) -> None:
    prefix = [42] * 9
    prompt = prefix + sum((_window() for _ in range(120)), [])
    config = KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer.0"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
            )
        ],
    )
    manager = KVCacheManager(config, max_model_len=2048, scheduler_block_size=16, hash_block_size=16, enable_caching=False)
    requests = [Request(f"duplex-trim-{i}", prompt.copy(), SamplingParams(max_tokens=32), None) for i in range(2)]
    for request in requests:
        assert manager.allocate_slots(request, num_new_tokens=len(prompt)) is not None
        request.num_computed_tokens = len(prompt)
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.kv_cache_manager = manager
    scheduler._duplex_history_trim_pending = {}
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen3OmniDuplexForConditionalGeneration"],
            max_model_len=len(prompt) + remaining_capacity,
            enforce_eager=True,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    other_before = manager.get_block_ids(requests[1].request_id)[0]
    update = SimpleNamespace(
        prompt_token_ids=[DUPLEX_TOKENS.unit_start] * 9,
        model_intermediate_buffer={
            "duplex": {"data_plane": True, "runtime_config": {"system_prefix_token_ids": prefix}}
        },
    )

    trim = scheduler._trim_qwen3_omni_duplex_history(requests[0], update)

    assert trim is not None
    held = trim["held_blocks"]
    assert held and all(block.ref_cnt == 1 for block in held)
    assert manager.get_block_ids(requests[1].request_id)[0] == other_before
    assert set(manager.get_block_ids(requests[0].request_id)[0]).isdisjoint(block.block_id for block in held)
    assert requests[0]._duplex_trimmed_windows > 0
    assert requests[0]._duplex_trimmed_tokens % 16 == 0
    assert (
        requests[0].num_computed_tokens + len(update.prompt_token_ids) + requests[0].sampling_params.max_tokens
        <= scheduler.vllm_config.model_config.max_model_len
    )
    scheduler._duplex_history_trim_pending[requests[0].request_id] = (held, requests[0].num_computed_tokens, 0)
    assert all(block.ref_cnt == 1 for block in held)
    manager.block_pool.free_blocks(reversed(held))
    assert all(block.ref_cnt == 0 for block in held)
    for request in requests:
        manager.free(request)


def test_worker_history_trim_copies_prefix_page_tail_and_restores_absolute_positions() -> None:
    prefix_tokens = 9
    cache = torch.arange(5 * 2 * 16 * 3).reshape(5, 2, 16, 3).clone()
    original_prefix = cache[0, :, :prefix_tokens, :].clone()
    original_tail = cache[2, :, prefix_tokens:, :].clone()
    req_state = SimpleNamespace(
        num_prompt_tokens=26,
        mrope_positions=torch.arange(26).repeat(3, 1),
        mrope_position_delta=0,
    )
    trim = {
        "source_block_id": 2,
        "prefix_block_id": 0,
        "prefix_tokens": prefix_tokens,
        "block_size": 16,
        "trimmed_tokens": 352,
    }
    runner = SimpleNamespace(requests={"req": req_state}, kv_caches=[cache])
    scheduled = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req", model_intermediate_buffer={"duplex": {"history_trim": trim}})]
    )

    Qwen3OmniDuplexForConditionalGeneration.apply_duplex_kv_reanchor(
        SimpleNamespace(model_stage="thinker"), runner, scheduled
    )

    assert torch.equal(cache[0, :, :prefix_tokens, :], original_prefix)
    assert torch.equal(cache[0, :, prefix_tokens:, :], original_tail)
    assert req_state.mrope_positions[:, :prefix_tokens].tolist() == [list(range(prefix_tokens))] * 3
    assert req_state.mrope_positions[:, prefix_tokens:].tolist() == [list(range(prefix_tokens + 352, 26 + 352))] * 3
    assert req_state.mrope_position_delta == 352
    assert "source_block_id" not in trim


def test_audio_window_rejects_missing_audio() -> None:
    with pytest.raises(ValueError, match="audio payload"):
        build_audio_window_prompt(
            request_id="request",
            fence=DuplexFence(session_id="session", epoch=0, turn_id=1),
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload={},
            final=False,
        )
