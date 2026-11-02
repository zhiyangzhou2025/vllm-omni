# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import torch

from vllm_omni.model_executor.models.qwen3_omni_duplex.qwen3_omni_duplex import (
    Qwen3OmniDuplexForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS


class _Thinker:
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).reshape(-1, 1)

    def _process_audio_input(self, audio_input: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        features = audio_input["input_features"]
        assert tuple(features.shape) == (128, 48)
        assert int(audio_input["audio_feature_lengths"]) == 48
        return (torch.arange(6, dtype=torch.float32).reshape(6, 1),)


class _Stage0(Qwen3OmniDuplexForConditionalGeneration):
    def __init__(self) -> None:
        self.model_stage = "thinker"
        self.thinker = _Thinker()
        self.calls = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.full((input_ids.shape[0], 1), -1.0)

    def _duplex_feature_extractor(self):
        def extract(samples, sampling_rate, return_tensors, return_attention_mask):
            assert samples.shape == (7680,) and sampling_rate == 16000
            assert return_tensors == "pt" and return_attention_mask is True
            self.calls += 1
            return type(
                "Features",
                (),
                {
                    "input_features": torch.zeros(1, 128, 3000),
                    "attention_mask": torch.cat([torch.ones(1, 48), torch.zeros(1, 2952)], dim=1),
                },
            )()

        return extract


def test_stage0_prefill_replaces_exact_audio_window() -> None:
    stage = _Stage0()
    payload = {"audio": np.zeros(7680, dtype=np.float32).tobytes(), "sample_rate_hz": 16000}
    info = {
        "duplex": {"data_plane": True, "model_type": "qwen3_omni_duplex", "payload": payload},
        "_omni_prompt_len": 9,
        "_omni_num_computed_tokens": 0,
    }

    token_ids, embeddings, updates = stage.thinker_preprocess(torch.ones(9, dtype=torch.long), None, **info)

    assert token_ids[0].item() == DUPLEX_TOKENS.unit_start
    assert token_ids[-1].item() == 151670
    assert embeddings.shape == (9, 1)
    assert embeddings[2:8].reshape(-1).tolist() == list(range(6))
    assert updates == {}
    assert stage.calls == 1


def test_stage0_decode_keeps_sampled_token_embedding() -> None:
    stage = _Stage0()
    info = {
        "duplex": {"data_plane": True, "model_type": "qwen3_omni_duplex", "payload": {}},
        "_omni_prompt_len": 9,
        "_omni_num_computed_tokens": 9,
    }

    token_ids, embeddings, updates = stage.thinker_preprocess(torch.tensor([DUPLEX_TOKENS.listen]), None, **info)

    assert token_ids.tolist() == [DUPLEX_TOKENS.listen]
    assert embeddings.tolist() == [[-1.0]]
    assert updates == {}
    assert stage.calls == 0


def test_prefill_keeps_prefix_tokens_and_replaces_only_the_window() -> None:
    stage = _Stage0()
    payload = {"audio": np.zeros(7680, dtype=np.float32).tobytes(), "sample_rate_hz": 16000}
    prefix = [11, 22, 33]
    info = {
        "duplex": {"data_plane": True, "model_type": "qwen3_omni_duplex", "payload": payload},
        "_omni_prompt_len": 12,
        "_omni_num_computed_tokens": 0,
    }

    token_ids, embeddings, updates = stage.thinker_preprocess(torch.tensor([*prefix, *([1] * 9)]), None, **info)

    assert token_ids[:3].tolist() == prefix
    assert embeddings[:3].reshape(-1).tolist() == [-1.0, -1.0, -1.0]
    assert token_ids[3].item() == DUPLEX_TOKENS.unit_start
    assert token_ids[-1].item() == 151670
    assert embeddings[5:11].reshape(-1).tolist() == list(range(6))
    assert updates == {}
