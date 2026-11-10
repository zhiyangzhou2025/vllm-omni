# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable, Iterator

import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni_duplex.qwen3_omni_duplex import (
    Qwen3OmniDuplexForConditionalGeneration,
)

# This checkpoint stores the Talker's code predictor in two shard files, with
# layer 3 cut between its q/k/v shards and its gate/up shards. Weights arrive
# one shard file at a time rather than in the index's order, so the subtree
# reaches the model in two runs with the rest of the Talker in between.
SPLIT_SUBTREE_STREAM = (
    "talker.code_predictor.model.layers.3.self_attn.q_proj.weight",
    "talker.codec_head.weight",
    "talker.code_predictor.model.layers.3.mlp.gate_proj.weight",
)


class _StageRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.names = [name for name, _ in weights]
        return set(self.names)


def _talker_stage(monkeypatch) -> Qwen3OmniDuplexForConditionalGeneration:
    model = object.__new__(Qwen3OmniDuplexForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "talker"
    model.thinker = None
    model.talker = _StageRecorder()
    model.code2wav = None
    monkeypatch.setattr(model, "_init_special_tokens_embeddings", lambda: set())
    return model


def _stream(names: Iterable[str]) -> Iterator[tuple[str, torch.Tensor]]:
    for name in names:
        yield name, torch.empty(0)


def test_the_code_predictor_subtree_reaches_the_talker_as_one_run(monkeypatch):
    model = _talker_stage(monkeypatch)

    model.load_weights(_stream(SPLIT_SUBTREE_STREAM))

    # Its loader buffers the q/k/v and gate/up shards for the length of a single
    # call and fuses them there, so a split subtree would reach it as two calls
    # holding one shard each and leave layer 3's gate_up_proj unwritten.
    assert model.talker.names == [
        "talker.codec_head.weight",
        "talker.code_predictor.model.layers.3.self_attn.q_proj.weight",
        "talker.code_predictor.model.layers.3.mlp.gate_proj.weight",
    ]


def test_everything_outside_the_code_predictor_keeps_its_arrival_order(monkeypatch):
    # Every other Talker weight is written straight into its parameter when it
    # arrives, so reordering them would buy nothing and only delay the load.
    names = ["talker.model.layers.0.mlp.gate_proj.weight", "talker.codec_head.weight"]
    model = _talker_stage(monkeypatch)

    model.load_weights(_stream(names))

    assert model.talker.names == names
