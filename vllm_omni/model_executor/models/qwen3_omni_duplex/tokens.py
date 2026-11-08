# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint-backed Qwen3-Omni-Duplex protocol tokens."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DuplexTokenIDs:
    """IDs verified in save-chkpt-2325-merged ``tokenizer.json``."""

    unit_start: int = 151676
    unit_end: int = 151677
    listen: int = 151678
    speak: int = 151679
    # Retired by the protocol but kept in the vocabulary so the IDs after it stay
    # stable. It never appears in a sequence, and membership in DUPLEX_TOKEN_IDS
    # is what masks it in every phase.
    chunk_eos: int = 151680
    turn_eos: int = 151681
    asr: int = 151682
    asr_eos: int = 151683
    launch: int = 151684
    cancel: int = 151685
    idle: int = 151686
    non_idle: int = 151687

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("duplex token IDs must be integers")
        if len(set(values)) != len(values):
            raise ValueError("duplex token IDs must be distinct")


DUPLEX_TOKENS = DuplexTokenIDs()
DUPLEX_TOKEN_IDS = frozenset(getattr(DUPLEX_TOKENS, name) for name in DUPLEX_TOKENS.__dataclass_fields__)
