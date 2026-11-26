# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 append planning for Qwen3-Omni-Duplex."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.intermediate import build_duplex_append_prompt
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS

INPUT_SAMPLES_PER_UNIT = 7680
INPUT_SAMPLE_RATE_HZ = 16000
# One 0.48-second window: ``INPUT_SAMPLES_PER_UNIT / INPUT_SAMPLE_RATE_HZ``. It
# is also the clock the user-side event timestamps are counted in.
UNIT_DURATION_MS = 480
AUDIO_TOKENS_PER_UNIT = 6
AUDIO_BOUNDARY_TOKENS = 2
UNIT_START_TOKENS = 1
AUDIO_WINDOW_PROMPT_TOKENS = UNIT_START_TOKENS + AUDIO_BOUNDARY_TOKENS + AUDIO_TOKENS_PER_UNIT
# A System 2 result enters the context as ``<|im_start|>tool\n{payload}<|im_end|>\n``
# between the previous window's ``<|unit_end|>`` and the next window, as in the
# reference runtime. IDs verified in the user_signal checkpoints' tokenizer.json.
TOOL_BLOCK_START = 151644  # <|im_start|>
TOOL_BLOCK_END = 151645  # <|im_end|>
TOOL_BLOCK_NEWLINE = 198  # "\n"


def tool_block_length(token_ids: Sequence[int], start: int) -> int:
    """Length of the ToolBlock opening at ``start``, or 0 when none opens there."""
    if start >= len(token_ids) or token_ids[start] != TOOL_BLOCK_START:
        return 0
    for index in range(start + 1, len(token_ids)):
        if token_ids[index] == TOOL_BLOCK_END:
            if index + 1 >= len(token_ids) or token_ids[index + 1] != TOOL_BLOCK_NEWLINE:
                raise ValueError("Malformed Qwen3-Omni-Duplex tool block")
            return index + 2 - start
    raise ValueError("Unterminated Qwen3-Omni-Duplex tool block")


def window_prompt_layout(prompt_ids: Sequence[int], *, prefix_tokens: int, windows: int) -> list[tuple[int, int]]:
    """``(closure, tool)`` lengths of the first ``windows`` window prompts after the prefix.

    A window's prompt is ``[<|unit_end|>] [ToolBlock] <|unit_start|> x 9``. The
    first window after the prefix has no ``<|unit_end|>``: either it is the
    session's first window or a trim removed that token with the window before.
    """
    layout: list[tuple[int, int]] = []
    cursor = prefix_tokens
    for index in range(windows):
        closure = 1 if index else 0
        tool = tool_block_length(prompt_ids, cursor + closure)
        layout.append((closure, tool))
        cursor += closure + tool + AUDIO_WINDOW_PROMPT_TOKENS
    return layout


def trimmed_prompt_tokens(prompt_ids: Sequence[int], *, prefix_tokens: int, windows: int) -> int:
    """Prompt tokens a trim of ``windows`` windows removes after the prefix.

    That is every trimmed window's prompt plus the ``<|unit_end|>`` that opens
    the next window's prompt, since it closed the last trimmed window.
    """
    layout = window_prompt_layout(prompt_ids, prefix_tokens=prefix_tokens, windows=windows)
    return sum(closure + tool + AUDIO_WINDOW_PROMPT_TOKENS for closure, tool in layout) + 1


@dataclass(frozen=True)
class HistoryTrimPlan:
    prefix_tokens: int
    removed_tokens: int
    removed_windows: int


def plan_history_trim(
    token_ids: Sequence[int],
    *,
    prefix_tokens: int,
    block_size: int,
    required_tokens: int,
    target_tokens: int,
    max_removed_tokens: int | None = None,
) -> HistoryTrimPlan | None:
    """Select a page-aligned prefix of complete, closed speaking-turn groups."""
    if prefix_tokens < 0 or prefix_tokens > len(token_ids) or block_size <= 0:
        raise ValueError("Invalid Qwen3-Omni-Duplex history prefix or block size")
    if required_tokens <= 0:
        return None
    cursor = prefix_tokens
    windows = 0
    speaking = False
    candidates: list[HistoryTrimPlan] = []
    while cursor < len(token_ids):
        # A tool result injected ahead of the window belongs to it.
        cursor += tool_block_length(token_ids, cursor)
        end_prompt = cursor + AUDIO_WINDOW_PROMPT_TOKENS
        if list(token_ids[cursor:end_prompt]) != [DUPLEX_TOKENS.unit_start] * AUDIO_WINDOW_PROMPT_TOKENS:
            raise ValueError("Malformed Qwen3-Omni-Duplex history window prompt")
        cursor = end_prompt
        while cursor < len(token_ids) and token_ids[cursor] != DUPLEX_TOKENS.unit_end:
            if token_ids[cursor] == DUPLEX_TOKENS.speak:
                speaking = True
            elif token_ids[cursor] == DUPLEX_TOKENS.turn_eos:
                if not speaking:
                    raise ValueError("Qwen3-Omni-Duplex history closes an unopened speaking turn")
                speaking = False
            cursor += 1
        if cursor == len(token_ids):
            break
        cursor += 1
        windows += 1
        removed = cursor - prefix_tokens
        if (
            not speaking
            and removed % block_size == 0
            and removed >= required_tokens
            and (max_removed_tokens is None or removed <= max_removed_tokens)
            and cursor < len(token_ids)
        ):
            candidates.append(HistoryTrimPlan(prefix_tokens, removed, windows))
    return min(candidates, key=lambda plan: abs(plan.removed_tokens - target_tokens), default=None)


def build_audio_window_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: Mapping[str, object],
    runtime_config: Mapping[str, object],
    seq: int,
    turn_seq: int,
    payload: object,
    final: bool,
    prefix_token_ids: tuple[int, ...] = (),
) -> dict[str, object]:
    """Reserve one exact 0.48-second audio window in the Stage 0 scheduler.

    The worker replaces the trailing placeholder IDs with ``<|unit_start|>``,
    audio-boundary embeddings, and the six audio-tower embeddings. Tokens
    before them keep their own embeddings: the system prefix on the first
    append, and the previous window's ``<|unit_end|>`` after that. The window
    count must stay exact because surplus scheduler slots become KV entries.
    """

    if not isinstance(payload, Mapping) or payload.get("audio") is None:
        raise ValueError("Qwen3-Omni-Duplex Stage 0 requires one audio payload")
    token_id = DUPLEX_TOKENS.unit_start
    # Every window but the first closes the previous one. Every window stops on
    # <|unit_end|>, and the scheduler drops that sampled token, so one slot
    # holding it is all the previous window leaves behind.
    closure = () if seq <= 1 else (DUPLEX_TOKENS.unit_end,)
    prompt_token_ids = [*prefix_token_ids, *closure, *([token_id] * AUDIO_WINDOW_PROMPT_TOKENS)]
    return build_duplex_append_prompt(
        request_id=request_id,
        fence=fence,
        session_config=session_config,
        runtime_config=runtime_config,
        seq=seq,
        turn_seq=turn_seq,
        payload=payload,
        final=final,
        prompt_token_ids=prompt_token_ids,
        model_fields={
            "model_type": "qwen3_omni_duplex",
            "scheduler_token_id": token_id,
            "audio_tokens_per_unit": AUDIO_TOKENS_PER_UNIT,
            "input_samples_per_unit": INPUT_SAMPLES_PER_UNIT,
        },
    )
