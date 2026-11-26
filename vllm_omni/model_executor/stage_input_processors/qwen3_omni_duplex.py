# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processors for the three-stage Qwen3-Omni-Duplex deployment.

Thinker -> Talker (``thinker2talker``)
    The duplex plugin forwards every speak window the Thinker closes. Within a
    turn the Talker is one autoregressive decode, as in the reference runtime:

    * a turn's first window replaces the Talker's prompt with the session's
      whole context so far plus the six bootstrap rows;
    * every later window of the turn adds no context row. It hands the Talker
      only the window's text, which the Talker appends to its text queue, and
      one placeholder token the Talker decodes as an ordinary step.

    The Thinker forwards its hidden states with ``engine_output_type: latent``.
    Their rows are ``prefix + sum_w [unit_end (w > 0)] + window_w + gen_w[:-1]``:
    every prompt row and every generated row but the window's last, the
    ``<|unit_end|>`` that stopped it. The prompt rows are the Talker's context;
    the open window's generated rows carry its text.

Talker -> Code2Wav (``talker2code2wav_async_chunk``)
    The shared streamed edge restarts its frame cache with every segment, and a
    segment is one window, so each window would reach Code2Wav with no left
    context. This wrapper keeps the turn's frames and prepends up to
    ``codec_left_context_frames`` of them.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.talker_row import BOOTSTRAP_ROWS
from vllm_omni.model_executor.models.qwen3_omni_duplex.stage0 import (
    AUDIO_WINDOW_PROMPT_TOKENS,
    trimmed_prompt_tokens,
    window_prompt_layout,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKENS
from vllm_omni.model_executor.stage_input_processors import qwen3_omni as _qwen3_omni

# Keys the Talker reads off ``model_intermediate_buffer``. ``TURN_START`` and
# ``WINDOW`` travel on every window; the Talker records the last window whose
# text it queued under ``APPENDED``.
TURN_START_KEY = "duplex_talker_turn_start"
WINDOW_KEY = "duplex_talker_window"
APPENDED_KEY = "duplex_talker_appended"
APPEND_KEY = "duplex_talker_append"
# Tensors cross the stage boundary as nested lists; this maps ``"top.key"`` to
# the dtype the Talker rebuilds them with.
TRANSPORT_DTYPES_KEY = "duplex_transport_dtypes"

# The Thinker layer whose input embeddings feed the Talker's text rows, and the
# one whose output projects the acoustic rows (``accept_hidden_layer``).
_EMBED_LAYER = 0
_HIDDEN_LAYER = 24
_CODEC_GROUPS = 16
_BRIDGE_KEY = "qwen3_omni_duplex_talker"
logger = init_logger(__name__)


def _windows(token_ids: list[int]) -> list[list[int]]:
    """Split the session's generated tokens into closed windows."""
    windows: list[list[int]] = []
    current: list[int] = []
    for token_id in token_ids:
        current.append(int(token_id))
        if token_id == DUPLEX_TOKENS.unit_end:
            windows.append(current)
            current = []
    if current:
        raise ValueError(f"Qwen3-Omni-Duplex forwarded a window that is still open: {current}")
    return windows


def _row_index(
    prefix_len: int, windows: list[list[int]], layout: list[tuple[int, int]]
) -> tuple[list[int], list[int], list[int], int]:
    """Context rows, their prompt positions, the last window's generated rows, and the row count.

    ``layout`` holds each window prompt's ``(closure, tool)`` lengths. A
    ToolBlock is Thinker context but not Talker context: ``audio_row_mask``
    reads every window as the nine rows ahead of its ``<|unit_end|>``.
    """
    context: list[int] = list(range(prefix_len))
    positions: list[int] = list(range(prefix_len))
    generated_last: list[int] = []
    row = position = prefix_len
    for index, (window, (closure, tool)) in enumerate(zip(windows, layout, strict=True)):
        # Every append after the first reserves the previous window's <|unit_end|>.
        context.extend(range(row, row + closure))
        positions.extend(range(position, position + closure))
        row += closure + tool
        position += closure + tool
        context.extend(range(row, row + AUDIO_WINDOW_PROMPT_TOKENS))
        positions.extend(range(position, position + AUDIO_WINDOW_PROMPT_TOKENS))
        row += AUDIO_WINDOW_PROMPT_TOKENS
        position += AUDIO_WINDOW_PROMPT_TOKENS
        n_generated = len(window) - 1
        if index == len(windows) - 1:
            generated_last = list(range(row, row + n_generated))
        row += n_generated
    return context, positions, generated_last, row


def _duplex_payload(streaming_context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The Talker's ``duplex`` payload and the session's bridge state."""
    bridge = getattr(streaming_context, "bridge_states", None)
    duplex = bridge.get("duplex") if isinstance(bridge, dict) else None
    if not isinstance(duplex, dict):
        raise ValueError("Qwen3-Omni-Duplex Talker input needs the session's duplex bridge state")
    payload: dict[str, Any] = {"data_plane": True}
    for key in ("session_id", "epoch"):
        if duplex.get(key) is not None:
            payload[key] = duplex[key]
    payload["turn_id"] = duplex.get("model_turn_id", duplex.get("turn_id"))
    for key in ("session_config", "runtime_config"):
        if isinstance(duplex.get(key), dict):
            payload[key] = dict(duplex[key])
    return payload, bridge


def _to_transport(buffer: dict[str, Any]) -> dict[str, Any]:
    """Replace every tensor under ``embed`` / ``hidden_states`` with a list."""
    dtypes: dict[str, str] = {}
    for top in ("embed", "hidden_states"):
        for key, value in buffer.get(top, {}).items():
            dtypes[f"{top}.{key}"] = str(value.dtype).removeprefix("torch.")
            buffer[top][key] = value.tolist()
    buffer[TRANSPORT_DTYPES_KEY] = dtypes
    return buffer


def thinker2talker(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    _streaming_context: Any = None,
) -> list[OmniTokensPrompt]:
    """Hand one closed speak window to the Talker."""
    del prompt, requires_multimodal_data
    results: list[OmniTokensPrompt] = []
    for output in source_outputs:
        completion = next(iter(getattr(output, "outputs", ()) or ()), None)
        multimodal = getattr(output, "multimodal_output", None)
        if multimodal is None and completion is not None:
            multimodal = getattr(completion, "multimodal_output", None)
        token_ids = list(getattr(completion, "cumulative_token_ids", ()) or ())
        if token_ids and token_ids[-1] != DUPLEX_TOKENS.unit_end:
            logger.warning(
                "Duplex forwarded open window req=%s output_finished=%s finish_reason=%s stop_reason=%s "
                "tail=%s prompt_len=%s",
                getattr(output, "request_id", None),
                getattr(output, "finished", None),
                getattr(completion, "finish_reason", None),
                getattr(completion, "stop_reason", None),
                token_ids[-16:],
                len(getattr(output, "prompt_token_ids", ()) or ()),
            )
        all_windows = _windows(token_ids)
        trimmed_marker = multimodal.get("duplex_trimmed_windows", torch.tensor([0]))
        base_marker = multimodal.get("duplex_history_window_base", torch.tensor([0]))
        trimmed_windows = int(trimmed_marker.reshape(-1)[-1].item()) - int(base_marker.reshape(-1)[-1].item())
        if trimmed_windows < 0 or trimmed_windows >= len(all_windows):
            raise ValueError("Qwen3-Omni-Duplex trim must retain the current window")
        windows = all_windows[trimmed_windows:]
        window = windows[-1]
        # Only speak windows are forwarded; the first <|speak|> is the control
        # token, and the user_signal prefix ahead of it is not for the Talker.
        speak_at = window.index(DUPLEX_TOKENS.speak)
        duplex, bridge = _duplex_payload(_streaming_context)
        state = bridge.setdefault(_BRIDGE_KEY, {"window": -1, "turn_open": False})
        turn_start = not state["turn_open"]
        final_window = DUPLEX_TOKENS.turn_eos in window

        layers = multimodal["hidden_states"]["layers"]
        embed_rows, hidden_rows = layers[_EMBED_LAYER], layers[_HIDDEN_LAYER]
        prefix_len = len(duplex["runtime_config"]["system_prefix_token_ids"])
        # The session's next append can reach the Thinker before this window is
        # forwarded, so the prompt may already hold rows the hidden states do not
        # cover. Those belong to the next window.
        prompt_source = list(getattr(output, "prompt_token_ids", None) or ())
        if trimmed_windows:
            cut = trimmed_prompt_tokens(prompt_source, prefix_tokens=prefix_len, windows=trimmed_windows)
            prompt_source = prompt_source[:prefix_len] + prompt_source[prefix_len + cut :]
            output.prompt_token_ids = prompt_source
        layout = window_prompt_layout(prompt_source, prefix_tokens=prefix_len, windows=len(windows))
        context_idx, prompt_positions, generated_idx, total = _row_index(prefix_len, windows, layout)
        if not total == int(embed_rows.shape[0]) == int(hidden_rows.shape[0]):
            raise ValueError(
                f"Qwen3-Omni-Duplex Thinker forwarded {int(embed_rows.shape[0])} hidden rows, "
                f"the window layout accounts for {total}"
            )
        if prompt_positions and prompt_positions[-1] >= len(prompt_source):
            raise ValueError(
                f"Qwen3-Omni-Duplex Thinker prompt has {len(prompt_source)} rows, "
                f"the window layout needs {prompt_positions[-1] + 1}"
            )
        prompt_ids = [prompt_source[position] for position in prompt_positions]
        generated_ids = window[speak_at:-1]
        generated_t = torch.tensor(generated_idx[speak_at:], dtype=torch.long)
        window_seq = int(state["window"]) + 1
        state["window"] = window_seq
        state["turn_open"] = not final_window
        meta: dict[str, Any] = {TURN_START_KEY: turn_start, WINDOW_KEY: window_seq}
        duplex = {**duplex, "payload": {"final_window": final_window}}
        if turn_start:
            context_t = torch.tensor(context_idx, dtype=torch.long)
            embed = multimodal["embed"]
            buffer: dict[str, Any] = {
                "embed": {
                    "prefill": torch.cat(
                        [embed_rows.index_select(0, context_t), embed_rows.index_select(0, generated_t)], 0
                    ).cpu(),
                    "tts_bos": embed["tts_bos"][-1:].cpu(),
                    "tts_eos": embed["tts_eos"][-1:].cpu(),
                    "tts_pad": embed["tts_pad"][-1:].cpu(),
                },
                "hidden_states": {"output": hidden_rows.index_select(0, context_t).cpu()},
                "ids": {"all": prompt_ids + generated_ids, "prompt": prompt_ids},
                "duplex": duplex,
                # A new turn starts the Talker over from this context.
                "meta": {**meta, "replace_streaming_prompt": True},
            }
            talker_prompt = [0] * (len(prompt_ids) + BOOTSTRAP_ROWS)
        else:
            buffer = {
                "hidden_states": {APPEND_KEY: embed_rows.index_select(0, generated_t).cpu()},
                "ids": {APPEND_KEY: generated_ids},
                "duplex": duplex,
                "meta": meta,
            }
            # The Talker decodes this placeholder from its own last sampled codec.
            talker_prompt = [0]
        results.append(
            OmniTokensPrompt(prompt_token_ids=talker_prompt, model_intermediate_buffer=_to_transport(buffer))
        )
    return results


# Frames already sent to Code2Wav for the current turn, per request. The
# connector snapshots the request when it enqueues a chunk and processes the
# snapshot later, so the chunk's own meta still names the previous window and
# cannot tell where a turn starts. The Talker, which runs in the same process,
# marks the request instead when it prefills a turn's first window.
_TURN_FRAMES: dict[str, list[torch.Tensor]] = {}
_TURN_STARTED: set[str] = set()


def mark_turn_start(request_id: str) -> None:
    """Drop the request's frames before its next chunk: a new turn has begun."""
    _TURN_STARTED.add(request_id)


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> Any:
    """The shared streamed edge, with the turn's earlier frames as left context."""
    out = _qwen3_omni.talker2code2wav_async_chunk(transfer_manager, multimodal_output, request, is_finished)
    if out is None or getattr(out, "codes", None) is None or out.codes.audio is None:
        return out
    request_id = request.external_req_id
    if request_id in _TURN_STARTED:
        _TURN_STARTED.discard(request_id)
        _TURN_FRAMES.pop(request_id, None)
    history = _TURN_FRAMES.setdefault(request_id, [])
    flat = out.codes.audio
    frames = flat.reshape(_CODEC_GROUPS, int(flat.numel()) // _CODEC_GROUPS)
    segment_left = int(out.meta.left_context_size)
    new = frames[:, segment_left:]
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    max_left = int(cfg.get("codec_left_context_frames", 25))
    context = torch.cat(history, 1)[:, -max_left:] if history else new[:, :0]
    out.codes.audio = torch.cat([context.to(new.device, new.dtype), new], 1).reshape(-1)
    out.meta.left_context_size = int(context.shape[1])
    # Only the last max_left frames can ever be context again.
    _TURN_FRAMES[request_id] = [torch.cat([*history, new.detach().clone()], 1)[:, -max_left:]]
    return out
