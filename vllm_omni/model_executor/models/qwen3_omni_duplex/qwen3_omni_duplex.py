# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Window-protocol execution for Qwen3-Omni-Duplex.

The checkpoint's three stages share one model class, so this subclass serves
every stage that speaks the duplex window protocol: the Thinker writes the
window rows and constrains its own sampling to the protocol, and the Talker
reads those rows back and is held to one window's worth of codec frames.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

from vllm.config import CUDAGraphMode

from vllm_omni.platforms import current_omni_platform

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.artifact import read_user_signal
from vllm_omni.model_executor.models.qwen3_omni_duplex.audio_window import (
    AUDIO_WINDOW_PROMPT_TOKENS,
    audio_window_from_duplex,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.sampling import (
    Stage0Windows,
    ThinkerProtocolError,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.duplex.talker_row import (
    BOOTSTRAP_ROWS,
    audio_row_mask,
    frame_budget,
)
from vllm_omni.model_executor.models.qwen3_omni_duplex.tokens import DUPLEX_TOKEN_IDS

# Stands in for every text token when asking a window what its phase allows.
# Any id outside the protocol set works; the window never reads its value.
_TEXT_PROBE_TOKEN_ID = 0
# The Talker's code predictor packs the checkpoint's separate q/k/v and gate/up
# tensors into fused projections, buffering the shards for the length of one
# call. Its subtree under the checkpoint's own naming.
_CODE_PREDICTOR_PREFIX = "talker.code_predictor."

# The stage input processor ships the Thinker's rows as lists, and names the
# dtype each list rebuilds as under this key of the Talker's payload.
_TRANSPORT_DTYPES_KEY = "duplex_transport_dtypes"
# The processor numbers the windows of one turn, and records under this key the
# last one whose text the Talker already queued.
_WINDOW_KEY = "duplex_talker_window"
_APPENDED_KEY = "duplex_talker_appended"


class Qwen3OmniDuplexForConditionalGeneration(Qwen3OmniMoeForConditionalGeneration):
    """Qwen3-Omni runtime that holds both stages to the duplex window."""

    # The runner only calls prepare_duplex_sampling for models that opt in.
    prefer_model_sampler = True

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._wire_stage_hooks()
        # The frame-synchronous Talker (additional_config flag) keeps the codec
        # just sampled, and the summed embedding the code predictor returned for
        # it, so the next step can decode from them.
        self._duplex_talker_stash: dict[str, dict[str, torch.Tensor | int]] = {}

    def _frame_sync(self) -> bool:
        """Whether this Talker decodes each frame from the codec just sampled."""

        config = self.__dict__.get("vllm_config")
        additional = getattr(config, "additional_config", None)
        return bool(isinstance(additional, Mapping) and additional.get("qwen3_omni_duplex_frame_sync"))

    def _wire_stage_hooks(self) -> None:
        """Expose only the hooks that belong to the stage being served.

        One class serves every stage of the checkpoint, and the runner enables
        each optional path by testing whether its attribute is callable. A hook
        left visible on the wrong stage is therefore silently taken for that
        stage's implementation.
        """

        if self.model_stage == "thinker":
            # Reading the manifest here rejects a pre-v3 checkpoint at startup
            # rather than once per session.
            self._stage0_windows = Stage0Windows(read_user_signal(self.vllm_config.model_config.model))
            # The parent exposes talker_mtp for every stage. Stage 0 is the
            # thinker: it has no talker, and that path would demand mtp_inputs
            # the thinker preprocess never produces.
            self.talker_mtp = None
            self.has_preprocess = True
            self.set_custom_preprocess(self.thinker_preprocess)
            return
        # post_sample_talker_mtp below is the thinker's window bookkeeping, and
        # it answers with placeholder rows. On the talker stage the runner would
        # merge those over the codec codes talker_mtp just produced. The
        # frame-synchronous Talker is the exception: it has no lagged decode-time
        # code predictor, and its post-sample hook returns the step's codes.
        if self.model_stage == "talker" and self._frame_sync():
            self.talker_mtp = None
            self.post_sample_talker_mtp = self._frame_sync_codes
            return
        self.post_sample_talker_mtp = None

    def apply_duplex_kv_reanchor(self, runner: Any, scheduler_output: Any = None) -> None:
        """Copy the retained Stage 0 tail into its prefix page before forward."""
        if self.model_stage != "thinker" or scheduler_output is None:
            return
        for new_request in scheduler_output.scheduled_new_reqs:
            req_id = new_request.req_id
            model_buffer = getattr(new_request, "model_intermediate_buffer", None)
            duplex = model_buffer.get("duplex") if isinstance(model_buffer, dict) else None
            trim = duplex.get("history_trim") if isinstance(duplex, dict) else None
            if not isinstance(trim, dict):
                continue
            req_state = runner.requests[req_id]
            mrope_positions = getattr(req_state, "mrope_positions", None)
            if mrope_positions is None or mrope_positions.shape[1] != req_state.num_prompt_tokens:
                raise ValueError("Qwen3-Omni-Duplex history trim requires complete MRoPE prompt positions")
            prefix_tokens = int(trim["prefix_tokens"])
            cumulative_trim = int(trim["trimmed_tokens"])
            if req_state.mrope_position_delta != 0:
                raise ValueError("Qwen3-Omni-Duplex history trim expected freshly initialized MRoPE positions")
            if "source_block_id" in trim:
                block_size = int(trim["block_size"])
                if not runner.kv_caches or any(
                    cache.ndim != 4 or cache.shape[2] != block_size for cache in runner.kv_caches
                ):
                    raise ValueError("Qwen3-Omni-Duplex history trim requires uniform full-attention KV page layout")
                source = int(trim["source_block_id"])
                target = int(trim["prefix_block_id"])
                for cache in runner.kv_caches:
                    cache[target, :, prefix_tokens:block_size, :].copy_(cache[source, :, prefix_tokens:block_size, :])
                trim.pop("source_block_id")
                trim.pop("prefix_block_id")
            mrope_positions[:, prefix_tokens:] += cumulative_trim
            req_state.mrope_position_delta = cumulative_trim

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream the checkpoint with the code predictor's subtree kept whole.

        The code predictor's loader buffers the checkpoint's separate q/k/v and
        gate/up tensors and fuses them at the end of the call that delivered
        them, refusing to return while a fused projection is still incomplete.
        It therefore has to see its whole subtree in one call, and vLLM's weight
        loader gives a child one call per *adjacent* run of the stream while the
        stream itself arrives one shard file at a time rather than in the
        index's order. This checkpoint splits the subtree across two shard
        files, cutting layer 3 between its q/k/v and its gate/up shards, so the
        order it arrives in has to be repaired before the parent routes it.
        """
        loaded = super().load_weights(_code_predictor_last(weights))
        # The code predictor's first call compiles under the dtype the step
        # thread runs with, which is not the dtype the weights load under. Two
        # calls here pay that cost at startup instead of on the first frame.
        if self.model_stage == "talker" and self._frame_sync():
            device = next(self.parameters()).device
            hidden = torch.zeros(
                (1, self.talker_config.text_config.hidden_size),
                device=device,
                dtype=self.vllm_config.model_config.dtype,
            )
            token = torch.zeros((1,), device=device, dtype=torch.long)
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                with torch.inference_mode():
                    for _ in range(2):
                        self._run_code_predictor(token, hidden)
            finally:
                torch.set_default_dtype(previous_dtype)
        return loaded

    def thinker_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **info: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Replace one scheduled prefill span with the exact audio window."""

        duplex = info.get("duplex")
        if not isinstance(duplex, Mapping) or duplex.get("data_plane") is not True:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}

        prompt_len = _optional_int(info.get("_omni_prompt_len"), info.get("duplex_prompt_len"))
        token_offset = _optional_int(info.get("_omni_num_computed_tokens"), info.get("duplex_token_offset")) or 0
        span_length = int(input_ids.shape[0])
        # Decode steps carry one already-sampled token. Only a prefill span can
        # hold the audio window.
        if prompt_len is not None and token_offset >= prompt_len:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}
        window_start = _window_start(prompt_len, token_offset, span_length)
        if window_start is None:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}

        window = audio_window_from_duplex(
            duplex,
            embed_tokens=self._embed_boundary_tokens,
            extract_features=self._extract_audio_features,
            encode_audio=self._encode_audio_features,
            device=input_ids.device,
        )
        # Tokens before the window are the system prefix or the previous
        # window's <|unit_end|>. They keep their own embeddings.
        if input_embeds is not None and int(input_embeds.shape[0]) == span_length:
            embeds = input_embeds.clone()
        else:
            embeds = self.embed_input_ids(input_ids)
        ids = input_ids.clone()
        ids[window_start:] = window.input_ids
        embeds[window_start:] = window.embeddings.to(dtype=embeds.dtype)
        return ids, embeds, {}

    def _embed_boundary_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.thinker.embed_input_ids(token_ids)

    def _extract_audio_features(self, samples: Any, sample_rate_hz: int) -> tuple[torch.Tensor, int]:
        processor = self._duplex_feature_extractor()
        features = processor(
            samples,
            sampling_rate=sample_rate_hz,
            return_tensors="pt",
            return_attention_mask=True,
        )
        mask = features.attention_mask.bool()
        valid = features.input_features.permute(0, 2, 1)[mask].permute(1, 0)
        return valid, int(mask.sum())

    def _encode_audio_features(self, features: torch.Tensor, feature_length: int) -> torch.Tensor:
        device = features.device
        if device.type == "cpu" and hasattr(self.thinker, "audio_tower"):
            device = next(self.thinker.audio_tower.parameters()).device
        features = features.to(device=device, dtype=torch.get_default_dtype())
        lengths = torch.tensor([feature_length], dtype=torch.long, device=device)
        encoded = self.thinker._process_audio_input(
            {
                "input_features": features,
                "audio_feature_lengths": lengths,
            }
        )
        if len(encoded) != 1:
            raise ValueError("Qwen3-Omni-Duplex audio window must encode as one item")
        return encoded[0]

    def _thinker_to_talker_prefill(
        self,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        multimodal_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        thinker_result_ids: torch.Tensor,
        speaker_id: object,
        tts_bos_thinker: torch.Tensor | None = None,
        tts_eos_thinker: torch.Tensor | None = None,
        tts_pad_thinker: torch.Tensor | None = None,
        input_ids_cpu: torch.Tensor | None = None,
        thinker_result_ids_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Turn one window run into the Talker's prefill rows.

        The Thinker's prompt is the context, so the whole of it projects as one
        span. The parent instead splits the prompt on ``<|im_start|>``, drops the
        system segment and keeps only the last assistant one, because a turn-based
        prompt is a ChatML transcript. A window run is not: it is
        ``prefix, window_0, unit_end, window_1, ...`` with no role headers after
        the prefix, so a split on ``<|im_start|>`` would hand the Talker the
        prefix alone and throw every window away.

        Which rows carry acoustic content comes from the window layout rather
        than from ``audio_token_id``, which the parent keys on and a duplex
        window never contains.

        The Thinker's generated text becomes the turn's text steps: the first of
        them opens the bootstrap rows and the rest queue up for decode. Protocol
        tokens are dropped from that queue -- they pace the window, they are not
        speech. ``<|tts_eos|>`` is not appended here either: a non-final window
        would then end the turn's text early, so the final window's EOS is left
        to the decode path, which emits it once the queue runs dry.
        """

        del multimodal_mask  # The parent overwrites this parameter unconditionally.
        prompt_ids = (input_ids_cpu if input_ids_cpu is not None else input_ids).reshape(-1)
        all_ids = (thinker_result_ids_cpu if thinker_result_ids_cpu is not None else thinker_result_ids).reshape(-1)

        tts_bos_embed, _tts_eos_embed, tts_pad_embed = self._get_tts_embed(
            thinker_embed, tts_bos_thinker, tts_eos_thinker, tts_pad_thinker
        )
        context = self._get_talker_user_parts(
            0,
            int(prompt_ids.shape[0]),
            audio_row_mask(prompt_ids),
            thinker_hidden,
            thinker_embed,
        )
        # _get_talker_user_parts clamps the span to the rows it was actually
        # given, so its own answer is the one the ids have to match.
        context_rows = int(context.shape[0])
        text_queue = self._duplex_text_queue(all_ids[context_rows:], thinker_embed[context_rows:])
        bootstrap_ids, bootstrap = self._duplex_bootstrap(speaker_id, text_queue, tts_bos_embed, tts_pad_embed)

        talker_embeds = torch.cat([context.to(bootstrap.device, bootstrap.dtype), bootstrap], dim=0)
        talker_ids = torch.cat(
            [thinker_result_ids.reshape(-1)[:context_rows].to(bootstrap_ids.device), bootstrap_ids],
            dim=0,
        )
        # The bootstrap already spent the first entry; decode consumes the rest.
        trailing = text_queue[1:] if int(text_queue.shape[0]) > 1 else None
        return talker_ids, talker_embeds, trailing

    def _duplex_text_queue(self, generated_ids: torch.Tensor, generated_embed: torch.Tensor) -> torch.Tensor:
        """Project the Thinker's spoken rows, in order, into Talker text space."""

        ids = generated_ids.reshape(-1)
        rows = min(int(ids.shape[0]), int(generated_embed.shape[0]))
        ids = ids[:rows]
        protocol = torch.tensor(sorted(DUPLEX_TOKEN_IDS), dtype=ids.dtype, device=ids.device)
        spoken = torch.nonzero(~torch.isin(ids, protocol)).flatten()
        selected = generated_embed[:rows].index_select(0, spoken.to(generated_embed.device))
        return self.talker.text_projection(selected)

    def _duplex_bootstrap(
        self,
        speaker_id: object,
        text_queue: torch.Tensor,
        tts_bos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the rows a turn opens with: text channel plus codec channel.

        The parent spends nine rows here because it keeps the three
        ``<|im_start|>assistant\\n`` rows it split the prompt on. A window run has
        no such header, so only the six the codec channel needs are left.
        """

        codec_ids = torch.tensor(
            [
                self.talker_config.codec_nothink_id,
                self.talker_config.codec_think_bos_id,
                self.talker_config.codec_think_eos_id,
                speaker_id,
                self.talker_config.codec_pad_id,
                self.talker_config.codec_bos_id,
            ],
            dtype=torch.long,
            device=tts_pad_embed.device,
        )
        codec = self.talker.embed_input_ids(codec_ids).to(device=tts_pad_embed.device, dtype=tts_pad_embed.dtype)
        first_text = text_queue[:1] if int(text_queue.shape[0]) else torch.zeros_like(tts_bos_embed)
        text = torch.cat(
            [
                tts_pad_embed.expand(BOOTSTRAP_ROWS - 2, -1),
                tts_bos_embed,
                first_text.to(tts_bos_embed.device, tts_bos_embed.dtype),
            ],
            dim=0,
        )
        ids = torch.full(
            (BOOTSTRAP_ROWS,),
            fill_value=self.config.tts_pad_token_id,
            dtype=torch.long,
            device=codec.device,
        )
        return ids, text.to(codec.dtype) + codec

    def talker_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **info: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Rebuild the transported rows, then decode a frame from the last codec.

        The Thinker's rows cross the stage boundary as lists. They are rebuilt
        here, in place, because the runner keeps handing this payload back.

        With frame sync on, a turn's later window arrives as one placeholder
        token the scheduler treats as prefill. It is a decode step: the window
        adds no context, and its codec is the one the previous step sampled.
        That step's summed codec embedding, stashed by the post-sample hook,
        replaces the embedding the runner built for the placeholder.
        """

        dtypes = info.get(_TRANSPORT_DTYPES_KEY)
        if isinstance(dtypes, Mapping):
            for dotted, dtype_name in dtypes.items():
                top, key = str(dotted).split(".", 1)
                sub = info.get(top)
                if isinstance(sub, dict) and isinstance(sub.get(key), list):
                    sub[key] = torch.tensor(sub[key], dtype=getattr(torch, dtype_name))
        if self._frame_sync() and self._is_continuation_step(input_ids, info):
            info = {**info, "_omni_is_prefill": False}
        meta = info.get("meta") if isinstance(info.get("meta"), Mapping) else {}
        if info.get("_omni_is_prefill") and meta.get("duplex_talker_turn_start") is True:
            # A new turn: Code2Wav must not take the last turn's frames as context.
            from vllm_omni.model_executor.stage_input_processors.qwen3_omni_duplex import mark_turn_start

            mark_turn_start(str(info.get("request_id")))
        ids, embeds, update = super().talker_preprocess(input_ids, input_embeds, **info)
        if self._frame_sync() and not info.get("_omni_is_prefill"):
            request_id = info.get("request_id")
            stash = self._duplex_talker_stash.get(request_id)
            if stash is None:
                raise RuntimeError(f"Qwen3-Omni-Duplex Talker decoded before its first frame was sampled: {request_id}")
            _, text_step = update.pop("mtp_inputs")
            text_step = text_step.reshape(1, -1)
            summed = stash["summed"]
            embeds = (summed.to(text_step.device, text_step.dtype) + text_step).to(embeds.dtype)
            ids = torch.full_like(input_ids, int(stash["tok"]))
        return ids, embeds, update

    def _is_continuation_step(self, input_ids: torch.Tensor, info: Mapping[str, object]) -> bool:
        """A turn's later window: one placeholder token, and not the turn's first."""

        meta = info.get("meta") if isinstance(info.get("meta"), Mapping) else {}
        if meta.get("duplex_talker_turn_start") is True:
            return False
        span = int(input_ids.shape[0])
        computed = _optional_int(info.get("_omni_num_computed_tokens"), None)
        prompt_len = _optional_int(info.get("_omni_prompt_len"), None)
        return bool(info.get("_omni_is_prefill")) and span == 1 and computed == prompt_len - 1

    def _frame_sync_codes(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        req_ids: list[str],
        **_: object,
    ) -> torch.Tensor:
        """Run the code predictor on the codec token just sampled.

        The runner calls this after its sampler and passes the sampled token per
        request. The codes it returns are this step's frame; the summed codec
        embedding is what the next step decodes from.
        """

        tokens = input_ids.reshape(-1).to(torch.long)
        hidden = hidden_states.reshape(len(req_ids), -1)
        summed, codes = self._run_code_predictor(tokens, hidden)
        for index, request_id in enumerate(req_ids):
            self._duplex_talker_stash[request_id] = {
                "tok": int(tokens[index]),
                "summed": summed[index : index + 1].detach().clone(),
            }
        return codes.reshape(len(req_ids), -1)

    def _run_code_predictor(self, tokens: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(hidden)
        with current_omni_platform.set_forward_context(
            None, self.vllm_config, cudagraph_runtime_mode=CUDAGraphMode.NONE, batch_descriptor=None
        ):
            summed, codes = Qwen3OmniMoeForConditionalGeneration.talker_mtp(self, tokens, hidden, hidden, zeros)
        torch.cuda.synchronize()
        return summed, codes

    def talker_preprocess_decode(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        update_dict: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Take each decode step's text from the queue the prefill built.

        The parent switches on ``async_chunk``: with it set, it reads the text
        from the upstream Thinker's live decode stream instead of
        ``hidden_states.trailing_text``. In a deployment that flag also selects
        the streamed Talker -> Code2Wav edge, which this topology uses. The duplex
        Talker heads its topology, so no live stream ever arrives, and the
        parent's fallback would hand the code predictor one EOS and then pad,
        dropping the Thinker's text. The window's whole text is already in the
        prefill's queue (``_thinker_to_talker_prefill``), so the queue is read
        whatever the flag says.
        """

        device, dtype = input_embeds.device, input_embeds.dtype
        pad = self.tts_pad_embed.to(device, dtype=dtype)
        hs = payload.get("hidden_states", {})
        queue = hs.get("trailing_text")
        queue = queue.to(device, dtype=dtype) if isinstance(queue, torch.Tensor) else None
        queue = self._append_window_text(queue, update_dict, payload, device, dtype)
        if queue is not None and queue.numel() > 0:
            text_step = queue[0:1, :]
            # An exhausted queue is stored empty. A stored pad row would sit in
            # front of the text the turn's next window appends.
            update_dict.setdefault("hidden_states", {})["trailing_text"] = queue[1:, :].detach()
        else:
            text_step = pad

        last = hs.get("last")
        hidden_size = self.talker_config.text_config.hidden_size
        if last is None:
            last_talker_hidden = torch.zeros((1, hidden_size), device=device, dtype=dtype)
        else:
            last_talker_hidden = last.to(device, dtype=dtype).reshape(*last.shape[-2:])
        return last_talker_hidden, text_step, update_dict

    def _append_window_text(
        self,
        queue: torch.Tensor | None,
        update_dict: dict[str, Any],
        payload: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Queue one window's text the first time the Talker decodes it.

        A window's text arrives with the window and is appended once. The final
        window of a turn also ends the text with ``<|tts_eos|>``.
        """

        meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
        window = meta.get(_WINDOW_KEY)
        if window is None or window == meta.get(_APPENDED_KEY):
            return queue
        duplex = payload.get("duplex") if isinstance(payload.get("duplex"), Mapping) else {}
        final = bool((duplex.get("payload") or {}).get("final_window"))
        addition = None
        if meta.get("duplex_talker_turn_start") is not True:
            ids = torch.as_tensor(payload["ids"]["duplex_talker_append"], dtype=torch.long, device=device)
            rows = payload["hidden_states"]["duplex_talker_append"].to(device)
            addition = self._duplex_text_queue(ids, rows).to(device, dtype=dtype)
        if final:
            eos = self.tts_eos_embed.to(device, dtype=dtype).reshape(1, -1)
            addition = eos if addition is None else torch.cat([addition, eos], 0)
        if addition is not None:
            queue = addition if queue is None or queue.numel() == 0 else torch.cat([queue, addition], 0)
        update_dict.setdefault("meta", {})[_APPENDED_KEY] = window
        return queue

    def prepare_duplex_sampling(self, logits: torch.Tensor, sampling_metadata: object, rows: object) -> None:
        """Mask every logit the current window phase does not allow.

        The runner calls this before ``sample`` and only for rows whose prompt
        carries a duplex payload. A window that has already closed has nothing
        to constrain: its request stops on the protocol token.
        """

        del sampling_metadata
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            return
        if self.model_stage == "talker":
            self._mask_premature_codec_eos(logits, rows)
            return
        if self.model_stage != "thinker":
            return
        for row in rows:
            row_idx = getattr(row, "row_idx", -1)
            if row_idx < 0 or row_idx >= logits.shape[0]:
                continue
            payload = getattr(row, "payload", None)
            system2_result = payload.get("system2_result") if isinstance(payload, dict) else None
            window = self._stage0_windows.window_for(row.request_id, system2_result=system2_result)
            if window.closed:
                continue
            _mask_illegal_logits(logits[row_idx], window)

    def _mask_premature_codec_eos(self, logits: torch.Tensor, rows: object) -> None:
        """Deny ``codec_eos`` to every window that is not the turn's last.

        ``codec_eos`` is the Talker stage's stop token, so drawing it ends the
        turn. The protocol paces a turn one window at a time, and a window the
        Thinker has already followed with more audio has no right to end it.
        Only the last window of a turn may, which is what ``frame_budget``
        answers; the flag itself travels on the window's own duplex payload.
        """

        codec_eos = int(self.talker_config.codec_eos_token_id)
        if codec_eos >= int(logits.shape[1]):
            return
        for row in rows:
            row_idx = getattr(row, "row_idx", -1)
            if row_idx < 0 or row_idx >= logits.shape[0]:
                continue
            payload = getattr(row, "payload", None)
            if not isinstance(payload, Mapping):
                continue
            if frame_budget(is_final_window=bool(payload.get("final_window"))).allow_codec_eos:
                continue
            logits[row_idx, codec_eos] = float("-inf")

    def sample(self, logits: torch.Tensor, sampling_metadata: object) -> object:
        """Constrain, then draw the token.

        The runner calls this only because ``prefer_model_sampler`` is set, and
        that flag exists so the runner invokes ``prepare_duplex_sampling`` before
        sampling. The constraint is already applied to ``logits`` by then, and
        the draw itself falls back to the runner: returning ``None`` is its
        contract for declining the step, and it then samples from the masked
        logits itself.

        ``_sample_with_parent`` stays so tests can substitute a deterministic
        draw. A substitute that returns tokens advances the window here, because
        no runner calls ``post_sample_talker_mtp`` with those tokens afterwards.
        """

        output = self._sample_with_parent(logits, sampling_metadata)
        if output is None:
            return None
        self._advance_windows(getattr(output, "sampled_token_ids", None), self._sampling_request_ids(sampling_metadata))
        return output

    def post_sample_talker_mtp(
        self,
        *,
        req_ids: list[str],
        req_infos: list[dict[str, Any]],
        input_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Advance each window with the token the runner sampled.

        The runner calls this after its own sampler and passes the sampled token
        per request as ``input_ids``. That is the only place the sampled token
        reaches the model, so the window advances here. The returned rows are
        empty: the next append reserves its own ``<|unit_end|>`` slot, so nothing
        has to travel back through the engine.
        """

        del req_infos, kwargs
        self._advance_windows(input_ids, req_ids)
        return torch.full((len(req_ids), 1), -1, dtype=torch.long)

    def _sample_with_parent(self, logits: torch.Tensor, sampling_metadata: object) -> object:
        del logits, sampling_metadata
        # The parent forwards the draw to the stage module, and neither stage
        # this class serves implements one: the thinker module has no sampler
        # at all, and the talker's layer-0 codec id comes from the runner's
        # sampler exactly as it does in turn-based serving, where this hook is
        # never reached. The 15 residual groups are not drawn here either;
        # talker_mtp runs the code predictor over the drawn layer-0 id.
        return None

    def _advance_windows(self, sampled: object, request_ids: object) -> None:
        """Move each window to the phase its sampled token produces.

        The sampled token is one column per request. A window that was not
        constrained this step, or that already closed, is left untouched.
        """

        if self.model_stage != "thinker" or not isinstance(sampled, torch.Tensor) or not sampled.ndim:
            return
        tokens = sampled.reshape(sampled.shape[0], -1) if sampled.ndim > 1 else sampled.unsqueeze(-1)
        for row_idx, request_id in enumerate(request_ids):
            if row_idx >= tokens.shape[0]:
                break
            token_id = int(tokens[row_idx, 0].item())
            if token_id < 0:
                continue
            window = self._stage0_windows._windows.get(request_id)
            if window is None or window.closed:
                continue
            try:
                window.observe(token_id)
            except ThinkerProtocolError:
                # An illegal token means the constraint was bypassed. Drop the
                # window so the next one starts clean instead of wedging here.
                self._stage0_windows._windows.pop(request_id, None)

    @staticmethod
    def _sampling_request_ids(sampling_metadata: object) -> tuple[object, ...]:
        req_ids = getattr(getattr(sampling_metadata, "input_batch", None), "req_ids", ())
        return tuple(req_ids)

    def _duplex_feature_extractor(self) -> Any:
        extractor = getattr(self, "_qwen3_omni_duplex_feature_extractor", None)
        if extractor is None:
            from vllm.transformers_utils.processor import cached_processor_from_config

            extractor = cached_processor_from_config(self.vllm_config.model_config).feature_extractor
            self._qwen3_omni_duplex_feature_extractor = extractor
        return extractor


def _code_predictor_last(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Move the code predictor's subtree to the end as one uninterrupted run."""
    held: list[tuple[str, torch.Tensor]] = []
    for name, weight in weights:
        if name.startswith(_CODE_PREDICTOR_PREFIX):
            held.append((name, weight))
            continue
        yield name, weight
    yield from held


def _optional_int(primary: object, fallback: object) -> int | None:
    value = primary if primary is not None else fallback
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _window_start(prompt_len: int | None, token_offset: int, span_length: int) -> int | None:
    """Locate the audio window inside one scheduled prefill span.

    The window is the last nine tokens of the append. Anything before it is the
    system prefix or the previous window's ``<|unit_end|>``, and a span that
    ends before the window belongs entirely to those.
    """

    if prompt_len is None:
        return 0 if span_length == AUDIO_WINDOW_PROMPT_TOKENS else None
    window_abs_start = prompt_len - AUDIO_WINDOW_PROMPT_TOKENS
    if token_offset > window_abs_start or token_offset + span_length < prompt_len:
        return None
    return window_abs_start - token_offset


def _mask_illegal_logits(row_logits: torch.Tensor, window: object) -> None:
    """Keep the row's legal logits and set every other one to ``-inf``."""

    vocab = int(row_logits.shape[-1])
    present = [token_id for token_id in DUPLEX_TOKEN_IDS if token_id < vocab]
    values = {token_id: float(row_logits[token_id].item()) for token_id in present}
    # Text tokens are legal in the text phases and are not listed individually,
    # so ask about one of them instead: a phase that keeps text keeps the probe,
    # and a phase that forces a protocol token drops it. Passing protocol tokens
    # alone cannot tell the two apart, which masked text out of every phase.
    values[_TEXT_PROBE_TOKEN_ID] = 0.0
    legal = set(window.constrain(values).keys())
    if _TEXT_PROBE_TOKEN_ID in legal:
        illegal = [token_id for token_id in present if token_id not in legal]
        if illegal:
            row_logits[illegal] = float("-inf")
        return
    masked = row_logits.clone()
    row_logits.fill_(float("-inf"))
    candidates = [token_id for token_id in legal if token_id < vocab]
    for token_id in candidates:
        row_logits[token_id] = masked[token_id]
    # The phase keeps only these few candidates, and ``<|unit_end|>`` is often
    # the only one. It is also the request's stop token, so an upstream
    # processor can mask it. The row would then be all ``-inf``, the sampler
    # would draw an arbitrary token, and the window would reject it with no
    # trace of why.
    if not bool(torch.isfinite(row_logits[candidates]).any()):
        raise ThinkerProtocolError(f"no drawable token left among {sorted(candidates)}")
