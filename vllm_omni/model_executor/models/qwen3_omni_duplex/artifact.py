# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint gate for the Qwen3-Omni-Duplex runtime protocol.

A v2 checkpoint is schema-valid under v3 but was trained to stop a window on
``<|chunk_eos|>``, which v3 never samples. The version string is the only thing
that can reject it, so the Thinker reads this manifest before it serves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

from vllm_omni.model_executor.models.qwen3_omni_duplex.thinker_window import UserSignalMode

RUNTIME_PROTOCOL = "qwen3omni_duplex_unit_end_user_signal/v3"
MANIFEST_FILENAME = "duplex-artifact-manifest.json"


def read_user_signal(model_path: str) -> UserSignalMode:
    """Return the checkpoint's user-signal mode, rejecting a non-v3 artifact."""

    manifest_path = Path(model_path) / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Qwen3-Omni-Duplex needs {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} is not a duplex artifact manifest")

    protocol = manifest.get("runtime_protocol")
    if protocol != RUNTIME_PROTOCOL:
        raise ValueError(f"{manifest_path} declares runtime protocol {protocol!r}, not {RUNTIME_PROTOCOL!r}")

    provenance = manifest.get("training_provenance")
    plugin_config = provenance.get("plugin_config") if isinstance(provenance, dict) else None
    user_signal = plugin_config.get("user_signal") if isinstance(plugin_config, dict) else None
    if user_signal not in get_args(UserSignalMode):
        raise ValueError(f"{manifest_path} declares user signal mode {user_signal!r}")
    return user_signal
