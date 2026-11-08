# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Thinker refuses to serve a checkpoint that is not a v3 duplex artifact."""

import json
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.qwen3_omni_duplex.artifact import (
    MANIFEST_FILENAME,
    RUNTIME_PROTOCOL,
    read_user_signal,
)


def _checkpoint(tmp_path: Path, manifest: object) -> str:
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return str(tmp_path)


@pytest.mark.parametrize("user_signal", ["none", "transcript", "activity"])
def test_manifest_user_signal_selects_the_window_protocol(tmp_path: Path, user_signal: str) -> None:
    path = _checkpoint(
        tmp_path,
        {
            "runtime_protocol": RUNTIME_PROTOCOL,
            "training_provenance": {"plugin_config": {"user_signal": user_signal}},
        },
    )

    assert read_user_signal(path) == user_signal


def test_a_v2_checkpoint_is_rejected(tmp_path: Path) -> None:
    # A v2 artifact is schema-valid but stops its windows on <|chunk_eos|>.
    path = _checkpoint(
        tmp_path,
        {
            "runtime_protocol": "qwen3omni_duplex_system2_continuous_talker/v2",
            "training_provenance": {"plugin_config": {"user_signal": "none"}},
        },
    )

    with pytest.raises(ValueError, match="runtime protocol"):
        read_user_signal(path)


def test_an_unknown_user_signal_and_a_missing_manifest_are_rejected(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, {"runtime_protocol": RUNTIME_PROTOCOL, "training_provenance": {}})

    with pytest.raises(ValueError, match="user signal mode"):
        read_user_signal(path)
    with pytest.raises(ValueError, match=MANIFEST_FILENAME):
        read_user_signal(str(tmp_path / "absent"))
