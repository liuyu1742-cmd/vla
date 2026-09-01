from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_load_strict_pass_source_rejects_failed_episode(tmp_path: Path) -> None:
    from tools.replay_vla82_strict_pass_video import load_strict_pass_source

    episode = tmp_path / "episode-2000.npz"
    episode.write_bytes(b"npz")
    episode.with_suffix(".json").write_text(
        json.dumps({"status": "FAIL", "predicate_success": False}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict PASS"):
        load_strict_pass_source(episode)


def test_load_strict_pass_source_returns_identity(tmp_path: Path) -> None:
    from tools.replay_vla82_strict_pass_video import load_strict_pass_source

    episode = tmp_path / "episode-2000.npz"
    episode.write_bytes(b"npz")
    episode.with_suffix(".json").write_text(
        json.dumps(
            {
                "selection_id": "VLA82-045",
                "seed": 2000,
                "status": "PASS",
                "predicate_success": True,
            }
        ),
        encoding="utf-8",
    )

    source = load_strict_pass_source(episode)

    assert source.selection_id == "VLA82-045"
    assert source.seed == 2000
    assert source.npz_path == episode.resolve()


def test_script_bootstraps_project_root_from_any_working_directory(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / "replay_vla82_strict_pass_video.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
