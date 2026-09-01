import json
from pathlib import Path

import pytest

from tools.build_openvla_oft_4_6_2_report import (
    compute_rate,
    discover_manifests,
    validate_run,
)


def _write_manifest(root: Path, suite: str, successes: int = 9) -> Path:
    run_dir = root / suite / "seed_7" / "run_001"
    run_dir.mkdir(parents=True)
    videos = []
    for index in range(10):
        video = run_dir / f"episode={index + 1}--success=True.mp4"
        video.write_bytes(b"video")
        videos.append(str(video))
    manifest = {
        "suite": suite,
        "completed": True,
        "total_episodes": 10,
        "total_successes": successes,
        "requested_trials": 10,
        "videos": videos,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_discovers_all_four_suites_and_computes_rate(tmp_path: Path):
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        _write_manifest(tmp_path, suite)

    found = discover_manifests(tmp_path)

    assert set(found) == {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    assert compute_rate(json.loads(found["libero_goal"].read_text())) == 90.0


def test_validate_run_rejects_missing_video(tmp_path: Path):
    path = _write_manifest(tmp_path, "libero_goal")
    data = json.loads(path.read_text())
    Path(data["videos"][0]).unlink()

    with pytest.raises(ValueError, match="missing video"):
        validate_run(data)
