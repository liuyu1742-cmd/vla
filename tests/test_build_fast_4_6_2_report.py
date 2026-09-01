import json
from pathlib import Path

import pytest

import tools.build_fast_4_6_2_report as report
from tools.build_fast_4_6_2_report import (
    aggregate_method_suite,
    suite_display_name,
)


def write_episode(
    path: Path,
    *,
    success: bool,
    task_id: int,
    method: str = "diffusion_policy",
    suite: str = "libero_goal",
) -> None:
    path.write_text(
        json.dumps(
            {
                "method": method,
                "suite": suite,
                "task_id": task_id,
                "trial": 0,
                "success": success,
                "checkpoint_sha256": "a" * 64,
                "seed": 7,
                "started_at": "2026-07-31T01:00:00Z",
                "finished_at": "2026-07-31T01:01:00Z",
                "gpu": "NVIDIA GeForce RTX 3090",
            }
        ),
        encoding="utf-8",
    )


def test_aggregation_counts_only_concrete_boolean_episode_evidence(tmp_path: Path):
    paths = []
    successes = (True, False, True, False, True, False, True, False, True, False)
    for task_id, success in enumerate(successes):
        path = tmp_path / f"episode_{task_id}.json"
        write_episode(path, success=success, task_id=task_id)
        paths.append(path)

    record = aggregate_method_suite(paths)
    assert record.method == "diffusion_policy"
    assert record.suite == "libero_goal"
    assert record.successes == 5
    assert record.episodes == 10
    assert record.rate == 0.5
    assert record.provenance == "local_measured"
    assert record.source_paths == tuple(str(path.resolve()) for path in paths)


@pytest.mark.parametrize("bad_success", [None, 1, "true"])
def test_aggregation_rejects_missing_or_non_boolean_success(
    tmp_path: Path,
    bad_success,
):
    path = tmp_path / "episode.json"
    write_episode(path, success=True, task_id=0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if bad_success is None:
        payload.pop("success")
    else:
        payload["success"] = bad_success
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="boolean success"):
        aggregate_method_suite([path])


def test_fourth_suite_is_labeled_as_libero_10_not_mobile_navigation():
    assert suite_display_name("libero_10") == "LIBERO-10（长时序任务）"


def test_aggregation_rejects_incomplete_ten_task_protocol(tmp_path: Path):
    paths = []
    for task_id in range(9):
        path = tmp_path / f"episode_{task_id}.json"
        write_episode(path, success=False, task_id=task_id)
        paths.append(path)

    with pytest.raises(ValueError, match="exactly 10 tasks"):
        aggregate_method_suite(paths)


def test_aggregation_rejects_error_episode_instead_of_counting_failure(tmp_path: Path):
    paths = []
    for task_id in range(10):
        path = tmp_path / f"episode_{task_id}.json"
        write_episode(path, success=False, task_id=task_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["error"] = "renderer unavailable" if task_id == 4 else None
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="error episode"):
        aggregate_method_suite(paths)


def _write_oft_manifest(
    root: Path,
    *,
    suite: str = "libero_goal",
    successes: int = 9,
) -> Path:
    run_dir = root / "oft_run"
    run_dir.mkdir(parents=True)
    videos = []
    for task_id in range(10):
        succeeded = task_id < successes
        video = run_dir / (
            f"run--episode={task_id + 1}--success={succeeded}--task=fixture.mp4"
        )
        video.write_bytes(b"video")
        videos.append(str(video))
    manifest = {
        "schema_version": "openvla_oft_libero_run_v2",
        "suite": suite,
        "completed": True,
        "returncode": 0,
        "seed": 7,
        "trials_per_task": 1,
        "tasks": 10,
        "requested_trials": 10,
        "total_episodes": 10,
        "total_successes": successes,
        "uses_expert_recovery": False,
        "videos": videos,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_baseline_summary(root: Path, *, suite: str = "libero_object") -> Path:
    run_dir = root / suite / "seed_7"
    episode_paths = []
    successes = 0
    for task_id in range(10):
        success = task_id % 2 == 0
        successes += int(success)
        path = run_dir / "episodes" / f"task_{task_id:03d}" / "trial_000.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "openvla_baseline_libero_episode_v1",
                    "suite": suite,
                    "seed": 7,
                    "checkpoint_sha256": "b" * 64,
                    "task_id": task_id,
                    "trial": 0,
                    "status": "completed",
                    "done": success,
                    "success": success,
                    "error": None,
                    "uses_expert_recovery": False,
                }
            ),
            encoding="utf-8",
        )
        episode_paths.append(str(path))
    summary = {
        "schema_version": "openvla_baseline_libero_suite_v1",
        "suite": suite,
        "seed": 7,
        "checkpoint_sha256": "b" * 64,
        "uses_expert_recovery": False,
        "task_ids": list(range(10)),
        "trials_per_task": 1,
        "requested_episodes": 10,
        "completed_episodes": 10,
        "error_episodes": 0,
        "successes": successes,
        "success_rate": successes / 10,
        "integrity_status": "valid",
        "episode_paths": episode_paths,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _write_diffusion_summary(root: Path, *, suite: str = "libero_spatial") -> Path:
    run_dir = root / suite / "seed_42"
    rows = []
    successes = 0
    for task_id in range(10):
        success = task_id < 6
        successes += int(success)
        row = {
            "suite": suite,
            "task_id": task_id,
            "trial_index": 0,
            "checkpoint_sha256": "d" * 64,
            "seed": 42,
            "success": success,
            "error": None,
        }
        path = run_dir / "episodes" / f"task_{task_id:03d}_trial_000.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row), encoding="utf-8")
        rows.append(row)
    summary = {
        "suite": suite,
        "checkpoint_sha256": "d" * 64,
        "seed": 42,
        "episodes": 10,
        "successes": successes,
        "success_rate": successes / 10,
        "episode_results": rows,
    }
    summary_path = run_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_oft_manifest_normalizes_explicit_episode_boolean_and_navigation_alias(
    tmp_path: Path,
):
    manifest_path = _write_oft_manifest(
        tmp_path,
        suite="LIBERO-Navigation*",
        successes=9,
    )

    rows = report.normalize_oft_manifest(manifest_path)

    assert len(rows) == 10
    assert {row.method for row in rows} == {"openvla_oft"}
    assert {row.suite for row in rows} == {"libero_10"}
    assert {(row.task_id, row.trial) for row in rows} == {
        (task_id, 0) for task_id in range(10)
    }
    assert sum(row.success for row in rows) == 9
    assert all(Path(row.source).is_absolute() for row in rows)


def test_oft_manifest_rejects_total_that_disagrees_with_episode_booleans(
    tmp_path: Path,
):
    manifest_path = _write_oft_manifest(tmp_path, successes=9)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_successes"] = 8
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="total_successes"):
        report.normalize_oft_manifest(manifest_path)


def test_oft_manifest_rejects_video_without_explicit_boolean_success(tmp_path: Path):
    manifest_path = _write_oft_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = Path(manifest["videos"][0])
    ambiguous = original.with_name(
        original.name.replace("success=True", "success=1")
    )
    original.rename(ambiguous)
    manifest["videos"][0] = str(ambiguous)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="explicit boolean success"):
        report.normalize_oft_manifest(manifest_path)


def test_baseline_nested_episode_files_normalize_to_common_contract(tmp_path: Path):
    summary_path = _write_baseline_summary(tmp_path)

    rows = report.normalize_baseline_summary(summary_path)

    assert len(rows) == 10
    assert rows[0] == report.NormalizedEpisodeEvidence(
        method="openvla_baseline",
        suite="libero_object",
        task_id=0,
        trial=0,
        success=True,
        source=str(
            (
                summary_path.parent
                / "episodes"
                / "task_000"
                / "trial_000.json"
            ).resolve()
        ),
    )


def test_baseline_rejects_error_status_instead_of_counting_false(tmp_path: Path):
    summary_path = _write_baseline_summary(tmp_path)
    episode_path = summary_path.parent / "episodes" / "task_004" / "trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode.update({"status": "error", "success": False, "error": "model failed"})
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="completed.*error|null error"):
        report.normalize_baseline_summary(summary_path)


def test_baseline_rejects_suite_mismatch(tmp_path: Path):
    summary_path = _write_baseline_summary(tmp_path)
    episode_path = summary_path.parent / "episodes" / "task_003" / "trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["suite"] = "libero_goal"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="suite"):
        report.normalize_baseline_summary(summary_path)


def test_baseline_rejects_non_boolean_success(tmp_path: Path):
    summary_path = _write_baseline_summary(tmp_path)
    episode_path = summary_path.parent / "episodes" / "task_003" / "trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["success"] = 0
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="boolean success"):
        report.normalize_baseline_summary(summary_path)


def test_baseline_requires_explicit_null_error_field(tmp_path: Path):
    summary_path = _write_baseline_summary(tmp_path)
    episode_path = summary_path.parent / "episodes" / "task_003" / "trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode.pop("error")
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="null error"):
        report.normalize_baseline_summary(summary_path)


def test_diffusion_flat_episode_files_and_summary_normalize_trial_index(tmp_path: Path):
    summary_path = _write_diffusion_summary(tmp_path)

    rows = report.normalize_diffusion_summary(summary_path)

    assert len(rows) == 10
    assert {row.method for row in rows} == {"diffusion_policy"}
    assert {row.trial for row in rows} == {0}
    assert sum(row.success for row in rows) == 6
    assert all(Path(row.source).name.startswith("task_") for row in rows)


def test_diffusion_rejects_execution_error_instead_of_counting_false(tmp_path: Path):
    summary_path = _write_diffusion_summary(tmp_path)
    episode_path = summary_path.parent / "episodes" / "task_007_trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["error"] = "simulator disconnected"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="error episode"):
        report.normalize_diffusion_summary(summary_path)


def test_diffusion_rejects_missing_success(tmp_path: Path):
    summary_path = _write_diffusion_summary(tmp_path)
    episode_path = summary_path.parent / "episodes" / "task_007_trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode.pop("success")
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="boolean success"):
        report.normalize_diffusion_summary(summary_path)


def test_diffusion_rejects_duplicate_task_trial_identity(tmp_path: Path):
    summary_path = _write_diffusion_summary(tmp_path)
    episode_path = summary_path.parent / "episodes" / "task_009_trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["task_id"] = 8
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate task/trial"):
        report.normalize_diffusion_summary(summary_path)


def test_diffusion_rejects_incomplete_ten_task_suite(tmp_path: Path):
    summary_path = _write_diffusion_summary(tmp_path)
    (summary_path.parent / "episodes" / "task_009_trial_000.json").unlink()

    with pytest.raises(ValueError, match="exactly 10 tasks"):
        report.normalize_diffusion_summary(summary_path)


def test_diffusion_rejects_summary_that_disagrees_with_episode_files(tmp_path: Path):
    summary_path = _write_diffusion_summary(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["successes"] = 5
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="summary successes"):
        report.normalize_diffusion_summary(summary_path)
