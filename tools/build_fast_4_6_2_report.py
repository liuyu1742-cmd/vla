"""Strict provenance aggregation for the reduced OpenVLA 4.6.2 experiment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SUITE_DISPLAY_NAMES = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-10（长时序任务）",
}

EXPECTED_EPISODE_KEYS = frozenset((task_id, 0) for task_id in range(10))
OFT_VIDEO_EVIDENCE_PATTERN = re.compile(
    r"(?:^|--)episode=(?P<episode>\d+)--success=(?P<success>True|False)(?=--|\.mp4$)"
)


def normalize_suite_name(suite: str) -> str:
    """Return the canonical LIBERO suite key, including the legacy navigation alias."""

    if not isinstance(suite, str) or not suite.strip():
        raise ValueError(f"invalid LIBERO suite: {suite!r}")
    key = suite.strip().lower().rstrip("*").replace("-", "_")
    aliases = {
        "libero_spatial": "libero_spatial",
        "libero_object": "libero_object",
        "libero_goal": "libero_goal",
        "libero_10": "libero_10",
        "libero_navigation": "libero_10",
        "navigation": "libero_10",
    }
    try:
        return aliases[key]
    except KeyError as error:
        raise ValueError(f"unknown LIBERO suite: {suite}") from error


def suite_display_name(suite: str) -> str:
    try:
        return SUITE_DISPLAY_NAMES[normalize_suite_name(suite)]
    except KeyError as error:
        raise ValueError(f"unknown LIBERO suite: {suite}") from error


@dataclass(frozen=True)
class NormalizedEpisodeEvidence:
    method: str
    suite: str
    task_id: int
    trial: int
    success: bool
    source: str


@dataclass(frozen=True)
class EvidenceRecord:
    method: str
    suite: str
    successes: int
    episodes: int
    rate: float
    checkpoint_sha256: str
    seed: int
    started_at: str
    finished_at: str
    gpu: str
    source_paths: tuple[str, ...]
    provenance: str = "local_measured"
    protocol_deviations: tuple[str, ...] = ()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return payload


def _require_int(payload: Mapping[str, Any], key: str, source: Path) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ValueError(f"evidence must contain integer {key}: {source}")
    return value


def _require_string(payload: Mapping[str, Any], key: str, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"evidence must contain non-empty {key}: {source}")
    return value


def _require_boolean_success(payload: Mapping[str, Any], source: Path) -> bool:
    success = payload.get("success")
    if type(success) is not bool:
        raise ValueError(f"episode must contain boolean success: {source}")
    return success


def _validate_normalized_episodes(
    rows: Iterable[NormalizedEpisodeEvidence],
) -> tuple[NormalizedEpisodeEvidence, ...]:
    normalized = tuple(rows)
    if len(normalized) != 10:
        raise ValueError("suite evidence must contain exactly 10 tasks with trial 0")

    seen: set[tuple[int, int]] = set()
    for row in normalized:
        if not isinstance(row.method, str) or not row.method:
            raise ValueError("normalized episode method must be non-empty")
        if row.suite != normalize_suite_name(row.suite):
            raise ValueError(f"normalized episode suite is not canonical: {row.suite}")
        if type(row.task_id) is not int or type(row.trial) is not int:
            raise ValueError("normalized task_id and trial must be integers")
        if type(row.success) is not bool:
            raise ValueError(f"episode must contain boolean success: {row.source}")
        identity = (row.task_id, row.trial)
        if identity in seen:
            raise ValueError(f"duplicate task/trial episode: {identity}")
        seen.add(identity)

    if seen != EXPECTED_EPISODE_KEYS:
        raise ValueError("suite evidence must contain exactly 10 tasks with trial 0")
    if len({row.method for row in normalized}) != 1:
        raise ValueError("episode evidence disagrees on method")
    if len({row.suite for row in normalized}) != 1:
        raise ValueError("episode evidence disagrees on suite")
    return tuple(sorted(normalized, key=lambda row: (row.task_id, row.trial)))


def _summary_rate_matches(
    summary: Mapping[str, Any],
    *,
    successes: int,
    source: Path,
) -> None:
    summary_successes = _require_int(summary, "successes", source)
    if summary_successes != successes:
        raise ValueError(
            f"summary successes disagree with episode files: {summary_successes} != {successes}"
        )
    rate = summary.get("success_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError(f"summary success_rate must be numeric: {source}")
    expected_rate = successes / 10
    if abs(float(rate) - expected_rate) > 1e-12:
        raise ValueError(
            f"summary success_rate disagrees with episode files: {rate} != {expected_rate}"
        )


def normalize_oft_manifest(
    manifest_path: Path,
) -> tuple[NormalizedEpisodeEvidence, ...]:
    """Normalize one completed OFT suite manifest and its ten rollout videos."""

    path = Path(manifest_path)
    manifest = _read_json_object(path)
    if manifest.get("schema_version") != "openvla_oft_libero_run_v2":
        raise ValueError(f"unsupported OFT manifest schema: {path}")
    if manifest.get("completed") is not True or _require_int(manifest, "returncode", path) != 0:
        raise ValueError(f"OFT run is not completed successfully: {path}")
    if manifest.get("uses_expert_recovery") is not False:
        raise ValueError(f"OFT evidence must disable expert recovery: {path}")
    suite = normalize_suite_name(_require_string(manifest, "suite", path))
    if (
        _require_int(manifest, "tasks", path) != 10
        or _require_int(manifest, "trials_per_task", path) != 1
        or _require_int(manifest, "requested_trials", path) != 10
        or _require_int(manifest, "total_episodes", path) != 10
    ):
        raise ValueError(f"OFT manifest must describe exactly 10 tasks with one trial: {path}")

    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) != 10:
        raise ValueError(f"OFT manifest must list exactly 10 episode videos: {path}")
    rows = []
    seen_sources: set[Path] = set()
    for raw_video in videos:
        if not isinstance(raw_video, str) or not raw_video:
            raise ValueError(f"OFT manifest has invalid video path: {path}")
        video = Path(raw_video)
        if not video.is_absolute():
            video = path.parent / video
        video = video.resolve()
        if video in seen_sources:
            raise ValueError(f"duplicate OFT video source: {video}")
        seen_sources.add(video)
        if not video.is_file() or video.stat().st_size <= 0:
            raise ValueError(f"missing or empty OFT video: {video}")
        match = OFT_VIDEO_EVIDENCE_PATTERN.search(video.name)
        if match is None:
            raise ValueError(f"OFT video lacks explicit boolean success: {video}")
        episode_number = int(match.group("episode"))
        rows.append(
            NormalizedEpisodeEvidence(
                method="openvla_oft",
                suite=suite,
                task_id=episode_number - 1,
                trial=0,
                success=match.group("success") == "True",
                source=str(video),
            )
        )

    normalized = _validate_normalized_episodes(rows)
    parsed_successes = sum(row.success for row in normalized)
    declared_successes = _require_int(manifest, "total_successes", path)
    if declared_successes != parsed_successes:
        raise ValueError(
            "OFT manifest total_successes disagrees with explicit episode booleans: "
            f"{declared_successes} != {parsed_successes}"
        )
    return normalized


def normalize_baseline_summary(
    summary_path: Path,
) -> tuple[NormalizedEpisodeEvidence, ...]:
    """Normalize a strict baseline suite summary and nested episode JSON files."""

    path = Path(summary_path)
    summary = _read_json_object(path)
    if summary.get("schema_version") != "openvla_baseline_libero_suite_v1":
        raise ValueError(f"unsupported baseline summary schema: {path}")
    if summary.get("integrity_status") != "valid":
        raise ValueError(f"baseline summary integrity is not valid: {path}")
    if summary.get("uses_expert_recovery") is not False:
        raise ValueError(f"baseline evidence must disable expert recovery: {path}")
    suite = normalize_suite_name(_require_string(summary, "suite", path))
    seed = _require_int(summary, "seed", path)
    checkpoint_sha256 = _require_string(summary, "checkpoint_sha256", path)
    if (
        _require_int(summary, "requested_episodes", path) != 10
        or _require_int(summary, "completed_episodes", path) != 10
        or _require_int(summary, "error_episodes", path) != 0
        or _require_int(summary, "trials_per_task", path) != 1
    ):
        raise ValueError(f"baseline summary must contain 10 completed and zero error episodes: {path}")
    task_ids = summary.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or any(type(task_id) is not int for task_id in task_ids)
        or sorted(task_ids) != list(range(10))
    ):
        raise ValueError(f"baseline summary must cover exactly task IDs 0..9: {path}")

    expected_paths = {
        (
            path.parent
            / "episodes"
            / f"task_{task_id:03d}"
            / "trial_000.json"
        ).resolve()
        for task_id in range(10)
    }
    discovered_paths = {
        episode.resolve()
        for episode in (path.parent / "episodes").glob("task_*/trial_*.json")
    }
    if discovered_paths != expected_paths:
        raise ValueError("baseline suite must contain exactly 10 tasks with trial 0")
    declared_paths = summary.get("episode_paths")
    if not isinstance(declared_paths, list) or len(declared_paths) != 10:
        raise ValueError(f"baseline summary must declare exactly 10 episode paths: {path}")
    resolved_declared_paths = {
        Path(value).resolve()
        for value in declared_paths
        if isinstance(value, str) and value
    }
    if resolved_declared_paths != expected_paths:
        raise ValueError(f"baseline summary episode_paths disagree with nested files: {path}")

    rows = []
    for episode_path in sorted(expected_paths):
        episode = _read_json_object(episode_path)
        if episode.get("schema_version") != "openvla_baseline_libero_episode_v1":
            raise ValueError(f"unsupported baseline episode schema: {episode_path}")
        if (
            episode.get("status") != "completed"
            or "error" not in episode
            or episode["error"] is not None
        ):
            raise ValueError(
                f"baseline episode must be completed with null error: {episode_path}"
            )
        if type(episode.get("done")) is not bool:
            raise ValueError(f"baseline episode must contain boolean done: {episode_path}")
        success = _require_boolean_success(episode, episode_path)
        if episode.get("uses_expert_recovery") is not False:
            raise ValueError(f"baseline episode used expert recovery: {episode_path}")
        episode_suite = normalize_suite_name(_require_string(episode, "suite", episode_path))
        if episode_suite != suite:
            raise ValueError(f"baseline episode evidence disagrees on suite: {episode_path}")
        if _require_int(episode, "seed", episode_path) != seed:
            raise ValueError(f"baseline episode evidence disagrees on seed: {episode_path}")
        if _require_string(episode, "checkpoint_sha256", episode_path) != checkpoint_sha256:
            raise ValueError(
                f"baseline episode evidence disagrees on checkpoint_sha256: {episode_path}"
            )
        rows.append(
            NormalizedEpisodeEvidence(
                method="openvla_baseline",
                suite=suite,
                task_id=_require_int(episode, "task_id", episode_path),
                trial=_require_int(episode, "trial", episode_path),
                success=success,
                source=str(episode_path),
            )
        )

    normalized = _validate_normalized_episodes(rows)
    _summary_rate_matches(
        summary,
        successes=sum(row.success for row in normalized),
        source=path,
    )
    return normalized


def normalize_diffusion_summary(
    summary_path: Path,
) -> tuple[NormalizedEpisodeEvidence, ...]:
    """Normalize a diffusion-policy suite summary and flat episode JSON files."""

    path = Path(summary_path)
    summary = _read_json_object(path)
    suite = normalize_suite_name(_require_string(summary, "suite", path))
    seed = _require_int(summary, "seed", path)
    checkpoint_sha256 = _require_string(summary, "checkpoint_sha256", path)
    if _require_int(summary, "episodes", path) != 10:
        raise ValueError(f"diffusion summary must describe exactly 10 tasks: {path}")

    expected_paths = {
        (
            path.parent
            / "episodes"
            / f"task_{task_id:03d}_trial_000.json"
        ).resolve()
        for task_id in range(10)
    }
    discovered_paths = {
        episode.resolve()
        for episode in (path.parent / "episodes").glob("task_*_trial_*.json")
    }
    if discovered_paths != expected_paths:
        raise ValueError("diffusion suite must contain exactly 10 tasks with trial 0")

    rows = []
    episode_payloads: dict[tuple[int, int], dict[str, Any]] = {}
    for episode_path in sorted(expected_paths):
        episode = _read_json_object(episode_path)
        success = _require_boolean_success(episode, episode_path)
        if "error" not in episode or episode["error"] is not None:
            raise ValueError(f"diffusion error episode is not valid evidence: {episode_path}")
        episode_suite = normalize_suite_name(_require_string(episode, "suite", episode_path))
        if episode_suite != suite:
            raise ValueError(f"diffusion episode evidence disagrees on suite: {episode_path}")
        if _require_int(episode, "seed", episode_path) != seed:
            raise ValueError(f"diffusion episode evidence disagrees on seed: {episode_path}")
        if _require_string(episode, "checkpoint_sha256", episode_path) != checkpoint_sha256:
            raise ValueError(
                f"diffusion episode evidence disagrees on checkpoint_sha256: {episode_path}"
            )
        task_id = _require_int(episode, "task_id", episode_path)
        trial = _require_int(episode, "trial_index", episode_path)
        identity = (task_id, trial)
        if identity in episode_payloads:
            raise ValueError(f"duplicate task/trial episode: {identity}")
        episode_payloads[identity] = episode
        rows.append(
            NormalizedEpisodeEvidence(
                method="diffusion_policy",
                suite=suite,
                task_id=task_id,
                trial=trial,
                success=success,
                source=str(episode_path),
            )
        )

    normalized = _validate_normalized_episodes(rows)
    successes = sum(row.success for row in normalized)
    _summary_rate_matches(summary, successes=successes, source=path)

    summary_rows = summary.get("episode_results")
    if not isinstance(summary_rows, list) or len(summary_rows) != 10:
        raise ValueError(f"diffusion summary must contain 10 episode_results: {path}")
    summary_payloads: dict[tuple[int, int], dict[str, Any]] = {}
    for summary_row in summary_rows:
        if not isinstance(summary_row, dict):
            raise ValueError(f"diffusion summary episode_results must be objects: {path}")
        identity = (
            _require_int(summary_row, "task_id", path),
            _require_int(summary_row, "trial_index", path),
        )
        if identity in summary_payloads:
            raise ValueError(f"duplicate task/trial in diffusion summary: {identity}")
        summary_payloads[identity] = summary_row
    if set(summary_payloads) != EXPECTED_EPISODE_KEYS:
        raise ValueError("diffusion summary must contain exactly 10 tasks with trial 0")
    for identity, episode in episode_payloads.items():
        if summary_payloads[identity] != episode:
            raise ValueError(
                f"diffusion summary episode_results disagree with episode file: {identity}"
            )
    return normalized


def aggregate_method_suite(
    episode_paths: Iterable[Path],
) -> EvidenceRecord:
    """Aggregate only concrete, mutually consistent local episode JSON files."""

    paths = [Path(path) for path in episode_paths]
    if not paths:
        raise ValueError("at least one episode path is required")

    rows: list[dict[str, Any]] = []
    normalized_rows: list[NormalizedEpisodeEvidence] = []
    for path in paths:
        payload = _read_json_object(path)
        success = _require_boolean_success(payload, path)
        if payload.get("error") is not None:
            raise ValueError(f"error episode is not valid evidence: {path}")
        if "status" in payload and payload["status"] != "completed":
            raise ValueError(f"episode status is not completed: {path}")
        for key in (
            "method",
            "suite",
            "checkpoint_sha256",
            "seed",
            "started_at",
            "finished_at",
            "gpu",
            "task_id",
            "trial",
        ):
            if key not in payload:
                raise ValueError(f"episode is missing {key}: {path}")
        task_id = _require_int(payload, "task_id", path)
        trial = _require_int(payload, "trial", path)
        payload["suite"] = normalize_suite_name(str(payload["suite"]))
        normalized_rows.append(
            NormalizedEpisodeEvidence(
                method=str(payload["method"]),
                suite=str(payload["suite"]),
                task_id=task_id,
                trial=trial,
                success=success,
                source=str(path.resolve()),
            )
        )
        rows.append(payload)

    normalized = _validate_normalized_episodes(normalized_rows)

    for key in ("method", "suite", "checkpoint_sha256", "seed", "gpu"):
        values = {row[key] for row in rows}
        if len(values) != 1:
            raise ValueError(f"episode evidence disagrees on {key}")

    successes = sum(1 for row in normalized if row.success)
    deviations = sorted(
        {
            str(deviation)
            for row in rows
            for deviation in row.get("protocol_deviations", [])
        }
    )
    return EvidenceRecord(
        method=str(rows[0]["method"]),
        suite=normalized[0].suite,
        successes=successes,
        episodes=len(rows),
        rate=successes / len(rows),
        checkpoint_sha256=str(rows[0]["checkpoint_sha256"]),
        seed=int(rows[0]["seed"]),
        started_at=min(str(row["started_at"]) for row in rows),
        finished_at=max(str(row["finished_at"]) for row in rows),
        gpu=str(rows[0]["gpu"]),
        source_paths=tuple(str(path.resolve()) for path in paths),
        protocol_deviations=tuple(deviations),
    )
