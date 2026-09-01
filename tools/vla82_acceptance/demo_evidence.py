"""Materialize traceable public-human demonstration evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bindings(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bindings = payload.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("human demonstration binding file lacks a bindings list")
    return bindings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, int | float | str]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {
                "decoder": "opencv",
                "frame_count": 0,
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "duration_seconds": 0.0,
            }
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return {
            "decoder": "opencv",
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
            "duration_seconds": frame_count / fps if fps > 0 else 0.0,
        }
    finally:
        capture.release()


def extract_preview(path: Path, seconds: float, output: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"video cannot be decoded: {path}")
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(seconds)) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"video has no decodable frame: {path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise OSError(f"failed to write preview: {output}")
    finally:
        capture.release()


def validate_demo_binding(
    binding: Mapping[str, Any], *, require_human: bool = True, decode: bool = True
) -> list[str]:
    errors: list[str] = []
    if require_human and binding.get("source_kind") != "public_human_demonstration":
        errors.append("source_is_not_human_demonstration")
    for key in ("source_table", "task", "selection_id", "dataset", "clip_id"):
        if not str(binding.get(key, "")).strip():
            errors.append(f"missing_field:{key}")
    video = Path(str(binding.get("video", "")))
    if not video.is_file():
        errors.append("video_missing")
    elif decode:
        probe = probe_video(video)
        if int(probe["frame_count"]) < 1:
            errors.append("video_decode_failed")
        elif float(binding.get("stop_seconds", 0.0)) > float(
            probe["duration_seconds"]
        ) + 0.05:
            errors.append("segment_exceeds_video")
    try:
        start = float(binding.get("start_seconds", 0.0))
        stop = float(binding.get("stop_seconds", 0.0))
    except (TypeError, ValueError):
        errors.append("invalid_time_range")
    else:
        if start < 0 or stop <= start:
            errors.append("invalid_time_range")
    return errors


def validate_category_coverage(
    registry: Mapping[str, Any], bindings: Sequence[Mapping[str, Any]]
) -> list[str]:
    errors: list[str] = []
    expected = {
        (str(item["source_table"]), str(item["task"]))
        for item in registry.get("selected_tasks", [])
    }
    actual = {
        (str(item.get("source_table", "")), str(item.get("task", "")))
        for item in bindings
    }
    if len(bindings) != 8:
        errors.append(f"binding_count:{len(bindings)}")
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing_categories:{missing}")
    if extra:
        errors.append(f"unexpected_categories:{extra}")
    tables = [str(item.get("source_table", "")) for item in bindings]
    if len(set(tables)) != len(tables):
        errors.append("duplicate_source_table")
    selections = [str(item.get("selection_id", "")) for item in bindings]
    if len(set(selections)) != len(selections):
        errors.append("duplicate_selection_id")
    for binding in bindings:
        errors.extend(
            f"{binding.get('selection_id')}:{error}"
            for error in validate_demo_binding(binding, require_human=True, decode=True)
        )
    return errors


def _materialize_binding(
    binding: Mapping[str, Any], output_dir: Path, *, require_human: bool
) -> dict[str, Any]:
    errors = validate_demo_binding(binding, require_human=require_human, decode=True)
    if errors:
        raise ValueError("invalid public demonstration: " + ", ".join(errors))
    video = Path(str(binding["video"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = output_dir / "source_demo_preview.png"
    midpoint = (
        float(binding["start_seconds"]) + float(binding["stop_seconds"])
    ) / 2.0
    extract_preview(video, midpoint, preview)
    report = {
        "schema_version": "vla82_public_human_demo_evidence_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_table": binding["source_table"],
        "task": binding["task"],
        "object": binding.get("object"),
        "selection_id": binding["selection_id"],
        "source_kind": binding["source_kind"],
        "dataset": binding["dataset"],
        "clip_id": binding["clip_id"],
        "video": str(video),
        "start_seconds": float(binding["start_seconds"]),
        "stop_seconds": float(binding["stop_seconds"]),
        "semantic_operation": str(binding.get("semantic_operation", "")),
        "sha256": sha256_file(video),
        "video_probe": probe_video(video),
        "preview": str(preview.resolve()),
    }
    destination = output_dir / "source_demo.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return report


def materialize_demo(binding: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    return _materialize_binding(binding, output_dir, require_human=True)


def materialize_registry_object_demo(
    selection: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Materialize the exact selected object's existing public source video."""
    source_root = Path(str(selection["data_path"]))
    candidates = [source_root / str(name) for name in selection.get("video_files", [])]
    video = next((path for path in candidates if path.is_file()), None)
    if video is None:
        raise FileNotFoundError(f"no registry video exists for {selection['selection_id']}")
    source_manifest = source_root / "source_manifest.json"
    source_payload: dict[str, Any] = {}
    if source_manifest.is_file():
        source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    probe = probe_video(video)
    binding = {
        "source_table": selection["source_table"],
        "task": selection["task"],
        "object": selection["object"],
        "selection_id": selection["selection_id"],
        "source_kind": "public_robot_or_simulator_demonstration",
        "dataset": str(source_payload.get("dataset", "project_public_dataset")),
        "clip_id": f"{selection['selection_id']}:{video.name}",
        "video": str(video.resolve()),
        "start_seconds": 0.0,
        "stop_seconds": float(probe["duration_seconds"]),
        "semantic_operation": str(selection["operation_label"]),
    }
    return _materialize_binding(binding, output_dir, require_human=False)
