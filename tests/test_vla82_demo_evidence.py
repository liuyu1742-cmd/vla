from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tools.vla82_acceptance.demo_evidence import (
    load_bindings,
    load_registry,
    materialize_demo,
    materialize_registry_object_demo,
    validate_category_coverage,
    validate_demo_binding,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs/midterm_testing_vla82/vla82_midterm_registry.json"
BINDINGS = ROOT / "configs/vla82_acceptance/human_demo_8.json"


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 32)
    )
    assert writer.isOpened()
    for index in range(5):
        frame = np.full((32, 48, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_demo_binding_rejects_robot_only_source(tmp_path: Path):
    video = tmp_path / "demo.mp4"
    _write_video(video)
    binding = {
        "source_table": "5-8-1",
        "task": "室内卫生清洁",
        "selection_id": "VLA82-002",
        "source_kind": "robot_demonstration",
        "video": str(video),
        "start_seconds": 0.0,
        "stop_seconds": 0.8,
    }

    errors = validate_demo_binding(binding, require_human=True, decode=True)

    assert "source_is_not_human_demonstration" in errors


def test_eight_bindings_cover_eight_registry_tables():
    registry = load_registry(REGISTRY)
    bindings = load_bindings(BINDINGS)

    assert len(bindings) == 8
    assert validate_category_coverage(registry, bindings) == []


def test_materialize_demo_writes_hash_probe_and_preview(tmp_path: Path):
    video = tmp_path / "demo.mp4"
    _write_video(video)
    binding = {
        "source_table": "5-8-1",
        "task": "室内卫生清洁",
        "selection_id": "VLA82-002",
        "source_kind": "public_human_demonstration",
        "dataset": "synthetic-public-human-fixture",
        "clip_id": "fixture-001",
        "video": str(video),
        "start_seconds": 0.0,
        "stop_seconds": 0.8,
        "semantic_operation": "clean and place an object",
    }

    report = materialize_demo(binding, tmp_path / "bundle")

    saved = json.loads((tmp_path / "bundle/source_demo.json").read_text("utf-8"))
    assert report == saved
    assert len(saved["sha256"]) == 64
    assert saved["video_probe"]["frame_count"] == 5
    assert saved["video_probe"]["width"] == 48
    assert saved["video_probe"]["height"] == 32
    assert (tmp_path / "bundle/source_demo_preview.png").is_file()


def test_registry_object_demo_preserves_nonhuman_source_kind(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_video(source / "rgb_head.mp4")
    (source / "source_manifest.json").write_text(
        json.dumps({"dataset": "BEHAVIOR-1K 2025 Challenge"}), encoding="utf-8"
    )
    selection = {
        "selection_id": "VLA82-001",
        "source_table": "5-8-1",
        "task": "室内卫生清洁",
        "object": "垃圾桶",
        "operation_label": "垃圾投放",
        "data_path": str(source),
        "video_files": ["rgb_head.mp4"],
    }

    report = materialize_registry_object_demo(selection, tmp_path / "object_bundle")

    assert report["source_kind"] == "public_robot_or_simulator_demonstration"
    assert report["object"] == "垃圾桶"
    assert (tmp_path / "object_bundle/source_demo.json").is_file()
