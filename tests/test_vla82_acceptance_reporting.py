from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tools.vla82_acceptance.reporting import (
    aggregate,
    artifact_entry,
    draw_contact_sheet,
)


def _report(selection_id: str, *, task: bool = False) -> dict:
    return {
        "selection_id": selection_id,
        "task": "示例任务",
        "object": "示例物体",
        "operation_label": "示例操作",
        "acceptance_status": "PASS",
        "recognition": {"success": True},
        "operation": {"success": True},
        "bundle_kind": "task_learning" if task else "object_operation",
    }


def test_aggregate_rejects_59_object_reports():
    tasks = [_report(f"TASK-{index:02d}", task=True) for index in range(8)]
    objects = [_report(f"VLA82-{index:03d}") for index in range(1, 60)]

    manifest = aggregate(tasks, objects)

    assert manifest["overall_status"] == "FAIL"
    assert manifest["object_operation_pass_count"] == 59


def test_aggregate_passes_exactly_8_tasks_and_60_objects():
    tasks = [_report(f"TASK-{index:02d}", task=True) for index in range(8)]
    objects = [_report(f"VLA82-{index:03d}") for index in range(1, 61)]

    manifest = aggregate(tasks, objects)

    assert manifest["overall_status"] == "PASS"
    assert manifest["recognition_pass_count"] == 60


def test_artifact_entry_records_real_path_size_and_hash(tmp_path: Path):
    artifact = tmp_path / "evidence.bin"
    artifact.write_bytes(b"verified-evidence")

    entry = artifact_entry(artifact)

    assert entry["path"] == str(artifact.resolve())
    assert entry["size"] == len(b"verified-evidence")
    assert len(entry["sha256"]) == 64


def test_contact_sheet_uses_real_first_and_last_frames(tmp_path: Path):
    bundles = []
    for index in range(2):
        bundle = tmp_path / f"VLA82-{index + 1:03d}"
        bundle.mkdir()
        cv2.imwrite(
            str(bundle / "first_frame.png"),
            np.full((48, 64, 3), 40 + index * 20, dtype=np.uint8),
        )
        cv2.imwrite(
            str(bundle / "last_frame.png"),
            np.full((48, 64, 3), 160 + index * 20, dtype=np.uint8),
        )
        report = _report(bundle.name)
        report["bundle_dir"] = str(bundle)
        bundles.append(report)

    output = tmp_path / "contact.png"
    draw_contact_sheet(bundles, output)

    assert output.is_file()
    image = cv2.imread(str(output))
    assert image is not None
    assert image.shape[0] > 200
    assert json.dumps(bundles, ensure_ascii=False)
