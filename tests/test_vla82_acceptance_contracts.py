from __future__ import annotations

import json
from pathlib import Path

from tools.vla82_acceptance.contracts import (
    REQUIRED_EVIDENCE,
    build_acceptance_summary,
    validate_item_report,
)


def _write_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_EVIDENCE:
        target = root / name
        if target.suffix == ".json":
            target.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        else:
            target.write_bytes(b"evidence")


def _valid_report() -> dict:
    return {
        "selection_id": "VLA82-001",
        "status": "PASS",
        "validation_kind": "hybrid_visual_imitation_simulation",
        "asset": {"exact_class": True, "kind": "digital_twin"},
        "recognition": {"success": True},
        "operation": {"success": True},
        "trajectory": [{"executed_action": [0.0] * 7}],
    }


def test_smoke_or_replay_cannot_count_as_complete(tmp_path: Path):
    _write_bundle(tmp_path)
    report = _valid_report()
    report["validation_kind"] = "interface_smoke"

    errors = validate_item_report(report, tmp_path)

    assert "unsupported_validation_kind" in errors


def test_exact_asset_recognition_operation_and_nonempty_trajectory_are_required(
    tmp_path: Path,
):
    _write_bundle(tmp_path)
    report = _valid_report()
    report["asset"]["exact_class"] = False
    report["recognition"]["success"] = False
    report["operation"]["success"] = False
    report["trajectory"] = []

    errors = validate_item_report(report, tmp_path)

    assert "inexact_asset" in errors
    assert "recognition_failed" in errors
    assert "operation_failed" in errors
    assert "empty_trajectory" in errors


def test_missing_evidence_is_reported_by_name(tmp_path: Path):
    _write_bundle(tmp_path)
    (tmp_path / "rollout.mp4").unlink()

    errors = validate_item_report(_valid_report(), tmp_path)

    assert "missing_evidence:rollout.mp4" in errors


def test_valid_item_bundle_has_no_contract_errors(tmp_path: Path):
    _write_bundle(tmp_path)

    assert validate_item_report(_valid_report(), tmp_path) == []


def test_final_gate_requires_exact_counts():
    summary = build_acceptance_summary([], [])

    assert summary["overall_status"] == "FAIL"
    assert summary["task_learning_pass_count"] == 0
    assert summary["object_operation_pass_count"] == 0
    assert summary["recognition_pass_count"] == 0


def test_final_gate_passes_only_for_8_tasks_and_60_objects():
    tasks = [{"acceptance_status": "PASS"} for _ in range(8)]
    objects = [
        {"acceptance_status": "PASS", "recognition": {"success": True}}
        for _ in range(60)
    ]

    summary = build_acceptance_summary(tasks, objects)

    assert summary["overall_status"] == "PASS"
    assert summary["task_learning_pass_count"] == 8
    assert summary["object_operation_pass_count"] == 60
    assert summary["recognition_pass_count"] == 60
