"""Evidence contracts that prevent partial checks from becoming acceptance passes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_EVIDENCE = (
    "source_demo.json",
    "source_demo_preview.png",
    "skill_ir.json",
    "recognition.json",
    "rollout.mp4",
    "first_frame.png",
    "last_frame.png",
    "report.json",
)
ALLOWED_VALIDATION_KINDS = frozenset({"hybrid_visual_imitation_simulation"})


def validate_item_report(report: Mapping[str, Any], root: Path) -> list[str]:
    """Return stable error codes for every unsatisfied item-level requirement."""
    errors: list[str] = []
    if report.get("validation_kind") not in ALLOWED_VALIDATION_KINDS:
        errors.append("unsupported_validation_kind")
    if report.get("asset", {}).get("exact_class") is not True:
        errors.append("inexact_asset")
    if report.get("recognition", {}).get("success") is not True:
        errors.append("recognition_failed")
    if report.get("operation", {}).get("success") is not True:
        errors.append("operation_failed")
    trajectory = report.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        errors.append("empty_trajectory")
    for name in REQUIRED_EVIDENCE:
        if not (root / name).is_file():
            errors.append(f"missing_evidence:{name}")
    return errors


def build_acceptance_summary(
    task_reports: Sequence[Mapping[str, Any]],
    object_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable 8/60/60 aggregate acceptance gate."""
    task_pass = sum(
        report.get("acceptance_status") == "PASS" for report in task_reports
    )
    object_pass = sum(
        report.get("acceptance_status") == "PASS" for report in object_reports
    )
    recognition_pass = sum(
        report.get("recognition", {}).get("success") is True
        for report in object_reports
    )
    overall = (
        len(task_reports) == 8
        and task_pass == 8
        and len(object_reports) == 60
        and object_pass == 60
        and recognition_pass == 60
    )
    return {
        "task_learning_expected": 8,
        "task_learning_completed_count": len(task_reports),
        "task_learning_pass_count": task_pass,
        "object_operation_expected": 60,
        "object_operation_completed_count": len(object_reports),
        "object_operation_pass_count": object_pass,
        "recognition_expected": 60,
        "recognition_pass_count": recognition_pass,
        "overall_status": "PASS" if overall else "FAIL",
    }
