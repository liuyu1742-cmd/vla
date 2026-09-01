"""Data contracts shared by skill-coverage builders and validators."""

from __future__ import annotations

from enum import Enum
from typing import Any


class Applicability(str, Enum):
    DIRECT = "direct"
    DEVICE_CONTROL = "device_control"
    COMPOSITE_RESOURCE = "composite_resource"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


class ValidationState(str, Enum):
    UNTESTED = "untested"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


def _require_text(record: dict[str, Any], key: str) -> None:
    if not isinstance(record.get(key), str) or not record[key].strip():
        raise ValueError(f"{key} is required")


def validate_object_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one normalized candidate object record."""
    _require_text(record, "object_id")
    _require_text(record, "display_name")
    if not record.get("dataset_task_groups"):
        raise ValueError("dataset_task_groups are required")
    if not record.get("source_rows"):
        raise ValueError("source_rows are required")
    return record


def validate_matrix_entry(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one explicit service-task/object applicability decision."""
    _require_text(record, "service_task_id")
    _require_text(record, "object_id")
    applicability = Applicability(record["applicability"])
    validation = ValidationState(record["validation_state"])
    if validation is ValidationState.PASSED and applicability in {
        Applicability.NOT_APPLICABLE,
        Applicability.NEEDS_REVIEW,
    }:
        raise ValueError(f"{applicability.value} pair cannot be passed")
    return record
