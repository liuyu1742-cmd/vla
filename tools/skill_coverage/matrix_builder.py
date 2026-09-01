"""Build an explicit sparse service-task/object applicability matrix."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .contracts import Applicability, validate_matrix_entry


def build_task_object_matrix(
    service_tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate one explicit entry for every service-task/object pair."""
    task_ids = {item["service_task_id"] for item in service_tasks}
    if len(task_ids) != len(service_tasks):
        raise ValueError("duplicate service_task_id")

    primary = policy["primary_by_dataset_task"]
    secondary = policy["secondary_by_object"]
    entries: list[dict[str, Any]] = []

    for candidate in candidates:
        object_id = candidate["object_id"]
        decisions: dict[str, tuple[str, str]] = {}
        for dataset_group in candidate["dataset_task_groups"]:
            if dataset_group not in primary:
                raise ValueError(f"unmapped dataset task group: {dataset_group}")
            service_task_id = primary[dataset_group]
            if service_task_id not in task_ids:
                raise ValueError(f"unknown service task: {service_task_id}")
            decisions[service_task_id] = (
                Applicability.DIRECT.value,
                f"dataset_task:{dataset_group}",
            )

        for rule in secondary.get(object_id, []):
            service_task_id = rule["service_task_id"]
            if service_task_id not in task_ids:
                raise ValueError(f"unknown service task: {service_task_id}")
            applicability = Applicability(rule["applicability"]).value
            decisions[service_task_id] = (
                applicability,
                f"object_override:{object_id}",
            )

        for task in service_tasks:
            service_task_id = task["service_task_id"]
            applicability, basis = decisions.get(
                service_task_id,
                (Applicability.NOT_APPLICABLE.value, "default_not_applicable"),
            )
            entry = {
                "service_task_id": service_task_id,
                "object_id": object_id,
                "applicability": applicability,
                "mapping_basis": basis,
                "validation_state": "untested",
                "evidence": [],
            }
            validate_matrix_entry(entry)
            entries.append(entry)

    return sorted(
        entries,
        key=lambda item: (item["service_task_id"], item["object_id"]),
    )


def summarize_coverage(
    service_tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
    acceptance_target: int = 120,
) -> dict[str, Any]:
    """Keep semantic coverage separate from execution validation evidence."""
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in matrix:
        validate_matrix_entry(entry)
        by_object[entry["object_id"]].append(entry)

    applicable_values = {
        Applicability.DIRECT.value,
        Applicability.DEVICE_CONTROL.value,
        Applicability.COMPOSITE_RESOURCE.value,
    }
    semantically_mapped = {
        object_id
        for object_id, entries in by_object.items()
        if any(item["applicability"] in applicable_values for item in entries)
    }
    validated = {
        object_id
        for object_id, entries in by_object.items()
        if any(
            item["applicability"] in applicable_values
            and item["validation_state"] == "passed"
            for item in entries
        )
    }
    failed = {
        object_id
        for object_id, entries in by_object.items()
        if object_id not in validated
        and any(item["validation_state"] == "failed" for item in entries)
    }
    blocked = {
        object_id
        for object_id, entries in by_object.items()
        if object_id not in validated
        and any(item["validation_state"] == "blocked" for item in entries)
    }

    return {
        "service_task_count": len(service_tasks),
        "candidate_object_count": len(candidates),
        "matrix_entry_count": len(matrix),
        "semantically_mapped_object_count": len(semantically_mapped),
        "validated_object_count": len(validated),
        "failed_object_count": len(failed),
        "blocked_object_count": len(blocked),
        "acceptance_target": acceptance_target,
        "remaining_validation_gap": max(acceptance_target - len(validated), 0),
    }
