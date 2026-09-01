"""Validation helpers for ``household_skill_ir_v2`` records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tools.skill_coverage.task2_contract import relation_key


SCHEMA_VERSION = "household_skill_ir_v2"
RELATION_FIELDS = (
    "task_id",
    "task_name",
    "object_id",
    "object_name",
    "instruction",
    "target",
)
EVIDENCE_FIELDS = (
    "source_video",
    "perception_evidence",
    "parser_evidence",
    "execution_request",
    "execution_evidence",
)


def _flatten_numbers(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers: list[float] = []
        for item in value:
            numbers.extend(_flatten_numbers(item))
        return numbers
    return []


def validate_relation_record(record: Mapping[str, object]) -> dict[str, Any]:
    key = record.get("relation_key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("relation_key is required")
    for field in RELATION_FIELDS:
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            raise ValueError(f"{field} is required for {key}")
    expected_key = relation_key(str(record["task_id"]), str(record["object_id"]))
    if key != expected_key:
        raise ValueError(
            f"relation_key mismatch: expected {expected_key!r}, found {key!r}"
        )
    actions = record.get("canonical_actions")
    if not isinstance(actions, list) or not actions or not all(
        isinstance(action, str) and action.strip() for action in actions
    ):
        raise ValueError(f"canonical_actions are required for {key}")
    return dict(record)


def validate_skill_ir_record(
    skill_ir: Mapping[str, object],
    expected: Mapping[str, object],
) -> dict[str, Any]:
    if skill_ir.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Skill IR schema_version")
    if not isinstance(skill_ir.get("contract_version"), str) or not str(
        skill_ir["contract_version"]
    ).strip():
        raise ValueError("contract_version is required")

    relation = validate_relation_record(skill_ir)
    expected_relation = validate_relation_record(expected)
    if relation["relation_key"] != expected_relation["relation_key"]:
        raise ValueError("Skill IR relation_key does not match its contract record")
    for field in RELATION_FIELDS:
        if relation[field] != expected_relation[field]:
            raise ValueError(f"Skill IR {field} differs from the Task-2 contract")
    if relation["canonical_actions"] != expected_relation["canonical_actions"]:
        raise ValueError("Skill IR canonical_actions differ from the Task-2 contract")

    for field in EVIDENCE_FIELDS:
        value = skill_ir.get(field)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{field} must be null or an object")

    perception = skill_ir.get("perception_evidence")
    if isinstance(perception, Mapping) and "keypoints" in perception:
        keypoints = perception["keypoints"]
        numbers = _flatten_numbers(keypoints)
        if numbers and all(number == 0.0 for number in numbers):
            raise ValueError("all-zero keypoints cannot represent perception evidence")

    execution = skill_ir.get("execution_evidence")
    if isinstance(execution, Mapping) and execution.get("success") is True:
        expected_progress = len(expected_relation["canonical_actions"])
        if execution.get("canonical_action_progress") != expected_progress:
            raise ValueError(
                "successful execution canonical_action_progress must equal "
                f"{expected_progress}"
            )
        predicates = execution.get("success_predicates")
        if not isinstance(predicates, Mapping) or not predicates:
            raise ValueError("successful execution requires success_predicates")
        if not all(value is True for value in predicates.values()):
            raise ValueError("all successful execution predicates must be true")
    return dict(skill_ir)
