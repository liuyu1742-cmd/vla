"""Build the shared Task-2 relation/action contract without mutating its sources."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .registry_builder import build_source_registries


IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
ACTION_CALL = re.compile(r"^([a-z][a-z0-9_]*)\(([^()]*)\)$")


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"benchmark {key} is required")
    return value.strip()


def _identifier(value: str, label: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def relation_key(task_id: str, object_id: str) -> str:
    """Return the stable shared key for one exact task/object relation."""
    return f"{_identifier(task_id, 'task_id')}::{_identifier(object_id, 'object_id')}"


def parse_action_call(action: str) -> tuple[str, tuple[str, ...]]:
    """Parse one canonical ``verb(arg,...)`` action without interpreting it."""
    if not isinstance(action, str):
        raise ValueError("action call must be text")
    match = ACTION_CALL.fullmatch(action.strip())
    if match is None:
        raise ValueError(f"invalid action call: {action!r}")
    verb = match.group(1)
    arguments_text = match.group(2).strip()
    if not arguments_text:
        return verb, ()
    arguments = tuple(part.strip() for part in arguments_text.split(","))
    if any(not IDENTIFIER.fullmatch(argument) for argument in arguments):
        raise ValueError(f"invalid action call arguments: {action!r}")
    return verb, arguments


def validate_benchmark_record(record: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize one exported Task-2 benchmark relation."""
    task_id = _identifier(_text(record, "acceptance_task"), "acceptance_task")
    object_id = _identifier(_text(record, "object"), "object")
    expected_key = relation_key(task_id, object_id)
    actual_key = _text(record, "relation_key")
    if actual_key != expected_key:
        raise ValueError(
            f"relation_key mismatch: expected {expected_key!r}, found {actual_key!r}"
        )

    actions = record.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"actions must be a non-empty ordered list for {actual_key}")
    normalized_actions: list[str] = []
    for action in actions:
        if not isinstance(action, str):
            raise ValueError(f"actions must contain text for {actual_key}")
        normalized = action.strip()
        parse_action_call(normalized)
        normalized_actions.append(normalized)

    target = record.get("target", "none")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"target must be text for {actual_key}")
    legacy_task_id = record.get("legacy_task_id")
    if legacy_task_id is not None and (
        not isinstance(legacy_task_id, str)
        or not IDENTIFIER.fullmatch(legacy_task_id)
    ):
        raise ValueError(f"invalid legacy_task_id for {actual_key}")

    return {
        "benchmark_id": _text(record, "id"),
        "relation_key": actual_key,
        "task_id": task_id,
        "object_id": object_id,
        "instruction": _text(record, "instruction"),
        "target": target.strip(),
        "canonical_actions": normalized_actions,
        "legacy_task_id": legacy_task_id,
    }


def build_task2_contract(
    excel_rows: list[dict[str, str]],
    benchmark_records: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Join the Excel relation scope with the exact exported canonical actions."""
    registries = build_source_registries(excel_rows)
    source_relations = registries["source_relations"]
    source_by_key = {
        relation_key(item["dataset_task_id"], item["object_id"]): item
        for item in source_relations
    }

    benchmark_by_key: dict[str, dict[str, object]] = {}
    for raw_record in benchmark_records:
        record = validate_benchmark_record(raw_record)
        key = str(record["relation_key"])
        if key in benchmark_by_key:
            raise ValueError(f"duplicate benchmark relation: {key}")
        benchmark_by_key[key] = record

    excel_keys = set(source_by_key)
    benchmark_keys = set(benchmark_by_key)
    if excel_keys != benchmark_keys:
        excel_only = sorted(excel_keys - benchmark_keys)
        benchmark_only = sorted(benchmark_keys - excel_keys)
        raise ValueError(
            "Task-2 relation mismatch: "
            f"excel-only={excel_only}, benchmark-only={benchmark_only}"
        )

    contract: list[dict[str, Any]] = []
    for source in source_relations:
        key = relation_key(source["dataset_task_id"], source["object_id"])
        benchmark = benchmark_by_key[key]
        contract.append(
            {
                "relation_key": key,
                "task_id": source["dataset_task_id"],
                "task_name": source["dataset_task_name"],
                "object_id": source["object_id"],
                "object_name": source["display_name"],
                "source_row": source["source_row"],
                "object_dir": source["object_dir"],
                "media_counts": {
                    name: source[name]
                    for name in ("images", "videos", "annotations", "metadata")
                },
                "benchmark_id": benchmark["benchmark_id"],
                "instruction": benchmark["instruction"],
                "target": benchmark["target"],
                "canonical_actions": benchmark["canonical_actions"],
                "legacy_task_id": benchmark["legacy_task_id"],
            }
        )
    return contract
