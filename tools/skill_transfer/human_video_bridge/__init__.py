"""Active RobotProject video bridge with legacy Task-2 ID migration."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_source = Path(__file__).resolve().parents[1] / "human_video_bridge.py"
_spec = importlib.util.spec_from_file_location(
    "tools.skill_transfer._human_video_bridge_legacy", _source
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load human video bridge implementation: {_source}")
_implementation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_implementation)


def resolve_task2_relation(
    mapping: Mapping[str, Any],
    contract: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if mapping.get("mapping_status") != "mapped":
        raise ValueError("video record is not mapped")
    task = str(mapping.get("task_id", "")).strip()
    obj = str(mapping.get("object", "")).strip()
    if not task or not obj:
        raise ValueError("mapped video record is missing task_id or object")

    relation_id = str(mapping.get("relation_id", "")).strip()
    if relation_id:
        candidates = [
            dict(record)
            for record in contract.values()
            if record.get("benchmark_id") == relation_id
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"video relation_id must resolve exactly once in Task-2 contract: {relation_id}"
            )
        expected = candidates[0]
        if str(expected.get("object_id")) != obj:
            raise ValueError("video relation_id object differs from Task-2 contract")
        accepted_tasks = {
            str(expected.get("task_id", "")),
            str(expected.get("legacy_task_id", "")),
        }
        accepted_tasks.discard("")
        if task not in accepted_tasks:
            raise ValueError("video relation_id task differs from Task-2 current and legacy IDs")
    else:
        key = f"{task}::{obj}"
        if key not in contract:
            raise ValueError(f"mapped video relation is not in Task-2 contract: {key}")
        expected = dict(contract[key])

    actions = mapping.get("actions")
    if not isinstance(actions, list) or actions != expected["canonical_actions"]:
        raise ValueError(
            f"mapped video actions differ from Task-2 relation {expected['relation_key']}"
        )
    return expected


_implementation.resolve_task2_relation = resolve_task2_relation
assess_robotproject_record = _implementation.assess_robotproject_record
audit_robotproject_video_bridge = _implementation.audit_robotproject_video_bridge
build_video_conditioned_skill_ir = _implementation.build_video_conditioned_skill_ir
main = _implementation.main


__all__ = [
    "assess_robotproject_record",
    "audit_robotproject_video_bridge",
    "build_video_conditioned_skill_ir",
    "main",
    "resolve_task2_relation",
]
