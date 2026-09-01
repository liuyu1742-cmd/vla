"""Checkpoint and diagnose long-running 8×60 acceptance batches."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vla82_acceptance_run_state_v1"


def empty_run_state() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": None, "items": {}}


def load_run_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_run_state()
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported run-state schema: {state.get('schema_version')}")
    if not isinstance(state.get("items"), dict):
        raise ValueError("run-state items must be an object")
    return state


def ids_to_run(selection_ids: list[str], state: dict[str, Any]) -> list[str]:
    items = state.get("items", {})
    return [
        selection_id
        for selection_id in selection_ids
        if not (
            items.get(selection_id, {}).get("status") == "PASS"
            and items.get(selection_id, {}).get("evidence_valid") is True
        )
    ]


def retry_decision(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    recent = [item.get("root_cause") for item in attempts[-3:]]
    repeated = (
        len(recent) == 3 and recent[0] is not None and len(set(recent)) == 1
    )
    return {
        "action": "FULL_DIAGNOSIS_CONTINUE" if repeated else "RETRY",
        "stop_entire_test": False,
        "repeated_root_cause": recent[0] if repeated else None,
        "diagnostic_layers": (
            [
                "source_video",
                "recognition",
                "skill_ir",
                "asset",
                "openvla_service",
                "action_adapter",
                "simulator_state",
                "success_rule",
            ]
            if repeated
            else []
        ),
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def record_attempt(
    state_path: Path,
    *,
    selection_id: str,
    status: str,
    evidence_valid: bool,
    root_cause: str | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    state = load_run_state(state_path)
    now = datetime.now().astimezone().isoformat()
    item = state["items"].setdefault(selection_id, {"attempts": []})
    item["attempts"].append(
        {
            "timestamp": now,
            "status": status,
            "evidence_valid": bool(evidence_valid),
            "root_cause": root_cause,
            "details": details,
        }
    )
    item["status"] = status
    item["evidence_valid"] = bool(evidence_valid)
    item["last_root_cause"] = root_cause
    item["retry_decision"] = retry_decision(item["attempts"])
    state["updated_at"] = now
    _write_state(state_path, state)
    return state
