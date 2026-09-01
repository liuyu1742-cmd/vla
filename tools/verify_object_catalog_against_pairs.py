"""Verify candidate household objects against locally paired robot demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify_object_row(row: dict[str, Any], paired_by_task: dict[int, dict[str, Any]]) -> bool:
    """Return true only for an exact source phrase in a task with real pairs."""
    task = paired_by_task.get(int(row["source_task_index"]))
    if not task or int(task.get("pair_count", 0)) < 1:
        return False
    phrase = str(row.get("source_phrase", "")).strip().casefold()
    return bool(phrase and phrase in str(task.get("instruction", "")).casefold())


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an object catalogue against local paired BEHAVIOR demos.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.catalog.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    paired_by_task = {
        int(record["task_index"]): record
        for record in registry
        if record["dataset"] == "BEHAVIOR-1K 2025 Challenge" and record["pair_count"] > 0
    }
    verified, rejected = [], []
    for row in rows:
        enriched = dict(row)
        task = paired_by_task.get(int(row["source_task_index"]))
        enriched["paired_episode_count"] = int(task["pair_count"]) if task else 0
        enriched["source_instruction"] = task["instruction"] if task else None
        enriched["rgb_status"] = bool(task and task["pair_count"])
        enriched["action_status"] = bool(task and task["pair_count"])
        enriched["training_status"] = "conversion_required" if verify_object_row(row, paired_by_task) else "rejected"
        (verified if enriched["training_status"] != "rejected" else rejected).append(enriched)

    payload = {
        "criteria": "exact source phrase + paired local RGB MP4 + paired action Parquet",
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "verified": verified,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verified": len(verified), "rejected": len(rejected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
