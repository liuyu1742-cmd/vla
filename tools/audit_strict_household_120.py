"""Audit the final household folders: 32 network interfaces plus 88 VLA triples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit(dataset_root: Path) -> dict:
    records = []
    for task_dir in sorted(dataset_root.glob("chapter5_*")):
        if not task_dir.is_dir():
            continue
        for object_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            status_file = object_dir / "conversion_status.json"
            stub_file = object_dir / "archive_status.json"
            if status_file.is_file():
                status = json.loads(status_file.read_text(encoding="utf-8", errors="replace"))
                rgb = sorted(object_dir.glob("rgb_*.mp4"))
                actions = object_dir / "actions.parquet"
                instruction = object_dir / "instruction.json"
                action_ok = actions.is_file() and "action" in pd.read_parquet(actions).columns
                records.append({"task_dir": task_dir.name, "object": object_dir.name, "kind": "vla", "instruction": instruction.is_file(), "rgb": len(rgb), "actions": action_ok, "status": status.get("status")})
            elif stub_file.is_file():
                stub = json.loads(stub_file.read_text(encoding="utf-8", errors="replace"))
                records.append({"task_dir": task_dir.name, "object": object_dir.name, "kind": stub.get("archive_kind", "stub"), "instruction": False, "rgb": 0, "actions": False, "status": stub.get("conversion_status")})
            else:
                records.append({"task_dir": task_dir.name, "object": object_dir.name, "kind": "unknown", "instruction": False, "rgb": 0, "actions": False, "status": "missing_metadata"})
    vla = [row for row in records if row["kind"] == "vla"]
    network = [row for row in records if row["kind"] == "network_interface"]
    failures = [row for row in vla if not (row["instruction"] and row["rgb"] >= 1 and row["actions"])]
    other = [row for row in records if row["kind"] not in {"vla", "network_interface"}]
    return {"summary": {"object_count": len(records), "task_count": len({row["task_dir"] for row in records}), "vla_count": len(vla), "network_control_count": len(network), "invalid_vla_count": len(failures), "other_count": len(other)}, "invalid_vla": failures, "other": other, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
