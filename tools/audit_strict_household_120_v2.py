"""Classify network-interface stubs correctly in the strict household audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.audit_strict_household_120 import audit


def corrected_audit(dataset_root: Path) -> dict:
    result = audit(dataset_root)
    for row in result["records"]:
        if row["kind"] != "stub":
            continue
        status_file = dataset_root / row["task_dir"] / row["object"] / "archive_status.json"
        status = json.loads(status_file.read_text(encoding="utf-8", errors="replace"))
        if status.get("status") == "network_interface" and status.get("not_a_vla_training_sample"):
            row["kind"] = "network_control"
            row["status"] = "network_interface"
    vla = [row for row in result["records"] if row["kind"] == "vla"]
    network = [row for row in result["records"] if row["kind"] == "network_control"]
    invalid = [row for row in vla if not (row["instruction"] and row["rgb"] >= 1 and row["actions"])]
    other = [row for row in result["records"] if row["kind"] not in {"vla", "network_control"}]
    result["summary"] = {"object_count": len(result["records"]), "task_count": len({row["task_dir"] for row in result["records"]}), "vla_count": len(vla), "network_control_count": len(network), "invalid_vla_count": len(invalid), "other_count": len(other)}
    result["invalid_vla"] = invalid
    result["other"] = other
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = corrected_audit(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
