"""Synchronize the completed 600-case inference batch into formal registries."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "midterm_testing_vla82"
REGISTRY_PATH = BASE / "vla82_midterm_registry.json"
RUNS_PATH = BASE / "vla82_midterm_600_runs.csv"
RELATIONS_PATH = BASE / "vla82_midterm_relations.csv"
BATCH_PATH = BASE / "openvla_600_augmentation_batch" / "manifest.json"


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    batch = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
    if not batch["all_passed"] or batch["pass_count"] != 600:
        raise ValueError("600-case batch is not fully passing")
    batch_by_run = {row["run_id"]: row for row in batch["results"]}

    with RUNS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        runs = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if len(runs) != 600:
        raise ValueError(f"expected 600 formal runs, got {len(runs)}")
    for row in runs:
        result = batch_by_run[row["run_id"]]
        row["stage"] = "POLICY_AUGMENTATION_COMPLETE"
        row["environment_reset"] = "PASS_SELECTION_SMOKE"
        row["policy_inference"] = "PASS"
        row["success"] = "PASS_POLICY_INFERENCE"
        row["failure_type"] = ""
        row["evidence_path"] = str(BATCH_PATH.resolve())
        row["notes"] = (
            f"视觉条件={result['augmentation']}；真实OpenVLA输出7维有限动作；"
            "环境证据引用60项接口冒烟；非物理闭环成功"
        )
    with RUNS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for row in registry["selected_objects"]:
        row["batch_test_status"] = "PASS_POLICY_INFERENCE_10X"
        row["closed_loop_test_status"] = "NOT_EXECUTED"
    summary = registry.setdefault("verification_summary", {})
    summary.update(
        {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "openvla_augmentation_600_pass_count": 600,
            "openvla_augmentation_600_fail_count": 0,
            "closed_loop_600_completed_count": 0,
            "closed_loop_success_claimed": False,
            "test_scope_completed": "policy_inference_visual_augmentation",
        }
    )
    summary.setdefault("artifacts", {})["openvla_augmentation_600"] = str(
        BATCH_PATH.resolve()
    )
    write_json(REGISTRY_PATH, registry)

    with RELATIONS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        relations = list(reader)
        relation_fields = list(reader.fieldnames or [])
    if "closed_loop_test_status" not in relation_fields:
        relation_fields.append("closed_loop_test_status")
    for row in relations:
        row["batch_test_status"] = "PASS_POLICY_INFERENCE_10X"
        row["closed_loop_test_status"] = "NOT_EXECUTED"
    with RELATIONS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=relation_fields)
        writer.writeheader()
        writer.writerows(relations)

    print("UPDATED: formal 600-case policy-inference status (600/600 PASS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
