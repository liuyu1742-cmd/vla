"""Merge initial and repaired VLA82 smoke results into one authoritative manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "midterm_testing_vla82"
PLAN_PATH = BASE / "simulator_mapping_plan.json"
OUTPUT_PATH = BASE / "simulator_smoke_final" / "manifest.json"
SOURCE_MANIFESTS = [
    BASE / "simulator_smoke" / "manifest.json",
    BASE / "simulator_smoke_shard1" / "manifest.json",
    BASE / "simulator_smoke_shard2" / "manifest.json",
    BASE / "simulator_smoke_shard3" / "manifest.json",
]
REPAIR_MANIFESTS = [
    BASE / "simulator_smoke_repair_v2" / "manifest.json",
    BASE / "simulator_smoke_repair_v3" / "manifest.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = read_json(PLAN_PATH)
    mappings = {row["selection_id"]: row for row in plan["mappings"]}
    by_id: dict[str, dict[str, Any]] = {}

    for path in SOURCE_MANIFESTS:
        for result in read_json(path)["results"]:
            by_id[result["selection_id"]] = result
    for path in REPAIR_MANIFESTS:
        for result in read_json(path)["results"]:
            if result["status"] == "PASS":
                by_id[result["selection_id"]] = result

    expected_ids = [f"VLA82-{index:03d}" for index in range(1, 61)]
    missing = [selection_id for selection_id in expected_ids if selection_id not in by_id]
    if missing:
        raise ValueError(f"missing smoke results: {missing}")

    results = []
    for selection_id in expected_ids:
        result = dict(by_id[selection_id])
        mapping = mappings[selection_id]
        result.update(
            {
                "selection_id": selection_id,
                "source_table": mapping["source_table"],
                "source_ordinal": mapping["source_ordinal"],
                "task": mapping["task"],
                "object": mapping["object"],
                "operation_label": mapping["operation_label"],
                "task_class": mapping["task_class"],
                "object_group": mapping.get("object_group"),
                "mapping_mode": mapping["mapping_mode"],
                "mapping_note": mapping["mapping_note"],
                "proxy_disclosed": mapping["proxy_disclosed"],
                "status": "PASS",
            }
        )
        results.append(result)

    manifest = {
        "schema_version": "vla82_simulator_smoke_final_v1",
        "mapping_plan": str(PLAN_PATH.resolve()),
        "source_scope": {
            "task_count": 8,
            "object_count": 60,
            "candidate_object_count": 82,
            "table_5_8_3_included": False,
        },
        "validation_scope": "environment_reset_camera_action_interface_smoke",
        "validation_scope_zh": "环境重置、相机观测与动作接口冒烟验证",
        "completed_count": len(results),
        "pass_count": sum(row["status"] == "PASS" for row in results),
        "fail_count": sum(row["status"] != "PASS" for row in results),
        "all_passed": all(row["status"] == "PASS" for row in results),
        "mode_counts": plan["mode_counts"],
        "closed_loop_vla_600_runs_completed": False,
        "actual_object_closed_loop_claimed_for_proxies": False,
        "proxy_disclaimer": (
            "功能代理只证明RoboCasa任务族、环境重置、观测和动作接口可运行；"
            "不等同于真实物体VLA闭环操作成功。真实物体证据仍来自工作簿限定范围内的视频与标注。"
        ),
        "repair_history": [
            {
                "stage": "initial",
                "pass_count": 52,
                "fail_count": 8,
                "failed_ids": [
                    "VLA82-008",
                    "VLA82-012",
                    "VLA82-025",
                    "VLA82-033",
                    "VLA82-034",
                    "VLA82-043",
                    "VLA82-044",
                    "VLA82-059",
                ],
            },
            {
                "stage": "repair_v2",
                "pass_count": 7,
                "fail_count": 1,
                "remaining_failed_ids": ["VLA82-044"],
            },
            {
                "stage": "repair_v3",
                "pass_count": 1,
                "fail_count": 0,
                "repaired_ids": ["VLA82-044"],
            },
        ],
        "source_manifests": [
            str(path.resolve()) for path in SOURCE_MANIFESTS + REPAIR_MANIFESTS
        ],
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"WROTE: {OUTPUT_PATH} "
        f"({manifest['pass_count']}/{manifest['completed_count']} PASS)"
    )
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
