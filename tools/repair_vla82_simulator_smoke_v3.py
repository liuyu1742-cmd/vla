"""Rerun VLA82-044 with the disclosed bowl functional proxy."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
OUTPUT_ROOT = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_smoke_repair_v3"
SEEDS = [730221, 910441, 820441, 731041, 731051]


def _pick_place_class(task_class: str) -> type:
    from robocasa.environments.kitchen.atomic import kitchen_pick_place

    environment_class = getattr(kitchen_pick_place, task_class, None)
    if not isinstance(environment_class, type):
        raise ValueError(f"unavailable pick-place class: {task_class}")
    return environment_class


def _stable_distractor_wrapper(environment_class: type, group: str = "apple"):
    original = environment_class._get_obj_cfgs

    def stable_get_obj_cfgs(environment: Any):
        configs = original(environment)
        for config in configs:
            if config.get("name") != "obj":
                config["obj_groups"] = group
        return configs

    environment_class._get_obj_cfgs = stable_get_obj_cfgs
    return original


def main() -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    from tools import audit_vla82_simulator_smoke as implementation

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    mapping = next(
        row for row in plan["mappings"] if row["selection_id"] == "VLA82-044"
    )
    output_dir = OUTPUT_ROOT / (
        f"{mapping['selection_id']}_{mapping['task_class']}_{mapping['object_group']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    for attempt, seed in enumerate(SEEDS, start=1):
        environment_class = _pick_place_class(mapping["task_class"])
        original = _stable_distractor_wrapper(environment_class)
        try:
            result = implementation._object_reset(mapping, seed=seed, output_dir=output_dir)
            result.update(
                {
                    "status": "PASS",
                    "selection_id": mapping["selection_id"],
                    "source_table": mapping["source_table"],
                    "source_ordinal": mapping["source_ordinal"],
                    "task": mapping["task"],
                    "object": mapping["object"],
                    "operation_label": mapping["operation_label"],
                    "task_class": mapping["task_class"],
                    "object_group": mapping["object_group"],
                    "mapping_mode": mapping["mapping_mode"],
                    "mapping_note": mapping["mapping_note"],
                    "proxy_disclosed": mapping["proxy_disclosed"],
                    "repair_attempt": attempt,
                    "repair_strategy": (
                        "use disclosed bowl functional proxy and force only "
                        "non-target distractors to stable apple group"
                    ),
                    "prior_failures": failures,
                }
            )
            break
        except Exception as error:
            failures.append(
                {
                    "attempt": attempt,
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            environment_class._get_obj_cfgs = original

    if result is None:
        result = {
            "status": "FAIL",
            **mapping,
            "failures": failures,
            "comprehensive_diagnosis": {
                "trigger": "all deterministic bowl-proxy seeds failed",
                "next_action": "continue compatible functional-proxy review",
            },
        }

    report_path = output_dir / "report.json"
    result["report_path"] = str(report_path.resolve())
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "vla82_simulator_smoke_repair_v3",
        "mapping_plan": str(PLAN_PATH.resolve()),
        "completed_count": 1,
        "pass_count": int(result["status"] == "PASS"),
        "fail_count": int(result["status"] != "PASS"),
        "all_passed": result["status"] == "PASS",
        "results": [result],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"VLA82-044 {result['status']}", flush=True)
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
