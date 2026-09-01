"""Reset-audit all 60 VLA82 mappings in RoboCasa.

The batch never stops on an individual mapping failure. After three failed
attempts for the same mapping it records a comprehensive diagnostic block and
continues, matching the project's current failure-handling rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_smoke"
CAMERA_KEY = "video.robot0_agentview_left"
OBJECT_REGISTRIES = ("objaverse", "lightwheel", "aigen")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _generic_furniture_reset(
    mapping: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    import gymnasium as gym
    import imageio.v3 as iio
    import robocasa  # noqa: F401

    task_class = str(mapping["task_class"])
    environment = None
    try:
        environment = gym.make(
            f"robocasa/{task_class}",
            split="pretrain",
            seed=seed,
            obj_registries=OBJECT_REGISTRIES,
            camera_names=["robot0_agentview_left"],
            disable_env_checker=True,
        )
        observation, reset_info = environment.reset(seed=seed)
        raw = environment.unwrapped.env
        frame = np.asarray(observation[CAMERA_KEY], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError(f"invalid RGB frame shape: {frame.shape}")
        frame_path = output_dir / "reset_frame.png"
        iio.imwrite(frame_path, frame)

        task_success_at_reset = bool(raw._check_success())
        if task_success_at_reset:
            raise RuntimeError("task is already successful at reset")
        lower, upper = raw.action_spec
        action_shape = list(np.asarray(lower).shape)
        if action_shape != [12] or np.asarray(upper).shape != (12,):
            raise RuntimeError(
                f"expected native RoboCasa action shape [12], got {action_shape}"
            )
        instruction = str(raw.get_ep_meta().get("lang", "")).strip()
        if not instruction:
            raise RuntimeError("reset instruction is empty")
        return {
            "status": "PASS",
            "task_class": task_class,
            "object_group": mapping.get("object_group"),
            "seed": seed,
            "instruction": instruction,
            "source_instruction": mapping["operation_label"],
            "object_registries": list(OBJECT_REGISTRIES),
            "actual_object": None,
            "allowed_categories": [],
            "camera": {
                "key": CAMERA_KEY,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "frame": str(frame_path.resolve()),
            },
            "action_shape": action_shape,
            "task_success_at_reset": task_success_at_reset,
            "reset_info": {
                str(key): _json_ready(value) for key, value in reset_info.items()
            },
        }
    finally:
        if environment is not None:
            environment.close()


def _object_reset(
    mapping: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    from tools import audit_vla_midterm_simulator_mapping_v2 as object_auditor

    relation = {
        "task_class": mapping["task_class"],
        "object_group": mapping["object_group"],
        "manipulated_object_text": mapping["manipulated_object_text"],
        "example_instruction": mapping["operation_label"],
    }
    return object_auditor._one_reset(relation, seed=seed, output_dir=output_dir)


def _comprehensive_diagnosis(
    mapping: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.models.objects.kitchen_objects import OBJ_GROUPS

    error_signatures = Counter(
        f"{failure['error_type']}: {failure['error']}" for failure in failures
    )
    source_path = Path(mapping["source_data_path"])
    return {
        "trigger": "same mapping failed three attempts; comprehensive diagnosis",
        "environment_id": f"robocasa/{mapping['task_class']}",
        "environment_registered": (
            f"robocasa/{mapping['task_class']}" in gym.envs.registry
        ),
        "object_group": mapping.get("object_group"),
        "object_group_available": (
            mapping.get("object_group") is None
            or mapping["object_group"] in OBJ_GROUPS
        ),
        "source_data_path": str(source_path),
        "source_data_path_exists": source_path.is_dir(),
        "source_video_file_count": mapping["source_video_file_count"],
        "mapping_mode": mapping["mapping_mode"],
        "mapping_note": mapping["mapping_note"],
        "error_signature_counts": dict(error_signatures),
        "next_action": (
            "batch continues; failed mapping remains queued for root-cause fix and rerun"
        ),
    }


def audit_mapping(
    mapping: dict[str, Any],
    *,
    output_root: Path,
    attempts: int,
) -> dict[str, Any]:
    relation_dir = output_root / (
        f"{mapping['selection_id']}_{mapping['task_class']}_"
        f"{mapping.get('object_group') or 'furniture'}"
    )
    relation_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    report: dict[str, Any]
    for attempt in range(1, attempts + 1):
        seed = 820000 + int(mapping["selection_id"].split("-")[-1]) * 10 + attempt
        try:
            if mapping.get("object_group"):
                report = _object_reset(mapping, seed=seed, output_dir=relation_dir)
            else:
                report = _generic_furniture_reset(
                    mapping,
                    seed=seed,
                    output_dir=relation_dir,
                )
            report["attempt"] = attempt
            report["prior_failures"] = failures
            break
        except Exception as error:  # retain evidence and continue retrying
            failures.append(
                {
                    "attempt": attempt,
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    else:
        report = {
            "status": "FAIL",
            "task_class": mapping["task_class"],
            "object_group": mapping.get("object_group"),
            "attempts": attempts,
            "failures": failures,
            "comprehensive_diagnosis": _comprehensive_diagnosis(mapping, failures),
        }

    report.update(
        {
            "selection_id": mapping["selection_id"],
            "source_table": mapping["source_table"],
            "source_ordinal": mapping["source_ordinal"],
            "task": mapping["task"],
            "object": mapping["object"],
            "operation_label": mapping["operation_label"],
            "mapping_mode": mapping["mapping_mode"],
            "mapping_note": mapping["mapping_note"],
            "proxy_disclosed": mapping["proxy_disclosed"],
        }
    )
    report_path = relation_dir / "report.json"
    report["report_path"] = str(report_path.resolve())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _manifest(
    plan: dict[str, Any],
    plan_path: Path,
    results: list[dict[str, Any]],
    *,
    attempts: int,
) -> dict[str, Any]:
    return {
        "schema_version": "vla82_simulator_smoke_v1",
        "mapping_plan": str(plan_path.resolve()),
        "task_count": plan["source_scope"]["task_count"],
        "object_count": plan["source_scope"]["object_count"],
        "attempts_per_failed_mapping": attempts,
        "completed_count": len(results),
        "pass_count": sum(result.get("status") == "PASS" for result in results),
        "fail_count": sum(result.get("status") != "PASS" for result in results),
        "all_passed": (
            len(results) == len(plan["mappings"])
            and all(result.get("status") == "PASS" for result in results)
        ),
        "mapping_mode_counts": plan["mode_counts"],
        "results": results,
    }


def run_smoke(
    plan_path: Path,
    output_root: Path,
    *,
    attempts: int,
    resume: bool,
    limit: int | None,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mappings = plan["mappings"][:limit] if limit else plan["mappings"]
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, mapping in enumerate(mappings, start=1):
        relation_dir = output_root / (
            f"{mapping['selection_id']}_{mapping['task_class']}_"
            f"{mapping.get('object_group') or 'furniture'}"
        )
        report_path = relation_dir / "report.json"
        previous = None
        if resume and report_path.is_file():
            try:
                previous = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = None
        if previous and previous.get("status") == "PASS":
            result = previous
            action = "SKIP PASS"
        else:
            result = audit_mapping(
                mapping,
                output_root=output_root,
                attempts=attempts,
            )
            action = result["status"]
        results.append(result)
        print(
            f"[{index:02d}/{len(mappings)}] {mapping['selection_id']} "
            f"{mapping['task_class']}::{mapping.get('object_group') or 'furniture'} "
            f"{action}",
            flush=True,
        )
        manifest = _manifest(
            plan,
            plan_path,
            results,
            attempts=attempts,
        )
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    manifest = run_smoke(
        arguments.plan,
        arguments.output,
        attempts=arguments.attempts,
        resume=not arguments.no_resume,
        limit=arguments.limit,
    )
    summary = {
        key: manifest[key]
        for key in (
            "task_count",
            "object_count",
            "completed_count",
            "pass_count",
            "fail_count",
            "all_passed",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print((arguments.output / "manifest.json").resolve(), flush=True)
    expected_count = (
        arguments.limit
        if arguments.limit
        else manifest["object_count"]
    )
    return 0 if manifest["pass_count"] == expected_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
