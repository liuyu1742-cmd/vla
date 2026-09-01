"""Run auditable, pure-OpenVLA closed-loop trials for eight VLA82 categories.

Every executed control decision comes directly from the local OpenVLA service.
No oracle, scripted waypoint, privileged simulator state, or recovery controller
is allowed to alter the action. Simulator state is read only to evaluate the
official RoboCasa success predicate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict
from tools.robocasa_action_scaling import scale_openvla_action


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "midterm_testing_vla82" / "pure_openvla_closed_loop_8"
CAMERA_KEY = "video.robot0_agentview_left"
OBJECT_REGISTRIES = ("objaverse", "lightwheel", "aigen")
REPRESENTATIVE_IDS = [
    "VLA82-002",
    "VLA82-005",
    "VLA82-017",
    "VLA82-021",
    "VLA82-030",
    "VLA82-037",
    "VLA82-050",
    "VLA82-056",
]
ATOMIC_MODULES = (
    "kitchen_pick_place",
    "kitchen_doors",
    "kitchen_drawer",
    "kitchen_microwave",
    "kitchen_oven",
    "kitchen_stove",
    "kitchen_toaster",
    "kitchen_toaster_oven",
    "kitchen_sink",
)


def adapt_policy_action(
    action: Sequence[float], *, translation_gain: float, rotation_gain: float
) -> list[float]:
    """Convert metric/radian deltas to RoboCasa's normalized OSC command."""
    return scale_openvla_action(
        action,
        translation_gain=translation_gain,
        rotation_gain=rotation_gain,
    )


def task_class_module_candidates(task_class: str) -> list[str]:
    """Return atomic modules searched for an official RoboCasa task class."""
    if not str(task_class).strip():
        raise ValueError("task_class must be non-empty")
    prefix = "robocasa.environments.kitchen.atomic."
    return [prefix + module for module in ATOMIC_MODULES]


def _resolve_environment_class(task_class: str) -> type:
    for module_name in task_class_module_candidates(task_class):
        module = importlib.import_module(module_name)
        candidate = getattr(module, task_class, None)
        if isinstance(candidate, type):
            return candidate
    raise ValueError(f"RoboCasa task class is unavailable: {task_class}")


def select_representatives(mappings: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the fixed, reviewable representative from each of eight tables."""
    by_id = {item["selection_id"]: item for item in mappings}
    missing = [selection_id for selection_id in REPRESENTATIVE_IDS if selection_id not in by_id]
    if missing:
        raise ValueError(f"mapping plan lacks representative IDs: {missing}")
    selected = [by_id[selection_id] for selection_id in REPRESENTATIVE_IDS]
    tables = [item["source_table"] for item in selected]
    if len(set(tables)) != len(REPRESENTATIVE_IDS):
        raise ValueError(f"representatives do not cover eight distinct categories: {tables}")
    return selected


def filter_selected_representatives(
    selected: Sequence[dict[str, Any]], selection_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    if not selection_ids:
        return list(selected)
    requested = list(dict.fromkeys(selection_ids))
    by_id = {item["selection_id"]: item for item in selected}
    missing = [selection_id for selection_id in requested if selection_id not in by_id]
    if missing:
        raise ValueError(f"requested IDs are not representatives: {missing}")
    return [by_id[selection_id] for selection_id in requested]


def _evidence_exists(result: dict[str, Any]) -> bool:
    required = ("video", "first_frame", "last_frame")
    return all(result.get(key) and Path(result[key]).is_file() for key in required)


def build_manifest(
    results: Sequence[dict[str, Any]], *, expected_count: int = 8
) -> dict[str, Any]:
    """Summarize closed-loop outcomes without equating execution with success."""
    completed = len(results)
    return {
        "schema_version": "vla82_pure_openvla_closed_loop_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "evaluation_kind": "pure_openvla_closed_loop",
        "pure_autonomous_vla": True,
        "expected_count": expected_count,
        "completed_count": completed,
        "success_count": sum(bool(item.get("success")) for item in results),
        "failure_count": sum(not bool(item.get("success")) for item in results),
        "execution_error_count": sum(item.get("status") == "ERROR" for item in results),
        "evidence_complete": (
            completed == expected_count
            and all(
                item.get("actual_closed_loop") is True
                and item.get("pure_autonomous_vla") is True
                and item.get("simulator_success_predicate_checked") is True
                and int(item.get("executed_decision_steps", 0)) > 0
                and int(item.get("executed_sim_steps", 0)) > 0
                and _evidence_exists(item)
                for item in results
            )
        ),
        "all_tasks_successful": (
            completed == expected_count and all(bool(item.get("success")) for item in results)
        ),
        "results": list(results),
    }


def stable_distractor_group(config: dict[str, Any]) -> str:
    """Choose a deterministic object compatible with fridge exclusion filters."""
    excluded = config.get("exclude_obj_groups") or ()
    if isinstance(excluded, str):
        excluded = (excluded,)
    if config.get("fridgable") or {"fruit", "vegetable"}.intersection(excluded):
        return "milk"
    return "apple"


@contextmanager
def _configured_environment(mapping: dict[str, Any], seed: int) -> Iterator[Any]:
    import gymnasium as gym
    import robocasa  # noqa: F401
    from tools import audit_vla_midterm_simulator_mapping_v1 as legacy

    task_class = str(mapping["task_class"])
    environment_class = _resolve_environment_class(task_class)
    original_get_obj_cfgs = environment_class._get_obj_cfgs
    object_group = mapping.get("object_group")

    if object_group:
        legacy.install_target_object_group(environment_class, str(object_group))
        targeted_get_obj_cfgs = environment_class._get_obj_cfgs

        def stable_object_configs(environment: Any):
            configs = targeted_get_obj_cfgs(environment)
            for config in configs:
                if config.get("name") != "obj":
                    config["obj_groups"] = stable_distractor_group(config)
            return configs

        environment_class._get_obj_cfgs = stable_object_configs
    else:

        def stable_furniture_configs(environment: Any):
            configs = original_get_obj_cfgs(environment)
            for config in configs:
                config["obj_groups"] = "apple"
            return configs

        environment_class._get_obj_cfgs = stable_furniture_configs

    environment = None
    try:
        make_args: dict[str, Any] = {
            "split": "pretrain",
            "seed": seed,
            "obj_registries": OBJECT_REGISTRIES,
            "camera_names": ["robot0_agentview_left"],
            "disable_env_checker": True,
        }
        if object_group:
            make_args["obj_groups"] = object_group
        environment = gym.make(f"robocasa/{task_class}", **make_args)
        yield environment
    finally:
        if environment is not None:
            environment.close()
        environment_class._get_obj_cfgs = original_get_obj_cfgs


def _frame(observation: dict[str, Any]) -> np.ndarray:
    frame = np.asarray(observation[CAMERA_KEY], dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f"invalid RGB observation: {frame.shape}")
    return frame.copy()


def run_trial(
    mapping: dict[str, Any],
    *,
    output_root: Path,
    host: str,
    port: int,
    max_decisions: int,
    action_repeat: int,
    seed: int,
    translation_gain: float,
    rotation_gain: float,
) -> dict[str, Any]:
    """Execute one pure policy trial and retain action/predicate evidence."""
    import imageio.v3 as iio

    trial_dir = output_root / mapping["selection_id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    current_path = trial_dir / "current_frame.png"
    first_path = trial_dir / "first_frame.png"
    last_path = trial_dir / "last_frame.png"
    video_path = trial_dir / "rollout.mp4"
    report_path = trial_dir / "report.json"
    frames: list[np.ndarray] = []
    trajectory: list[dict[str, Any]] = []
    inference_seconds: list[float] = []
    success = False
    simulator_steps = 0

    with _configured_environment(mapping, seed) as environment:
        observation, reset_info = environment.reset(seed=seed)
        raw = environment.unwrapped.env
        instruction = str(raw.get_ep_meta().get("lang", "")).strip()
        if not instruction:
            raise RuntimeError("RoboCasa returned an empty episode instruction")
        initial_frame = _frame(observation)
        frames.append(initial_frame)
        iio.imwrite(first_path, initial_frame)

        for decision in range(max_decisions):
            frame = _frame(observation)
            iio.imwrite(current_path, frame)
            started = time.perf_counter()
            raw_action = predict(
                host,
                port,
                {
                    "image_path": str(current_path.resolve()),
                    "instruction": instruction,
                    "relation_key": mapping["selection_id"],
                },
            )
            latency = time.perf_counter() - started
            inference_seconds.append(latency)
            bounded = adapt_policy_action(
                raw_action,
                translation_gain=translation_gain,
                rotation_gain=rotation_gain,
            )
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            repeats = 0
            for _ in range(action_repeat):
                observation, _, terminated, truncated, info = environment.step(
                    to_robocasa_action(bounded)
                )
                simulator_steps += 1
                repeats += 1
                success = bool(info.get("success", False) or raw._check_success())
                if success or terminated or truncated:
                    break
            frames.append(_frame(observation))
            trajectory.append(
                {
                    "decision_step": decision,
                    "raw_openvla_action": [float(value) for value in raw_action],
                    "executed_bounded_action": bounded,
                    "inference_seconds": latency,
                    "sim_repeats": repeats,
                    "simulator_success": success,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            print(
                f"{mapping['selection_id']} decision={decision + 1}/{max_decisions} "
                f"latency={latency:.3f}s success={success}",
                flush=True,
            )
            if success or terminated or truncated:
                break

        final_frame = frames[-1]
        iio.imwrite(last_path, final_frame)
        iio.imwrite(video_path, np.stack(frames), fps=10)
        final_success = bool(raw._check_success())
        success = bool(success or final_success)

    result = {
        "status": "SUCCESS" if success else "TASK_FAILED",
        "selection_id": mapping["selection_id"],
        "source_table": mapping["source_table"],
        "task": mapping["task"],
        "object": mapping["object"],
        "task_class": mapping["task_class"],
        "object_group": mapping.get("object_group"),
        "mapping_mode": mapping["mapping_mode"],
        "proxy_disclosed": mapping["proxy_disclosed"],
        "instruction": instruction,
        "seed": seed,
        "success": success,
        "actual_closed_loop": True,
        "pure_autonomous_vla": True,
        "oracle_or_scripted_action_used": False,
        "simulator_success_predicate_checked": True,
        "executed_decision_steps": len(trajectory),
        "executed_sim_steps": simulator_steps,
        "max_decision_steps": max_decisions,
        "action_repeat": action_repeat,
        "action_adapter": {
            "kind": "fixed_unit_conversion",
            "translation_gain": translation_gain,
            "rotation_gain": rotation_gain,
            "robocasa_translation_output_range_m": [-0.05, 0.05],
            "robocasa_rotation_output_range_rad": [-0.5, 0.5],
        },
        "mean_inference_seconds": (
            float(np.mean(inference_seconds)) if inference_seconds else None
        ),
        "reset_info": {str(key): str(value) for key, value in reset_info.items()},
        "video": str(video_path.resolve()),
        "first_frame": str(first_path.resolve()),
        "last_frame": str(last_path.resolve()),
        "trajectory": trajectory,
    }
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result["report"] = str(report_path.resolve())
    return result


def _error_result(mapping: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "ERROR",
        "selection_id": mapping["selection_id"],
        "source_table": mapping["source_table"],
        "task": mapping["task"],
        "object": mapping["object"],
        "task_class": mapping["task_class"],
        "object_group": mapping.get("object_group"),
        "success": False,
        "actual_closed_loop": False,
        "pure_autonomous_vla": True,
        "simulator_success_predicate_checked": False,
        "executed_decision_steps": 0,
        "executed_sim_steps": 0,
        "failures": failures,
        "comprehensive_diagnosis": {
            "trigger": "three execution attempts failed",
            "next_action": "continue with full component-boundary diagnosis",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--translation-gain", type=float, default=20.0)
    parser.add_argument("--rotation-gain", type=float, default=2.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-id", action="append")
    args = parser.parse_args(argv)
    if min(args.max_decisions, args.action_repeat, args.attempts) < 1:
        raise ValueError("max-decisions, action-repeat, and attempts must be positive")
    if min(args.translation_gain, args.rotation_gain) <= 0:
        raise ValueError("translation-gain and rotation-gain must be positive")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    selected = select_representatives(plan["mappings"])
    selected = filter_selected_representatives(selected, args.selection_id)
    if args.limit:
        selected = selected[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, mapping in enumerate(selected, start=1):
        failures: list[dict[str, Any]] = []
        for attempt in range(1, args.attempts + 1):
            seed = 824000 + int(mapping["selection_id"].split("-")[-1]) * 10 + attempt
            try:
                result = run_trial(
                    mapping,
                    output_root=args.output,
                    host=args.host,
                    port=args.port,
                    max_decisions=args.max_decisions,
                    action_repeat=args.action_repeat,
                    seed=seed,
                    translation_gain=args.translation_gain,
                    rotation_gain=args.rotation_gain,
                )
                result["attempt"] = attempt
                result["prior_execution_errors"] = failures
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
        else:
            result = _error_result(mapping, failures)
        results.append(result)
        manifest = build_manifest(results, expected_count=len(selected))
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{index}/8] {mapping['selection_id']} status={result['status']} "
            f"success={result['success']}",
            flush=True,
        )
    print((args.output / "manifest.json").resolve(), flush=True)
    return 0 if manifest["all_tasks_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
