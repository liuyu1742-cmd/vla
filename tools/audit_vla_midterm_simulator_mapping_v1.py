"""Audit that every selected VLA task-object pair resets in RoboCasa."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    ROOT / "outputs" / "midterm_testing_vla" / "vla_midterm_registry.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "midterm_testing_vla" / "simulator_mapping"
)
CAMERA_KEY = "video.robot0_agentview_left"


def install_target_object_group(
    environment_class: type,
    object_group: str,
) -> Callable[..., list[dict[str, Any]]]:
    """Force only the task's target ``obj`` config to one RoboCasa group."""

    normalized = str(object_group).strip()
    if not normalized:
        raise ValueError("object_group must be non-empty")
    original = environment_class._get_obj_cfgs

    def fixed_get_obj_cfgs(environment: Any) -> list[dict[str, Any]]:
        configs = original(environment)
        found = False
        for config in configs:
            if config.get("name") == "obj":
                config["obj_groups"] = normalized
                found = True
        if not found:
            raise RuntimeError(
                f"{environment_class.__name__} has no target object config"
            )
        return configs

    environment_class._get_obj_cfgs = fixed_get_obj_cfgs
    return original


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _task_class(task_class: str) -> type:
    from robocasa.environments.kitchen.atomic import kitchen_pick_place

    cls = getattr(kitchen_pick_place, task_class, None)
    if not isinstance(cls, type):
        raise ValueError(f"RoboCasa task class is unavailable: {task_class}")
    return cls


def _actual_object_metadata(raw: Any) -> dict[str, Any]:
    config = next(
        (
            item
            for item in raw.object_cfgs
            if isinstance(item, dict) and item.get("name") == "obj"
        ),
        None,
    )
    if config is None:
        raise RuntimeError("RoboCasa reset has no target object config")
    info = config.get("info") if isinstance(config.get("info"), dict) else {}
    model = raw.objects["obj"]
    return {
        "category": info.get("cat"),
        "mjcf_path": str(Path(model.mjcf_path).resolve()),
        "size": np.asarray(model.size, dtype=float).tolist(),
        "object_config_group": config.get("obj_groups"),
    }


def _one_reset(
    relation: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    import gymnasium as gym
    import imageio.v3 as iio
    import robocasa  # noqa: F401
    from robocasa.models.objects.kitchen_objects import OBJ_GROUPS

    task_class = str(relation["task_class"])
    object_group = str(relation["object_group"])
    if object_group not in OBJ_GROUPS:
        raise ValueError(f"RoboCasa object group is unavailable: {object_group}")
    cls = _task_class(task_class)
    original = install_target_object_group(cls, object_group)
    env = None
    try:
        env = gym.make(
            f"robocasa/{task_class}",
            split="pretrain",
            seed=seed,
            obj_registries=("lightwheel",),
            obj_groups=object_group,
            camera_names=["robot0_agentview_left"],
            disable_env_checker=True,
        )
        observation, reset_info = env.reset(seed=seed)
        raw = env.unwrapped.env
        frame = np.asarray(observation[CAMERA_KEY], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError(f"invalid RGB frame shape: {frame.shape}")
        frame_path = output_dir / "reset_frame.png"
        iio.imwrite(frame_path, frame)

        actual = _actual_object_metadata(raw)
        allowed_categories = list(OBJ_GROUPS[object_group])
        actual_category = actual["category"]
        if actual_category not in allowed_categories:
            raise RuntimeError(
                f"sampled category {actual_category!r} not in "
                f"OBJ_GROUPS[{object_group!r}]={allowed_categories!r}"
            )
        task_success_at_reset = bool(raw._check_success())
        if task_success_at_reset:
            raise RuntimeError("task is already successful at reset")
        lower, upper = raw.action_spec
        action_shape = list(np.asarray(lower).shape)
        if action_shape != [12] or np.asarray(upper).shape != (12,):
            raise RuntimeError(
                f"expected native RoboCasa action shape [12], got {action_shape}"
            )

        episode_meta = raw.get_ep_meta()
        instruction = str(episode_meta.get("lang", ""))
        expected_object = str(relation["manipulated_object_text"])
        if expected_object not in instruction.lower():
            raise RuntimeError(
                f"reset instruction does not name {expected_object!r}: "
                f"{instruction!r}"
            )
        return {
            "status": "PASS",
            "task_class": task_class,
            "object_group": object_group,
            "seed": seed,
            "instruction": instruction,
            "source_instruction": relation["example_instruction"],
            "actual_object": actual,
            "allowed_categories": allowed_categories,
            "camera": {
                "key": CAMERA_KEY,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "frame": str(frame_path.resolve()),
            },
            "action_shape": action_shape,
            "task_success_at_reset": task_success_at_reset,
            "reset_info": {
                str(key): _json_ready(value)
                for key, value in reset_info.items()
            },
        }
    finally:
        if env is not None:
            env.close()
        cls._get_obj_cfgs = original


def audit_relation(
    relation: dict[str, Any],
    *,
    relation_index: int,
    output_root: Path,
    attempts: int,
) -> dict[str, Any]:
    relation_dir = (
        output_root
        / f"{relation_index:03d}_{relation['task_class']}_{relation['object_group']}"
    )
    relation_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        seed = 730000 + relation_index * 10 + attempt
        try:
            report = _one_reset(
                relation,
                seed=seed,
                output_dir=relation_dir,
            )
            report["attempt"] = attempt
            report["prior_failures"] = failures
            break
        except Exception as exc:  # continue the batch and retain diagnostics
            failures.append(
                {
                    "attempt": attempt,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    else:
        report = {
            "status": "FAIL",
            "task_class": relation["task_class"],
            "object_group": relation["object_group"],
            "attempts": attempts,
            "failures": failures,
        }

    report_path = relation_dir / "report.json"
    report["report_path"] = str(report_path.resolve())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def run_audit(
    registry_path: Path,
    output_root: Path,
    *,
    attempts: int,
    resume: bool,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    relations = registry["relations"]
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, relation in enumerate(relations, start=1):
        relation_dir = (
            output_root
            / f"{index:03d}_{relation['task_class']}_{relation['object_group']}"
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
            print(
                f"[{index:02d}/{len(relations)}] "
                f"{relation['task_class']}::{relation['object_group']} SKIP PASS",
                flush=True,
            )
        else:
            result = audit_relation(
                relation,
                relation_index=index,
                output_root=output_root,
                attempts=attempts,
            )
            print(
                f"[{index:02d}/{len(relations)}] "
                f"{relation['task_class']}::{relation['object_group']} "
                f"{result['status']}",
                flush=True,
            )
        results.append(result)
        manifest = {
            "schema_version": "vla_midterm_simulator_mapping_v1",
            "registry": str(registry_path.resolve()),
            "task_count": registry["summary"]["task_count"],
            "object_count": registry["summary"]["unique_object_count"],
            "attempts_per_failed_relation": attempts,
            "pass_count": sum(
                item.get("status") == "PASS" for item in results
            ),
            "fail_count": sum(
                item.get("status") != "PASS" for item in results
            ),
            "completed_count": len(results),
            "all_passed": (
                len(results) == len(relations)
                and all(item.get("status") == "PASS" for item in results)
            ),
            "results": results,
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    manifest = run_audit(
        args.registry,
        args.output,
        attempts=args.attempts,
        resume=not args.no_resume,
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
    print((args.output / "manifest.json").resolve(), flush=True)
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
