"""Short event-aware calibration for placing a grasped toy inside the cabinet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools.collect_formal_skill_expert import cabinet_waypoints
from tools.openvla_simulator_adapter import to_robocasa_action


ROOT = Path(__file__).resolve().parents[1]


def _move(
    env: Any,
    controller: Any,
    eef: Callable[[], np.ndarray],
    target: np.ndarray,
    *,
    gripper: float,
    limit: float,
    max_steps: int,
    stop_when: Callable[[], bool] | None = None,
) -> dict[str, object]:
    for step in range(max_steps):
        if stop_when is not None and stop_when():
            return {"reached": False, "event": True, "steps": step}
        error = controller.world_to_origin_frame(target) - controller.world_to_origin_frame(
            eef()
        )
        if float(np.linalg.norm(error)) <= 0.028:
            return {"reached": True, "event": False, "steps": step}
        translation = np.clip(error / 0.05, -limit, limit)
        action = np.r_[translation, np.zeros(3), gripper]
        env.step(to_robocasa_action(action))
    return {"reached": False, "event": False, "steps": max_steps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--depths", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.40]
    )
    parser.add_argument("--closed-limit", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "organizing_toy_place_calibration.json",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    from robocasa.utils import object_utils as OU
    from tools.skill_transfer.robocasa_envs import create_formal_env

    env = None
    results: list[dict[str, object]] = []
    try:
        env, mapping = create_formal_env("organizing::toy", seed=args.seed)
        for depth in args.depths:
            env.reset(seed=args.seed)
            raw = env.unwrapped.env
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            eef_site_id = robot.eef_site_id["right"]
            object_body_id = raw.obj_body_id["obj"]
            obj_model = raw.objects["obj"]

            def eef() -> np.ndarray:
                return np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float).copy()

            def obj() -> np.ndarray:
                return np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float).copy()

            def grasped() -> bool:
                return bool(raw._check_grasp(robot.gripper["right"], obj_model))

            def inside() -> bool:
                return bool(OU.obj_inside_of(raw, "obj", raw.cab))

            phases: dict[str, object] = {}
            phases["approach"] = _move(
                env, controller, eef, obj() + [0, 0, 0.18],
                gripper=-1.0, limit=1.0, max_steps=260,
            )
            phases["descend"] = _move(
                env, controller, eef, obj() + [0, 0, 0.010],
                gripper=-1.0, limit=1.0, max_steps=260,
            )
            for _ in range(50):
                env.step(to_robocasa_action([0, 0, 0, 0, 0, 0, 1]))
            grasp_verified = grasped()
            if grasp_verified:
                phases["lift"] = _move(
                    env, controller, eef, eef() + [0, 0, 0.18],
                    gripper=1.0, limit=args.closed_limit, max_steps=220,
                )
            front, center, retreat = cabinet_waypoints(raw, depth_fraction=depth)
            if grasp_verified and grasped():
                phases["front"] = _move(
                    env, controller, eef, front,
                    gripper=1.0, limit=args.closed_limit, max_steps=420,
                )
            if grasp_verified and grasped():
                phases["inside"] = _move(
                    env, controller, eef, center,
                    gripper=1.0, limit=args.closed_limit, max_steps=360,
                    stop_when=inside,
                )
            inside_before_release = inside()
            grasp_before_release = grasped()
            if inside_before_release:
                for _ in range(45):
                    env.step(to_robocasa_action([0, 0, 0, 0, 0, 0, -1]))
                phases["retreat"] = _move(
                    env, controller, eef, retreat,
                    gripper=-1.0, limit=0.5, max_steps=220,
                )
                for _ in range(20):
                    env.step(to_robocasa_action([0, 0, 0, 0, 0, 0, -1]))
            result = {
                "depth_fraction": depth,
                "closed_translation_limit": args.closed_limit,
                "grasp_verified": grasp_verified,
                "grasp_before_release": grasp_before_release,
                "inside_before_release": inside_before_release,
                "success": bool(raw._check_success()),
                "final_grasped": grasped(),
                "final_inside": inside(),
                "final_object_position": obj().tolist(),
                "phases": phases,
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["success"]:
                break
    finally:
        if env is not None:
            env.close()

    report = {
        "schema_version": "toy_place_calibration_v1",
        "seed": args.seed,
        "mapping": mapping if "mapping" in locals() else None,
        "success": any(item["success"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve(), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
