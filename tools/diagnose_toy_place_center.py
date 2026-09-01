"""Verify one complete shallow-center cabinet placement without early release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.collect_formal_skill_expert import cabinet_waypoints
from tools.diagnose_toy_place_geometry import _move
from tools.openvla_simulator_adapter import to_robocasa_action


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=float, default=0.20)
    parser.add_argument("--closed-limit", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "organizing_toy_place_center.json",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    from robocasa.utils import object_utils as OU
    from tools.skill_transfer.robocasa_envs import create_formal_env

    env = None
    try:
        env, mapping = create_formal_env("organizing::toy", seed=args.seed)
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
        grasp_after_close = grasped()
        phases["lift"] = _move(
            env, controller, eef, eef() + [0, 0, 0.18],
            gripper=1.0, limit=args.closed_limit, max_steps=220,
        )
        front, center, retreat = cabinet_waypoints(raw, depth_fraction=args.depth)
        phases["front"] = _move(
            env, controller, eef, front,
            gripper=1.0, limit=args.closed_limit, max_steps=420,
        )
        grasp_at_front = grasped()
        phases["center"] = _move(
            env, controller, eef, center,
            gripper=1.0, limit=args.closed_limit, max_steps=360,
        )
        grasp_at_center = grasped()
        inside_at_center = inside()
        if grasp_at_center and inside_at_center:
            for _ in range(45):
                env.step(to_robocasa_action([0, 0, 0, 0, 0, 0, -1]))
            phases["retreat"] = _move(
                env, controller, eef, retreat,
                gripper=-1.0, limit=0.5, max_steps=220,
            )
            for _ in range(30):
                env.step(to_robocasa_action([0, 0, 0, 0, 0, 0, -1]))
        report = {
            "schema_version": "toy_place_center_check_v1",
            "seed": args.seed,
            "depth_fraction": args.depth,
            "closed_translation_limit": args.closed_limit,
            "mapping": mapping,
            "grasp_after_close": grasp_after_close,
            "grasp_at_front": grasp_at_front,
            "grasp_at_center": grasp_at_center,
            "inside_at_center": inside_at_center,
            "final_grasped": grasped(),
            "final_inside": inside(),
            "success": bool(raw._check_success()),
            "final_object_position": obj().tolist(),
            "phases": phases,
        }
    finally:
        if env is not None:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(args.output.resolve(), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
