"""Short bounded calibration of toy grasp height, contacts and Panda finger state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action


ROOT = Path(__file__).resolve().parents[1]


def _step_toward(
    env: Any,
    controller: Any,
    current: np.ndarray,
    target: np.ndarray,
    *,
    gripper: float,
) -> Any:
    error = controller.world_to_origin_frame(target) - controller.world_to_origin_frame(
        current
    )
    translation = np.clip(error / 0.05, -1.0, 1.0)
    action = np.r_[translation, np.zeros(3), gripper].astype(np.float32)
    return env.step(to_robocasa_action(action))


def _contacts(raw: Any, object_geoms: set[str]) -> list[list[str]]:
    pairs: set[tuple[str, str]] = set()
    for index in range(raw.sim.data.ncon):
        contact = raw.sim.data.contact[index]
        first = raw.sim.model.geom_id2name(contact.geom1) or str(contact.geom1)
        second = raw.sim.model.geom_id2name(contact.geom2) or str(contact.geom2)
        if first in object_geoms or second in object_geoms:
            pairs.add(tuple(sorted((first, second))))
    return [list(pair) for pair in sorted(pairs)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--offsets",
        nargs="+",
        type=float,
        default=[0.025, 0.04, 0.055, 0.07, 0.01],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "organizing_toy_grasp_calibration.json",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    from tools.skill_transfer.robocasa_envs import create_formal_env

    env = None
    results: list[dict[str, object]] = []
    try:
        env, mapping = create_formal_env("organizing::toy", seed=args.seed)
        for offset in args.offsets:
            env.reset(seed=args.seed)
            raw = env.unwrapped.env
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            eef_site_id = robot.eef_site_id["right"]
            object_body_id = raw.obj_body_id["obj"]
            obj_model = raw.objects["obj"]
            object_geoms = set(obj_model.contact_geoms)
            initial_z = float(raw.sim.data.body_xpos[object_body_id][2])

            def eef() -> np.ndarray:
                return np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float).copy()

            def obj() -> np.ndarray:
                return np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float).copy()

            move_steps = 0
            reached = True
            for height in (0.18, float(offset)):
                target = obj() + np.array([0.0, 0.0, height])
                for _ in range(260):
                    if float(np.linalg.norm(eef() - target)) <= 0.028:
                        break
                    _step_toward(env, controller, eef(), target, gripper=-1.0)
                    move_steps += 1
                else:
                    reached = False
                    break

            ever_grasped = False
            grasp_step = None
            min_distance = float(np.linalg.norm(eef() - obj()))
            contact_pairs: set[tuple[str, str]] = set()
            for close_step in range(50):
                env.step(
                    to_robocasa_action(
                        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                    )
                )
                min_distance = min(min_distance, float(np.linalg.norm(eef() - obj())))
                for pair in _contacts(raw, object_geoms):
                    contact_pairs.add(tuple(pair))
                grasped = bool(raw._check_grasp(robot.gripper["right"], obj_model))
                if grasped and not ever_grasped:
                    ever_grasped = True
                    grasp_step = close_step

            final_z = float(obj()[2])
            maintained_after_lift = False
            if ever_grasped:
                lift_target = eef() + np.array([0.0, 0.0, 0.15])
                for _ in range(180):
                    if float(np.linalg.norm(eef() - lift_target)) <= 0.028:
                        break
                    _step_toward(env, controller, eef(), lift_target, gripper=1.0)
                final_z = float(obj()[2])
                maintained_after_lift = bool(
                    raw._check_grasp(robot.gripper["right"], obj_model)
                )

            finger_qpos: dict[str, float] = {}
            for joint in robot.gripper["right"].joints:
                try:
                    finger_qpos[joint] = float(raw.sim.data.get_joint_qpos(joint))
                except Exception:
                    continue
            result = {
                "offset": offset,
                "reached": reached,
                "move_steps": move_steps,
                "ever_grasped": ever_grasped,
                "first_grasp_close_step": grasp_step,
                "maintained_after_lift": maintained_after_lift,
                "object_lift": final_z - initial_z,
                "minimum_object_eef_distance": min_distance,
                "object_quaternion": np.asarray(
                    raw.sim.data.body_xquat[object_body_id], dtype=float
                ).tolist(),
                "finger_qpos": finger_qpos,
                "object_contact_geoms": sorted(object_geoms),
                "object_contacts": [list(pair) for pair in sorted(contact_pairs)],
                "gripper_important_geoms": {
                    str(key): list(value) if isinstance(value, (list, tuple)) else value
                    for key, value in robot.gripper["right"].important_geoms.items()
                },
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if maintained_after_lift:
                break
    finally:
        if env is not None:
            env.close()

    report = {
        "schema_version": "toy_grasp_calibration_v1",
        "seed": args.seed,
        "mapping": mapping if "mapping" in locals() else None,
        "success": any(item["maintained_after_lift"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve(), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
