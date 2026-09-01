"""Generate one verified RoboCasa pick-place expert trajectory for VLA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def proportional_action(current: Sequence[float], target: Sequence[float], *, gripper: float, gain: float = 1.0) -> list[float]:
    """Return one bounded 7-D Cartesian action toward a world-coordinate target."""
    if len(current) != 3 or len(target) != 3:
        raise ValueError("current and target must be three-dimensional")
    delta = [min(1.0, max(-1.0, (float(target[i]) - float(current[i])) * gain)) for i in range(3)]
    return delta + [0.0, 0.0, 0.0, float(gripper)]


def cabinet_drop_target(interior_sites: Sequence[Sequence[float]], floor_clearance: float = 0.05) -> list[float]:
    """Use the center of the lowest cabinet interior plane as a drop target."""
    if len(interior_sites) < 4:
        raise ValueError("cabinet interior requires p0, px, py and pz sites")
    p0, px, py, _ = interior_sites[:4]
    return [
        (float(p0[0]) + float(px[0]) + float(py[0])) / 3.0,
        (float(p0[1]) + float(px[1]) + float(py[1])) / 3.0,
        float(p0[2]) + floor_clearance,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "robocasa_pick_place_expert")
    args = parser.parse_args()

    import numpy as np
    from robocasa.utils.env_utils import create_env

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = create_env("PickPlaceCounterToCabinet", camera_names=["robot0_agentview_left"], seed=args.seed, obj_groups="glass_cup")
    phases: list[dict[str, object]] = []
    try:
        env.reset()
        arm = env.robots[0].arms[0]
        site_id = env.robots[0].eef_site_id[arm]
        gripper = env.robots[0].gripper[arm]
        obj_body_id = env.obj_body_id["obj"]

        def eef() -> np.ndarray:
            return np.array(env.sim.data.site_xpos[site_id]).copy()

        def obj() -> np.ndarray:
            return np.array(env.sim.data.body_xpos[obj_body_id]).copy()

        def step(action: Sequence[float]) -> None:
            env.step(np.asarray(list(action) + [0.0] * 5, dtype=np.float32))

        def move(phase: str, target: Sequence[float], gripper_cmd: float, limit: int = 420) -> bool:
            reached = False
            for count in range(1, limit + 1):
                current = eef()
                if float(np.linalg.norm(current - np.asarray(target))) < 0.028:
                    reached = True
                    break
                step(proportional_action(current, target, gripper=gripper_cmd, gain=1.2))
            phases.append({"phase": phase, "reached": reached, "steps": count, "target": [float(v) for v in target], "eef": eef().tolist()})
            return reached

        object_pos = obj()
        move("approach_object", object_pos + np.array([0.0, 0.0, 0.18]), -1.0)
        move("descend_to_object", object_pos + np.array([0.0, 0.0, 0.025]), -1.0)
        for _ in range(40):
            step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        grasped = bool(env._check_grasp(gripper, env.objects["obj"].contact_geoms))
        phases.append({"phase": "verify_grasp", "grasped": grasped})
        move("lift_object", eef() + np.array([0.0, 0.0, 0.24]), 1.0)

        lower_level = next(iter(env.cab.get_int_sites(relative=False).values()))
        drop_target = np.asarray(cabinet_drop_target(lower_level))
        move("approach_cabinet", drop_target + np.array([0.0, 0.0, 0.20]), 1.0)
        move("move_inside_cabinet", drop_target, 1.0)
        for _ in range(40):
            step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        move("retreat", eef() + np.array([-0.20, 0.0, 0.12]), -1.0, limit=180)
        for _ in range(20):
            step([0.0] * 7)
        success = bool(env._check_success())
        report = {"task": "PickPlaceCounterToCabinet", "seed": args.seed, "object_group": "glass_cup", "uses_ground_truth_pose": True, "success": success, "phases": phases, "final_object": obj().tolist()}
    finally:
        env.close()
    report_path = args.output_dir / "expert_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
