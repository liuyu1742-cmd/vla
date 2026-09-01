"""Run a geometry-matched OpenVLA rollout with evaluation-only state diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.openvla_gym_water_cup_rollout_v2 import install_object_scale
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--object-scale", type=float, default=0.7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "openvla_gym_water_cup_diagnostic",
    )
    args = parser.parse_args()

    third_party = ROOT / "third_party"
    sys.path.insert(0, str(third_party / "robosuite"))
    sys.path.insert(0, str(third_party / "robocasa"))
    import gymnasium as gym
    import imageio.v3 as iio
    import robocasa  # noqa: F401
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
        PickPlaceCounterToCabinet,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_get_obj_cfgs = install_object_scale(
        PickPlaceCounterToCabinet, args.object_scale
    )
    env = None
    records = []
    initial_object_z = None
    try:
        env = gym.make(
            "robocasa/PickPlaceCounterToCabinet",
            split="pretrain",
            seed=args.seed,
            obj_registries=("lightwheel",),
            obj_groups="glass_cup",
            disable_env_checker=True,
        )
        observation, _ = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        initial_object_z = float(raw.sim.data.body_xpos[object_body_id][2])

        for step in range(args.steps):
            frame_path = args.output_dir / f"frame_{step:04d}.png"
            iio.imwrite(
                frame_path,
                np.asarray(observation["video.robot0_agentview_left"]),
            )
            action = predict(
                "127.0.0.1",
                args.port,
                {
                    "image_path": str(frame_path),
                    "instruction": "pick up the glass cup and place it in the cabinet",
                },
            )
            observation, _, _, _, info = env.step(to_robocasa_action(action))
            object_position = np.asarray(
                raw.sim.data.body_xpos[object_body_id], dtype=float
            )
            eef_position = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            success = bool(info.get("success", False))
            records.append(
                {
                    "step": step,
                    "action": action,
                    "grasped": grasped,
                    "success": success,
                    "object_position": object_position.tolist(),
                    "eef_position": eef_position.tolist(),
                    "object_eef_distance": float(
                        np.linalg.norm(object_position - eef_position)
                    ),
                }
            )
            if success:
                break
    finally:
        if env is not None:
            env.close()
        PickPlaceCounterToCabinet._get_obj_cfgs = original_get_obj_cfgs

    grasp_steps = [record["step"] for record in records if record["grasped"]]
    object_heights = [record["object_position"][2] for record in records]
    distances = [record["object_eef_distance"] for record in records]
    summary = {
        "seed": args.seed,
        "object_scale": args.object_scale,
        "executed_steps": len(records),
        "success": bool(records and records[-1]["success"]),
        "ever_grasped": bool(grasp_steps),
        "first_grasp_step": grasp_steps[0] if grasp_steps else None,
        "last_grasp_step": grasp_steps[-1] if grasp_steps else None,
        "grasped_step_count": len(grasp_steps),
        "initial_object_z": initial_object_z,
        "max_object_z": max(object_heights) if object_heights else None,
        "object_lift": (
            max(object_heights) - initial_object_z
            if object_heights and initial_object_z is not None
            else None
        ),
        "minimum_object_eef_distance": min(distances) if distances else None,
    }
    report = {"summary": summary, "steps": records}
    report_path = args.output_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
