"""Run OpenVLA with a stateful Cartesian safety/recovery supervisor."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.collect_water_cup_dagger import INSTRUCTION, cabinet_waypoints
from tools.openvla_gym_water_cup_rollout_v2 import install_object_scale
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict
from tools.water_cup_dagger_oracle import OracleSnapshot, WaterCupDaggerOracle
from tools.water_cup_guarded_control import supervise_action


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--object-scale", type=float, default=0.7)
    parser.add_argument("--max-translation-error", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "water_cup_guarded_rollout",
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
    query_path = args.output_dir / "current_frame.png"
    original_get_obj_cfgs = install_object_scale(PickPlaceCounterToCabinet, args.object_scale)
    env = None
    records: list[dict[str, Any]] = []
    sources: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    initial_object_z = None
    success = False
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
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        initial_object_z = float(raw.sim.data.body_xpos[object_body_id][2])
        front, center, retreat = cabinet_waypoints(raw)
        oracle = WaterCupDaggerOracle(
            front,
            center,
            retreat,
            world_to_origin=controller.world_to_origin_frame,
        )

        for step in range(args.steps):
            frame = np.asarray(observation["video.robot0_agentview_left"], dtype=np.uint8)
            iio.imwrite(query_path, frame)
            policy = np.asarray(
                predict(
                    "127.0.0.1",
                    args.port,
                    {"image_path": str(query_path.resolve()), "instruction": INSTRUCTION},
                ),
                dtype=np.float32,
            )
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            decision = oracle.decide(OracleSnapshot(eef, obj, grasped, success))
            guarded = supervise_action(
                policy,
                decision,
                max_translation_error=args.max_translation_error,
            )
            observation, _, _, _, info = env.step(to_robocasa_action(guarded.action))
            next_eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            next_obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            next_grasped = bool(
                raw._check_grasp(robot.gripper["right"], raw.objects["obj"])
            )
            success = bool(info.get("success", False) or raw._check_success())
            sources[guarded.source] += 1
            phases[decision.phase] += 1
            records.append(
                {
                    "step": step,
                    "phase": decision.phase,
                    "source": guarded.source,
                    "translation_error": guarded.translation_error,
                    "openvla_action": policy.tolist(),
                    "executed_action": guarded.action.tolist(),
                    "grasped": next_grasped,
                    "object_z": float(next_obj[2]),
                    "object_eef_distance": float(np.linalg.norm(next_obj - next_eef)),
                    "success": success,
                }
            )
            if step % 25 == 0 or success:
                print(
                    f"step={step} phase={decision.phase} source={guarded.source} "
                    f"distance={records[-1]['object_eef_distance']:.4f} "
                    f"grasped={next_grasped} success={success}",
                    flush=True,
                )
            if success:
                break
    finally:
        if env is not None:
            env.close()
        PickPlaceCounterToCabinet._get_obj_cfgs = original_get_obj_cfgs

    heights = [record["object_z"] for record in records]
    distances = [record["object_eef_distance"] for record in records]
    openvla_steps = int(sources.get("openvla", 0))
    report = {
        "seed": args.seed,
        "steps": len(records),
        "success": success,
        "object_scale": args.object_scale,
        "max_translation_error": args.max_translation_error,
        "uses_simulator_state_supervisor": True,
        "openvla_direct_steps": openvla_steps,
        "openvla_direct_fraction": openvla_steps / len(records) if records else 0.0,
        "source_counts": dict(sources),
        "phase_counts": dict(phases),
        "ever_grasped": any(record["grasped"] for record in records),
        "initial_object_z": initial_object_z,
        "max_object_z": max(heights) if heights else None,
        "object_lift": max(heights) - initial_object_z if heights else None,
        "minimum_object_eef_distance": min(distances) if distances else None,
        "records": records,
    }
    report_path = args.output_dir / "guarded_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    print(report_path.resolve())
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
