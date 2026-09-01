"""Evaluate water-cup OpenVLA in the exact geometry used by expert demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict


ROOT = Path(__file__).resolve().parents[1]


def install_object_scale(environment_class: type, object_scale: float) -> Callable:
    """Patch the task object scale and return the method that must be restored."""
    if object_scale <= 0:
        raise ValueError("object_scale must be positive")
    original_get_obj_cfgs = environment_class._get_obj_cfgs

    def scaled_get_obj_cfgs(environment: Any):
        configs = original_get_obj_cfgs(environment)
        for config in configs:
            if config.get("name") == "obj":
                config["object_scale"] = object_scale
        return configs

    environment_class._get_obj_cfgs = scaled_get_obj_cfgs
    return original_get_obj_cfgs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--query-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--object-scale", type=float, default=0.7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "openvla_gym_water_cup_v2",
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
    records: list[dict[str, Any]] = []
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
        current = [0.0] * 7
        for step in range(args.steps):
            if step % args.query_interval == 0:
                frame_path = args.output_dir / f"frame_{step:04d}.png"
                iio.imwrite(
                    frame_path,
                    np.asarray(observation["video.robot0_agentview_left"]),
                )
                current = predict(
                    "127.0.0.1",
                    args.port,
                    {
                        "image_path": str(frame_path),
                        "instruction": "pick up the glass cup and place it in the cabinet",
                    },
                )
            observation, _, _, _, info = env.step(to_robocasa_action(current))
            records.append(
                {
                    "step": step,
                    "action": current,
                    "success": bool(info.get("success", False)),
                }
            )
            if info.get("success", False):
                break
    finally:
        if env is not None:
            env.close()
        PickPlaceCounterToCabinet._get_obj_cfgs = original_get_obj_cfgs

    report = {
        "seed": args.seed,
        "object_scale": args.object_scale,
        "steps": records,
        "success": bool(records and records[-1]["success"]),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
