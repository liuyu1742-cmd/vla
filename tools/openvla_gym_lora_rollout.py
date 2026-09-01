"""Evaluate water-cup LoRA through the same RoboCasa Gym interface as its demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict


ROOT = Path(__file__).resolve().parents[1]


def action_for_gym(openvla_action: list[float]) -> dict:
    return to_robocasa_action(openvla_action)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--query-interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "openvla_gym_lora_eval")
    args = parser.parse_args()
    third_party = ROOT / "third_party"
    sys.path.insert(0, str(third_party / "robosuite")); sys.path.insert(0, str(third_party / "robocasa"))
    import imageio.v3 as iio
    import gymnasium as gym
    import robocasa  # noqa: F401
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("robocasa/PickPlaceCounterToCabinet", split="pretrain", seed=args.seed, obj_registries=("lightwheel",), obj_groups="glass_cup", disable_env_checker=True)
    obs, _ = env.reset(seed=args.seed); current = [0.0] * 7; records=[]
    try:
        for step in range(args.steps):
            if step % args.query_interval == 0:
                path = args.output_dir / f"frame_{step:04d}.png"
                iio.imwrite(path, np.asarray(obs["video.robot0_agentview_left"]))
                current = predict("127.0.0.1", args.port, {"image_path": str(path), "instruction": "pick up the glass cup and place it in the cabinet"})
            obs, _, _, _, info = env.step(action_for_gym(current))
            records.append({"step": step, "action": current, "success": bool(info.get("success", False))})
            if info.get("success", False): break
    finally:
        env.close()
    report = {"seed": args.seed, "steps": records, "success": bool(records and records[-1]["success"])}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output_dir / "report.json")
    return 0 if report["success"] else 1


if __name__ == "__main__": raise SystemExit(main())
