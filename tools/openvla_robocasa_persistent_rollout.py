"""Run a RoboCasa visual rollout using the persistent local OpenVLA service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.openvla_tcp_client import predict
from tools.robocasa_raw_executor import raw_robocasa_action
from tools.skill_phase_planner import PROMPTS, next_phase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "openvla_robocasa_persistent_rollout"


def rollout_instruction(phase: str) -> str:
    if phase not in PROMPTS:
        raise ValueError(f"unknown phase: {phase}")
    return PROMPTS[phase]


def _verified_grasp(env) -> bool:
    try:
        arm = env.robots[0].arms[0]
        gripper = env.robots[0].gripper[arm]
        cup = env.objects.get("glass_cup")
        return bool(cup is not None and env._check_grasp(gripper, cup))
    except (AttributeError, KeyError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    import imageio.v3 as iio
    from robocasa.utils.env_utils import create_env

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = create_env("PickPlaceCounterToCabinet", camera_names=["robot0_agentview_left"], seed=args.seed)
    phase = "pick"
    steps: list[dict[str, object]] = []
    try:
        env.reset()
        for step in range(args.steps):
            image_path = args.output_dir / f"frame_{step:04d}.png"
            frame = env.sim.render(height=224, width=224, camera_name="robot0_agentview_left")[::-1]
            iio.imwrite(image_path, frame)
            request = {"image_path": str(image_path), "instruction": rollout_instruction(phase)}
            action = raw_robocasa_action(predict(args.host, args.port, request))
            _, _, _, info = env.step(action)
            grasped = _verified_grasp(env)
            success = bool(info.get("success", False)) or bool(env._check_success())
            steps.append({"step": step, "phase": phase, "native_action": action, "grasped": grasped, "success": success})
            phase = next_phase(phase, {"grasped": grasped})
            if success:
                break
    finally:
        env.close()
    report = args.output_dir / "rollout_report.json"
    report.write_text(json.dumps({"seed": args.seed, "steps": steps}, indent=2), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
