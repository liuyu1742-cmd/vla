"""Run a bounded visual OpenVLA-to-RoboCasa rollout across two environments."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tools.robocasa_raw_executor import raw_robocasa_action
from tools.skill_phase_planner import PROMPTS, next_phase


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "openvla_robocasa_rollout"
DEFAULT_OPENVLA_PYTHON = Path(r"C:\Users\sjtu101\miniconda3\envs\openvla\python.exe")


def build_request(image_path: str, phase: str) -> dict[str, str]:
    """Build the contract consumed by the separate OpenVLA inference process."""
    if phase not in PROMPTS:
        raise ValueError(f"unknown phase: {phase}")
    return {"image_path": image_path, "instruction": PROMPTS[phase]}


def choose_phase(current: str, predicates: dict[str, bool]) -> str:
    """Keep phase logic separate from learned continuous actions."""
    return next_phase(current, predicates)


def _verified_grasp(env) -> bool:
    """Return a conservative grasp predicate; unknown API shapes mean false."""
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
    parser.add_argument("--openvla-python", type=Path, default=DEFAULT_OPENVLA_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    import imageio.v3 as iio
    from robocasa.utils.env_utils import create_env

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = create_env("PickPlaceCounterToCabinet", camera_names=["robot0_agentview_left"], seed=args.seed)
    phase = "pick"
    trajectory: list[dict[str, object]] = []
    try:
        env.reset()
        for step in range(args.steps):
            image_path = args.output_dir / f"frame_{step:04d}.png"
            request_path = args.output_dir / f"request_{step:04d}.json"
            response_path = args.output_dir / f"response_{step:04d}.json"
            frame = env.sim.render(height=224, width=224, camera_name="robot0_agentview_left")[::-1]
            iio.imwrite(image_path, frame)
            request_path.write_text(json.dumps(build_request(str(image_path), phase)), encoding="utf-8")
            subprocess.run(
                [str(args.openvla_python), str(ROOT / "tools" / "openvla_ipc_predict.py"), "--request", str(request_path), "--response", str(response_path)],
                check=True,
            )
            action = raw_robocasa_action(json.loads(response_path.read_text(encoding="utf-8"))["raw_action"])
            _, _, _, info = env.step(action)
            grasped = _verified_grasp(env)
            success = bool(info.get("success", False)) or bool(env._check_success())
            trajectory.append({
                "step": step,
                "phase": phase,
                "image_path": str(image_path),
                "native_action": action,
                "grasped": grasped,
                "success": success,
            })
            phase = choose_phase(phase, {"grasped": grasped})
            if success:
                break
    finally:
        env.close()
    report_path = args.output_dir / "rollout_report.json"
    report_path.write_text(json.dumps({"seed": args.seed, "steps": trajectory}, indent=2), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
