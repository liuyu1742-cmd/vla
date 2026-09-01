"""Record successful RoboCasa water-cup expert actions as VLA supervision."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\RobotProject\RobotProject\robocasa_gt_grasp.py")
INSTRUCTION = "pick up the glass cup and place it in the cabinet"


def flatten_gym_action(action: dict[str, Any]) -> list[float]:
    """Flatten the expert Gym action into OpenVLA-compatible arm supervision."""
    keys = ("action.end_effector_position", "action.end_effector_rotation", "action.gripper_close")
    values: list[float] = []
    for key in keys:
        values.extend(float(v) for v in np.asarray(action[key]).reshape(-1))
    if len(values) != 7:
        raise ValueError(f"expert arm action must have 7 values, got {len(values)}")
    return values


class RecordingEnv:
    def __init__(self, env: Any, stride: int) -> None:
        self._env = env
        self._stride = stride
        self._observation: dict[str, Any] | None = None
        self.frames: list[np.ndarray] = []
        self.actions: list[list[float]] = []
        self._steps = 0

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    def reset(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self._env.reset(*args, **kwargs)
        self._observation = observation
        return observation, info

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._observation is not None and self._steps % self._stride == 0:
            self.frames.append(np.asarray(self._observation["video.robot0_agentview_left"], dtype=np.uint8))
            self.actions.append(flatten_gym_action(action))
        self._steps += 1
        result = self._env.step(action)
        self._observation = result[0]
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "water_cup_expert")
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("water_cup_expert_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load expert source: {SOURCE}")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    original_make = source.gym.make
    holder: dict[str, RecordingEnv] = {}

    def recording_make(*make_args: Any, **make_kwargs: Any) -> RecordingEnv:
        wrapped = RecordingEnv(original_make(*make_args, **make_kwargs), args.frame_stride)
        holder["env"] = wrapped
        return wrapped

    source.gym.make = recording_make
    source_output = args.output_dir / f"seed_{args.seed:03d}"
    old_argv = sys.argv
    sys.argv = [str(SOURCE), "--seed", str(args.seed), "--object-group", "glass_cup", "--grasp-mode", "top", "--object-scale", "0.7", "--grasp-height-offset", "0.002", "--cabinet-depth-fraction", "0.5", "--output", str(source_output)]
    try:
        source.main()
    finally:
        sys.argv = old_argv
        source.gym.make = original_make

    report = json.loads((source_output / "gt_pick_place_report.json").read_text(encoding="utf-8"))
    recorded = holder["env"]
    episode_path = args.output_dir / f"episode_seed_{args.seed:03d}.npz"
    np.savez_compressed(episode_path, frames=np.asarray(recorded.frames), actions=np.asarray(recorded.actions, dtype=np.float32))
    manifest = {"seed": args.seed, "instruction": INSTRUCTION, "success": bool(report["success"]), "samples": len(recorded.actions), "frame_stride": args.frame_stride, "episode": str(episode_path), "report": str(source_output / "gt_pick_place_report.json")}
    (args.output_dir / f"episode_seed_{args.seed:03d}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))
    return 0 if manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
