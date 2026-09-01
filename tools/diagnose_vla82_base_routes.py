"""Probe direct cardinal base motions in one deterministic VLA82 scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_vla82_full_simulation import _load_compiled_specs, _scene_for_spec
from tools.vla82_full_sim.environment import make_environment
from tools.vla82_full_sim.expert import ActionLayout, _raw_environment


def move_to(environment: object, layout: ActionLayout, target: np.ndarray, steps: int) -> tuple[np.ndarray, float]:
    raw = _raw_environment(environment)
    base = raw.robots[0].composite_controller.part_controllers["base"]
    action = np.zeros(environment.action_space.shape, dtype=np.float32)
    for _ in range(steps):
        position, rotation = base.get_base_pose()
        local = np.asarray(rotation, dtype=float)[:2, :2].T @ (target - np.asarray(position, dtype=float)[:2])
        action[:] = 0.0
        action[layout.base[0]:layout.base[0] + 2] = np.clip(local / 0.08, -0.5, 0.5)
        action[layout.base_mode] = 1.0
        environment.step(action)
    position, _ = base.get_base_pose()
    point = np.asarray(position, dtype=float)[:2]
    return point, float(np.linalg.norm(target - point))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-id", default="VLA82-058")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    spec = next(item for item in _load_compiled_specs() if item.selection_id == args.selection_id)
    environment = make_environment(_scene_for_spec(spec), args.seed)
    try:
        raw = _raw_environment(environment)
        layout = ActionLayout.from_env(environment)
        base = raw.robots[0].composite_controller.part_controllers["base"]
        initial, _ = base.get_base_pose()
        origin = np.asarray(initial, dtype=float)[:2]
        print(f"start={origin.tolist()} layout={layout}")
        for label, delta in (("east", (0.45, 0.0)), ("west", (-0.45, 0.0)), ("south", (0.0, -0.45)), ("north", (0.0, 0.45))):
            target = origin + np.asarray(delta, dtype=float)
            position, remainder = move_to(environment, layout, target, args.steps)
            print(f"{label}: target={target.round(4).tolist()} position={position.round(4).tolist()} remainder={remainder:.4f}")
        route = (
            ("route-east", origin + np.asarray((0.45, 0.0))),
            ("route-south", origin + np.asarray((0.45, -0.85))),
            ("route-west", origin + np.asarray((-0.30, -0.85))),
        )
        for label, target in route:
            position, remainder = move_to(environment, layout, target, args.steps * 2)
            print(f"{label}: target={target.round(4).tolist()} position={position.round(4).tolist()} remainder={remainder:.4f}")
    finally:
        environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
