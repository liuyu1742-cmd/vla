"""Diagnostic: replay a public RoboCasa365 action sequence in a VLA82 scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.robocasa_oft_rollout import _mapping


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=824141)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "outputs/midterm_testing_vla82/simulator_mapping_plan.json",
    )
    args = parser.parse_args()
    sys.path[:0] = [
        str(ROOT / "third_party/robosuite"),
        str(ROOT / "third_party/robocasa"),
    ]
    from tools.vla82_pure_closed_loop import _configured_environment

    mapping = _mapping(args.plan, args.selection_id)
    actions = np.load(args.actions, allow_pickle=False)
    success = False
    success_step = None
    with _configured_environment(mapping, args.seed) as environment:
        _, _ = environment.reset(seed=args.seed)
        raw = environment.unwrapped.env
        for step, action in enumerate(actions):
            _, _, terminated, truncated, info = environment.step(
                to_robocasa_action(action)
            )
            success = bool(info.get("success", False) or raw._check_success())
            if success or terminated or truncated:
                success_step = step if success else None
                break
        final_success = bool(raw._check_success())
        success = bool(success or final_success)
    print(
        f"PUBLIC_REPLAY selection={args.selection_id} success={success} "
        f"success_step={success_step} actions={len(actions)}",
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
