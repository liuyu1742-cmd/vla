"""Execute a saved OpenVLA action in a raw RoboCasa environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE = ROOT / "outputs" / "openvla_robocasa_ipc" / "response_0000.json"
DEFAULT_REPORT = ROOT / "outputs" / "openvla_robocasa_ipc" / "execution_0000.json"


def raw_robocasa_action(openvla_action: Sequence[float]) -> list[float]:
    """Clamp OpenVLA's 7-D command and pad it for raw robosuite."""
    if len(openvla_action) != 7:
        raise ValueError(f"OpenVLA action must contain 7 values, got {len(openvla_action)}")
    return [min(1.0, max(-1.0, float(value))) for value in openvla_action] + [0.0] * 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    response = json.loads(args.response.read_text(encoding="utf-8"))
    action = raw_robocasa_action(response["raw_action"])

    from robocasa.utils.env_utils import create_env

    env = create_env("PickPlaceCounterToCabinet", camera_names=["robot0_agentview_left"], seed=args.seed)
    try:
        env.reset()
        _, _, _, info = env.step(action)
        report = {
            "response": str(args.response),
            "native_action": action,
            "native_action_dimension": len(action),
            "success": bool(info.get("success", False)),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(args.report)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
