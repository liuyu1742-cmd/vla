"""Render one RoboCasa virtual-camera frame for the OpenVLA IPC loop."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "openvla_robocasa_ipc"


def image_path_for_step(step: int) -> Path:
    return OUTPUT_DIR / f"frame_{step:04d}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import imageio.v3 as iio
    from robocasa.utils.env_utils import create_env

    env = create_env("PickPlaceCounterToCabinet", camera_names=["robot0_agentview_left"], seed=args.seed)
    try:
        env.reset()
        frame = env.sim.render(height=224, width=224, camera_name="robot0_agentview_left")[::-1]
        output = image_path_for_step(args.step)
        output.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output, frame)
        print(output)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
