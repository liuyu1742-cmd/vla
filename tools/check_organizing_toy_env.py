"""Reset and audit the real RoboCasa environment for ``organizing::toy``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "organizing_toy_env_check",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))

    import imageio.v3 as iio

    from tools.skill_transfer.robocasa_assets import inspect_toy_asset
    from tools.skill_transfer.robocasa_envs import create_formal_env

    args.output.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env, mapping = create_formal_env("organizing::toy", seed=args.seed)
        observation, reset_info = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        camera_key = "video.robot0_agentview_left"
        if camera_key not in observation:
            available = sorted(key for key in observation if str(key).startswith("video."))
            raise RuntimeError(
                f"required virtual camera {camera_key!r} is absent; available={available}"
            )
        frame = np.asarray(observation[camera_key], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"virtual camera produced invalid frame shape {frame.shape}")
        image_path = args.output / "reset_frame.png"
        iio.imwrite(image_path, frame)

        obj = raw.objects["obj"]
        body_id = raw.obj_body_id["obj"]
        object_position = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
        report = {
            "schema_version": "formal_env_check_v1",
            "reset_completed": True,
            "task_success_after_reset": bool(raw._check_success()),
            "mapping": mapping,
            "object": {
                "category": raw.object_cfgs[0].get("info", {}).get("cat", "toy"),
                "model_path": str(Path(obj.mjcf_path).resolve()),
                "size": list(obj.size),
                "position": object_position.tolist(),
            },
            "target": {"canonical": "storage", "fixture": "cabinet"},
            "camera": {
                "key": camera_key,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "frame": str(image_path.resolve()),
            },
            "asset": inspect_toy_asset(),
            "reset_info": {str(key): _json_ready(value) for key, value in reset_info.items()},
        }
        report_path = args.output / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(report_path.resolve(), flush=True)
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
