"""Export an actual recorded VLA82 simulation camera stream to MP4.

The script refuses episodes that did not satisfy the strict simulator
predicate, so failed trajectories cannot be presented as result videos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


def output_path_for_episode(episode_path: Path, camera: str) -> Path:
    return episode_path.with_suffix(f".{camera}.mp4")


def require_strict_pass(result: Mapping[str, object]) -> None:
    if result.get("status") != "PASS" or result.get("predicate_success") is not True:
        raise ValueError("refusing to export a non-strict-PASS episode as a result video")


def export_episode(episode_path: Path, *, camera: str, fps: float) -> Path:
    result_path = episode_path.with_suffix(".json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require_strict_pass(result)
    with np.load(episode_path) as data:
        frames = np.asarray(data[camera], dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"{episode_path}: camera {camera!r} is not an RGB frame sequence")
    height, width = map(int, frames.shape[1:3])
    output = output_path_for_episode(episode_path, camera)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open MP4 writer for {output}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"MP4 writer did not create {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path, help="strict PASS episode .npz files")
    parser.add_argument("--camera", default="primary")
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    for episode in args.episodes:
        print(export_episode(episode.resolve(), camera=args.camera, fps=args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
