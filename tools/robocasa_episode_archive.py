"""Create verifiable object-specific derived samples from RoboCasa LeRobot chunks.

RoboCasa365 stores many episodes in shared Parquet and MP4 chunks.  This tool
extracts one complete episode into an object directory, retaining source IDs,
the original row/time ranges, and a per-file verification manifest.  It never
modifies or deletes a source chunk.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


def normalized_object_id(value: str) -> str:
    """Return a predictable folder-safe object identifier."""
    value = value.strip().casefold()
    value = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", value, flags=re.UNICODE)
    return value.strip("_")


def episode_assets(root: Path, episode: dict[str, Any], camera: str) -> dict[str, Any]:
    """Resolve source asset paths plus exact row/time ranges from one episode."""
    data_chunk = int(episode["data/chunk_index"])
    data_file = int(episode["data/file_index"])
    video_chunk = int(episode[f"videos/{camera}/chunk_index"])
    video_file = int(episode[f"videos/{camera}/file_index"])
    return {
        "parquet": root / f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet",
        "video": root / f"videos/{camera}/chunk-{video_chunk:03d}/file-{video_file:03d}.mp4",
        "frame_slice": (int(episode["dataset_from_index"]), int(episode["dataset_to_index"])),
        "time_slice": (
            float(episode[f"videos/{camera}/from_timestamp"]),
            float(episode[f"videos/{camera}/to_timestamp"]),
        ),
    }


def _episode_row(metadata_path: Path, episode_index: int) -> dict[str, Any]:
    rows = pd.read_parquet(metadata_path, filters=[("episode_index", "=", episode_index)])
    if len(rows) != 1:
        raise ValueError(f"expected exactly one episode for index {episode_index}, found {len(rows)}")
    return rows.iloc[0].to_dict()


def _run_ffmpeg(source: Path, destination: Path, start: float, end: float) -> None:
    duration = end - start
    command = [
        "ffmpeg", "-y", "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(destination),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def archive_episode(source_root: Path, output_dir: Path, episode_index: int, instruction: str, object_id: str) -> dict[str, Any]:
    """Extract all three cameras and exact Parquet rows for one verified episode."""
    metadata_path = source_root / "meta/episodes/chunk-000/file-000.parquet"
    episode = _episode_row(metadata_path, episode_index)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_manifest: dict[str, Any] = {
        "dataset": "ember-lab-berkeley/robocasa365-pretrain-atomic",
        "episode_index": int(episode_index),
        "source_prefix": str(episode["source_prefix"]),
        "source_episode_index": int(episode["source_episode_index"]),
        "row_range": [int(episode["dataset_from_index"]), int(episode["dataset_to_index"])],
        "cameras": {},
    }
    left_assets = episode_assets(source_root, episode, CAMERAS[0])
    rows = pd.read_parquet(left_assets["parquet"])
    start_row, end_row = left_assets["frame_slice"]
    actions = rows.iloc[start_row:end_row].copy()
    if len(actions) != int(episode["length"]):
        raise ValueError(f"row count mismatch: expected {episode['length']}, found {len(actions)}")
    actions_path = output_dir / "actions.parquet"
    actions.to_parquet(actions_path, index=False)
    for camera in CAMERAS:
        assets = episode_assets(source_root, episode, camera)
        if not assets["parquet"].is_file() or not assets["video"].is_file():
            raise FileNotFoundError(f"missing source asset for {camera}")
        camera_name = camera.rsplit(".", 1)[-1]
        destination = output_dir / f"rgb_{camera_name}.mp4"
        _run_ffmpeg(assets["video"], destination, *assets["time_slice"])
        if not destination.is_file() or destination.stat().st_size == 0:
            raise ValueError(f"video extraction failed for {camera}")
        source_manifest["cameras"][camera] = {
            "source_video": str(assets["video"].relative_to(source_root)).replace("\\", "/"),
            "timestamp_range": list(assets["time_slice"]),
            "derived_video": destination.name,
            "bytes": destination.stat().st_size,
        }
    (output_dir / "instruction.json").write_text(json.dumps({"instruction": instruction, "object_id": object_id}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "source_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    status = {
        "status": "verified_convertible",
        "rgb_video_count": len(CAMERAS),
        "action_rows": len(actions),
        "action_column_present": "action" in actions.columns,
        "source_chunks_preserved": True,
    }
    (output_dir / "conversion_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--object-id", required=True)
    args = parser.parse_args()
    status = archive_episode(args.source_root, args.output_dir, args.episode_index, args.instruction, normalized_object_id(args.object_id))
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
