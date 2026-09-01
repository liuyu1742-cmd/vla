"""Extract source-verified RoboCasa episodes using chunk-relative action slices.

The official RoboCasa365 LeRobot release stores actions and RGB streams in
shared chunks.  This version converts one exact episode to its own directory
without moving or deleting any raw source file.
"""

from __future__ import annotations

import argparse
import json
import re
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
    value = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", value.strip().casefold(), flags=re.UNICODE)
    return value.strip("_")


def episode_assets(root: Path, episode: dict[str, Any], camera: str) -> dict[str, Any]:
    return {
        "parquet": root / f"data/chunk-{int(episode['data/chunk_index']):03d}/file-{int(episode['data/file_index']):03d}.parquet",
        "video": root / f"videos/{camera}/chunk-{int(episode[f'videos/{camera}/chunk_index']):03d}/file-{int(episode[f'videos/{camera}/file_index']):03d}.mp4",
        "global_rows": (int(episode["dataset_from_index"]), int(episode["dataset_to_index"])),
        "timestamps": (float(episode[f"videos/{camera}/from_timestamp"]), float(episode[f"videos/{camera}/to_timestamp"])),
    }


def chunk_relative_rows(episodes: pd.DataFrame, episode: dict[str, Any]) -> tuple[int, int]:
    """Map global indices to the selected shared Parquet file's row indices."""
    same_file = episodes[
        (episodes["data/chunk_index"] == int(episode["data/chunk_index"]))
        & (episodes["data/file_index"] == int(episode["data/file_index"]))
    ]
    start_global = int(episode["dataset_from_index"])
    end_global = int(episode["dataset_to_index"])
    chunk_global_start = int(same_file["dataset_from_index"].min())
    return start_global - chunk_global_start, end_global - chunk_global_start


def _cut_video(source: Path, dest: Path, timestamps: tuple[float, float]) -> None:
    start, end = timestamps
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{end - start:.6f}",
         "-map", "0:v:0", "-an", "-c:v", "libx264", "-crf", "18", str(dest)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )


def archive_episode(source_root: Path, output_dir: Path, episode_index: int, instruction: str, object_id: str) -> dict[str, Any]:
    meta = source_root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(meta)
    selected = episodes[episodes["episode_index"] == episode_index]
    if len(selected) != 1:
        raise ValueError(f"episode index {episode_index} did not resolve uniquely")
    episode = selected.iloc[0].to_dict()
    output_dir.mkdir(parents=True, exist_ok=False)
    assets = episode_assets(source_root, episode, CAMERAS[0])
    source_actions = pd.read_parquet(assets["parquet"])
    start, end = chunk_relative_rows(episodes, episode)
    actions = source_actions.iloc[start:end].copy()
    if len(actions) != int(episode["length"]) or "action" not in actions.columns:
        raise ValueError("action slice does not match official episode metadata")
    actions.to_parquet(output_dir / "actions.parquet", index=False)
    cameras: dict[str, Any] = {}
    for camera in CAMERAS:
        current = episode_assets(source_root, episode, camera)
        if not current["video"].is_file():
            raise FileNotFoundError(current["video"])
        derived = output_dir / f"rgb_{camera.rsplit('.', 1)[-1]}.mp4"
        _cut_video(current["video"], derived, current["timestamps"])
        if not derived.is_file() or derived.stat().st_size <= 0:
            raise ValueError(f"failed to derive {camera}")
        cameras[camera] = {
            "source_video": str(current["video"].relative_to(source_root)).replace("\\", "/"),
            "timestamp_range": list(current["timestamps"]), "derived_video": derived.name,
        }
    (output_dir / "instruction.json").write_text(json.dumps({"instruction": instruction, "object_id": normalized_object_id(object_id)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "source_manifest.json").write_text(json.dumps({
        "dataset": "ember-lab-berkeley/robocasa365-pretrain-atomic", "episode_index": episode_index,
        "source_prefix": episode["source_prefix"], "source_episode_index": int(episode["source_episode_index"]),
        "global_row_range": list(assets["global_rows"]), "chunk_relative_row_range": [start, end], "cameras": cameras,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    status = {"status": "verified_convertible", "action_rows": len(actions), "rgb_video_count": len(cameras), "source_chunks_preserved": True}
    (output_dir / "conversion_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--object-id", required=True)
    args = parser.parse_args()
    print(json.dumps(archive_episode(args.source_root, args.output_dir, args.episode_index, args.instruction, args.object_id), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
