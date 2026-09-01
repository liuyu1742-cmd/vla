"""Move one verified HABIT episode into an object-specific training directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def episode_paths(source_root: str | Path, episode_index: int) -> tuple[Path, Path]:
    root = Path(source_root)
    stem = f"episode_{episode_index:06d}.parquet"
    video = f"episode_{episode_index:06d}.mp4"
    return root / "data" / "chunk-000" / stem, root / "videos" / "chunk-000" / "observation.images.exo_view" / video


def archive_episode(source_root: Path, dataset_root: Path, task_dir: str, object_id: str, episode_index: int, instruction: str, operation: str, execute: bool) -> dict:
    parquet, rgb = episode_paths(source_root, episode_index)
    if not parquet.is_file() or not rgb.is_file():
        raise FileNotFoundError(f"missing HABIT pair for episode {episode_index}")
    columns = pd.read_parquet(parquet).columns
    if "action" not in columns:
        raise ValueError(f"episode {episode_index} lacks action field")
    destination = dataset_root / task_dir / object_id
    if destination.exists():
        raise FileExistsError(destination)
    if not execute:
        return {"status": "preflight_passed", "destination": str(destination), "action_field": True, "rgb": str(rgb)}
    destination.mkdir(parents=True, exist_ok=False)
    shutil.move(str(parquet), str(destination / "actions.parquet"))
    shutil.move(str(rgb), str(destination / "rgb_exo_view.mp4"))
    (destination / "instruction.json").write_text(json.dumps({"instruction": instruction, "object_id": object_id}, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "operation.json").write_text(json.dumps({"object": object_id, "operation": operation}, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "source_manifest.json").write_text(json.dumps({"dataset": "HABIT public sample", "episode_index": episode_index, "moved_from": {"parquet": str(parquet.relative_to(source_root)), "rgb": str(rgb.relative_to(source_root))}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "conversion_status.json").write_text(json.dumps({"status": "verified_convertible", "instruction": True, "rgb_video_count": 1, "action_column": "action", "source_files_moved": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "complete", "destination": str(destination), "action_field": True, "rgb_video_count": 1}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = archive_episode(args.source_root, args.dataset_root, args.task_dir, args.object_id, args.episode_index, args.instruction, args.operation, args.execute)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
