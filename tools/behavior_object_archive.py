"""Move one verified BEHAVIOR-1K RGB/action episode into each object folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED = {"task_dir", "object_id", "episode_index", "instruction", "operation"}


def validate_selection(rows: list[dict[str, Any]]) -> None:
    object_ids: set[tuple[str, str]] = set()
    episodes: set[int] = set()
    for position, row in enumerate(rows):
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"row {position} missing {sorted(missing)}")
        key = (str(row["task_dir"]), str(row["object_id"]))
        if key in object_ids:
            raise ValueError(f"duplicate destination {key}")
        episode = int(row["episode_index"])
        if episode in episodes:
            raise ValueError(f"duplicate source episode {episode}")
        object_ids.add(key)
        episodes.add(episode)


def load_manifest(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "manifest.jsonl"
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[int(record["episode_index"])] = record
    return records


def preflight(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_selection(rows)
    records = load_manifest(root)
    resolved = []
    for row in rows:
        episode = int(row["episode_index"])
        if episode not in records:
            raise ValueError(f"episode {episode} absent from manifest")
        record = records[episode]
        parquet = root / record["parquet_path"]
        rgb = root / record["head_rgb_path"]
        if not parquet.is_file() or not rgb.is_file():
            raise FileNotFoundError(f"episode {episode} is missing parquet or RGB")
        columns = pd.read_parquet(parquet).columns
        if "action" not in columns:
            raise ValueError(f"episode {episode} lacks action label")
        resolved.append({**row, "record": record, "parquet": parquet, "rgb": rgb})
    return resolved


def archive(root: Path, dataset_root: Path, rows: list[dict[str, Any]], execute: bool) -> dict[str, Any]:
    resolved = preflight(root, rows)
    report = {"status": "preflight_passed", "objects": len(resolved), "entries": []}
    for item in resolved:
        destination = dataset_root / item["task_dir"] / item["object_id"]
        if destination.exists():
            raise FileExistsError(destination)
        report["entries"].append({"object": item["object_id"], "episode_index": item["episode_index"], "destination": str(destination)})
    if not execute:
        return report
    for item in resolved:
        destination = dataset_root / item["task_dir"] / item["object_id"]
        destination.mkdir(parents=True, exist_ok=False)
        shutil.move(str(item["parquet"]), str(destination / "actions.parquet"))
        shutil.move(str(item["rgb"]), str(destination / "rgb_head.mp4"))
        (destination / "instruction.json").write_text(json.dumps({"instruction": item["instruction"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "operation.json").write_text(json.dumps({"object": item["object_id"], "operation": item["operation"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "source_manifest.json").write_text(json.dumps({"dataset": "BEHAVIOR-1K 2025 Challenge", "source_episode": item["record"], "moved_from": {"parquet": item["record"]["parquet_path"], "rgb": item["record"]["head_rgb_path"]}}, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "conversion_status.json").write_text(json.dumps({"status": "verified_convertible", "instruction": True, "rgb_video_count": 1, "action_column": "action", "source_files_moved": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    report["status"] = "complete"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.selection.read_text(encoding="utf-8"))
    result = archive(args.source_root, args.dataset_root, rows, args.execute)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "objects": result["objects"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
