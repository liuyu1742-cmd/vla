"""Export only language/RGB/action-paired candidates for household curation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def robocasa_candidates(root: Path) -> list[dict[str, object]]:
    episodes = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    records: list[dict[str, object]] = []
    for instruction, group in episodes.assign(instruction=episodes.tasks.map(lambda value: value[0] if value else "")).groupby("instruction", sort=True):
        first = group.iloc[0]
        records.append({
            "dataset": "RoboCasa365 official", "instruction": str(instruction),
            "episode_index": int(first["episode_index"]), "available_episodes": int(len(group)),
            "rgb_cameras": 3, "action_format": "12D robot action Parquet", "trainability": "conversion_required",
            "source_prefix": str(first["source_prefix"]),
        })
    return records


def existing_candidates(registry: Path) -> list[dict[str, object]]:
    result = []
    for row in json.loads(registry.read_text(encoding="utf-8")):
        if int(row.get("pair_count", 0)):
            result.append({
                "dataset": row["dataset"], "instruction": row["instruction"], "episode_index": None,
                "available_episodes": int(row["pair_count"]), "rgb_cameras": 1,
                "action_format": "robot action/state Parquet", "trainability": "conversion_required",
                "source_prefix": f"task-{int(row['task_index']):04d}",
            })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robocasa-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = robocasa_candidates(args.robocasa_root) + existing_candidates(args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": len(records), "robocasa": sum(row['dataset'] == 'RoboCasa365 official' for row in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
