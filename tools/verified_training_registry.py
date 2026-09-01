"""Create a strict local registry of paired household robot demonstrations.

This tool intentionally registers episodes before objects: an object is allowed
into the final 120-object catalogue only when it points back to one of these
records.  This prevents task-name-only catalogues from being mistaken for
training data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def has_trainable_triplet(instruction: str, rgb_path: Path, action_path: Path) -> bool:
    """A minimum training record needs text, a non-empty RGB file and actions."""
    return bool(
        instruction.strip()
        and rgb_path.is_file()
        and rgb_path.stat().st_size > 0
        and action_path.is_file()
        and action_path.stat().st_size > 0
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def behavior_records(root: Path) -> list[dict[str, Any]]:
    base = root / "datasets" / "behavior_1k" / "2025_challenge_household_subset"
    rows = _read_jsonl(base / "manifest.jsonl")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["task_index"]), str(row["instruction"]))].append(row)

    records: list[dict[str, Any]] = []
    for (task_index, instruction), episodes in sorted(grouped.items()):
        valid = [
            episode
            for episode in episodes
            if has_trainable_triplet(
                instruction,
                base / episode["head_rgb_path"],
                base / episode["parquet_path"],
            )
        ]
        records.append(
            {
                "dataset": "BEHAVIOR-1K 2025 Challenge",
                "task_index": task_index,
                "instruction": instruction,
                "pair_count": len(valid),
                "status": "conversion_required" if valid else "not_trainable",
                "rgb_stream": "observation.images.rgb.head MP4",
                "action_stream": "R1Pro action Parquet (23D)",
                "example_episode": valid[0]["episode_index"] if valid else None,
                "example_rgb": valid[0]["head_rgb_path"] if valid else None,
                "example_action": valid[0]["parquet_path"] if valid else None,
            }
        )
    return records


def habit_records(root: Path) -> list[dict[str, Any]]:
    base = root / "datasets" / "habit_sample" / "sample"
    episodes = _read_jsonl(base / "meta" / "episodes.jsonl")
    records: list[dict[str, Any]] = []
    for episode in episodes:
        index = int(episode["episode_index"])
        instruction = str(episode.get("high_level_instruction") or episode.get("tasks", [""])[0])
        action = base / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        rgb = base / "videos" / "chunk-000" / "observation.images.head" / f"episode_{index:06d}.mp4"
        if not rgb.is_file():
            candidates = sorted((base / "videos").rglob(f"episode_{index:06d}.mp4"))
            rgb = candidates[0] if candidates else rgb
        valid = has_trainable_triplet(instruction, rgb, action)
        records.append(
            {
                "dataset": "HABIT public sample",
                "task_index": int(episode.get("task_index", -1)),
                "instruction": instruction,
                "pair_count": 1 if valid else 0,
                "status": "conversion_required" if valid else "not_trainable",
                "rgb_stream": str(rgb.relative_to(base)) if rgb.is_file() else None,
                "action_stream": str(action.relative_to(base)) if action.is_file() else None,
                "example_episode": index,
                "example_rgb": str(rgb.relative_to(base)) if rgb.is_file() else None,
                "example_action": str(action.relative_to(base)) if action.is_file() else None,
            }
        )
    return records


def build_registry(root: Path) -> list[dict[str, Any]]:
    return behavior_records(root) + habit_records(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict paired-demo registry for household VLA work.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = build_registry(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "trainable_records": sum(x["pair_count"] > 0 for x in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
