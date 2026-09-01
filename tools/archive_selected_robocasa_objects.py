"""Preflight and archive a curated RoboCasa object-selection manifest.

Manifest rows must name the final task directory, unique object directory,
official episode index, exact instruction and operated object.  This keeps the
selection decision separate from video/action extraction and prevents a raw
chunk from being mistaken for an object-specific sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.robocasa_episode_archive_v2 import archive_episode, normalized_object_id


REQUIRED = {"task_dir", "object_id", "episode_index", "instruction", "operation"}


def validate_selection(rows: list[dict[str, Any]]) -> None:
    """Reject incomplete rows, duplicate objects, and duplicate source episodes."""
    object_dirs: set[tuple[str, str]] = set()
    episodes: set[int] = set()
    for position, row in enumerate(rows):
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"row {position} missing {sorted(missing)}")
        key = (str(row["task_dir"]), normalized_object_id(str(row["object_id"])))
        if key in object_dirs:
            raise ValueError(f"duplicate object destination {key}")
        episode = int(row["episode_index"])
        if episode in episodes:
            raise ValueError(f"source episode {episode} allocated twice")
        object_dirs.add(key)
        episodes.add(episode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually derive video/action files after preflight.")
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_selection(rows)
    planned = []
    for row in rows:
        destination = args.dataset_root / str(row["task_dir"]) / normalized_object_id(str(row["object_id"]))
        if destination.exists():
            raise FileExistsError(destination)
        planned.append({"destination": str(destination), "episode_index": int(row["episode_index"])})
    if not args.execute:
        print(json.dumps({"status": "preflight_passed", "objects": len(rows), "planned": planned}, ensure_ascii=False))
        return 0
    completed = []
    for row in rows:
        destination = args.dataset_root / str(row["task_dir"]) / normalized_object_id(str(row["object_id"]))
        status = archive_episode(args.source_root, destination, int(row["episode_index"]), str(row["instruction"]), str(row["object_id"]))
        completed.append({"destination": str(destination), **status})
    print(json.dumps({"status": "complete", "objects": len(completed), "completed": completed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
