"""Build exact, episode-level evidence for a proposed RoboCasa object phrase.

The output only includes rows with an official instruction, source RGB chunks and
an action Parquet chunk.  It deliberately does not infer object names from
images or use background objects as training targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


def verified_phrase_rows(root: Path, phrase: str) -> list[dict[str, object]]:
    """Return complete source-paired rows whose instruction explicitly names phrase."""
    episodes = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    needle = phrase.casefold().strip()
    matched: list[dict[str, object]] = []
    for row in episodes.itertuples(index=False):
        instruction = str(row.tasks[0]) if row.tasks else ""
        if not needle or needle not in instruction.casefold():
            continue
        data_path = root / f"data/chunk-{getattr(row, '_5'):03d}/file-{getattr(row, '_6'):03d}.parquet"
        # Column names with slashes are converted by itertuples, so use the
        # original mapping only for paths below.
        source = episodes.loc[episodes["episode_index"] == row.episode_index].iloc[0]
        data_path = root / f"data/chunk-{int(source['data/chunk_index']):03d}/file-{int(source['data/file_index']):03d}.parquet"
        cameras = []
        for camera in CAMERAS:
            video = root / f"videos/{camera}/chunk-{int(source[f'videos/{camera}/chunk_index']):03d}/file-{int(source[f'videos/{camera}/file_index']):03d}.mp4"
            if video.is_file():
                cameras.append(camera)
        if data_path.is_file() and len(cameras) == len(CAMERAS):
            matched.append({
                "episode_index": int(source["episode_index"]),
                "instruction": instruction,
                "source_prefix": str(source["source_prefix"]),
                "row_range": [int(source["dataset_from_index"]), int(source["dataset_to_index"])],
                "length": int(source["length"]),
                "camera_count": len(cameras),
                "action_parquet": str(data_path.relative_to(root)).replace("\\", "/"),
            })
    return matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {"phrase": args.phrase, "records": verified_phrase_rows(args.root, args.phrase)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phrase": args.phrase, "verified_episodes": len(payload["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
