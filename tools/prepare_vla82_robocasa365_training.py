"""Prepare a balanced 19-class RoboCasa365 subset for OpenVLA LoRA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "robocasa365_pretrain_atomic"
DEFAULT_OUTPUT = ROOT / "datasets" / "vla82_robocasa365_19class"
TASK_CLASSES = (
    "PickPlaceCounterToSink",
    "PickPlaceCounterToCabinet",
    "PickPlaceCabinetToCounter",
    "PickPlaceCounterToDrawer",
    "TurnOnStove",
    "CloseOven",
    "SlideToasterOvenRack",
    "SlideOvenRack",
    "PickPlaceFridgeDrawerToShelf",
    "CloseFridgeDrawer",
    "CloseFridge",
    "CloseMicrowave",
    "CloseDrawer",
    "CloseCabinet",
    "SlideDishwasherRack",
    "CloseDishwasher",
    "OpenToasterOvenDoor",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
)


def extract_openvla_action(action: Sequence[float]) -> np.ndarray:
    """Extract XYZ, rotation and gripper from official LeRobot action ordering."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (12,):
        raise ValueError("official RoboCasa365 action must contain 12 values")
    return values[5:12].copy()


def select_balanced_episodes(
    rows: Sequence[dict[str, Any]],
    task_classes: Sequence[str],
    *,
    episodes_per_class: int,
) -> list[dict[str, Any]]:
    if episodes_per_class < 1:
        raise ValueError("episodes_per_class must be positive")
    selected: list[dict[str, Any]] = []
    for task_class in task_classes:
        marker = f"/{task_class}/"
        matches = sorted(
            (
                dict(row)
                for row in rows
                if marker in str(row.get("source_prefix", ""))
            ),
            key=lambda row: int(row["episode_index"]),
        )
        if len(matches) < episodes_per_class:
            raise ValueError(
                f"{task_class} has {len(matches)} episodes, needs {episodes_per_class}"
            )
        for row in matches[:episodes_per_class]:
            row["task_class"] = task_class
            selected.append(row)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-class", type=int, default=1)
    args = parser.parse_args(argv)

    import pandas as pd
    from tools.robocasa_episode_archive_v2 import (
        CAMERAS,
        _cut_video,
        chunk_relative_rows,
        episode_assets,
    )

    metadata_path = args.source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pd.read_parquet(metadata_path)
    selected = select_balanced_episodes(
        episodes.to_dict("records"),
        TASK_CLASSES,
        episodes_per_class=args.episodes_per_class,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    camera = CAMERAS[0]
    for position, selected_row in enumerate(selected, start=1):
        episode_index = int(selected_row["episode_index"])
        task_class = str(selected_row["task_class"])
        episode = episodes[episodes["episode_index"] == episode_index].iloc[0].to_dict()
        episode_dir = args.output / task_class / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        actions_path = episode_dir / "actions_openvla.npy"
        video_path = episode_dir / "rgb_agentview_left.mp4"
        if not actions_path.is_file():
            assets = episode_assets(args.source, episode, camera)
            source_actions = pd.read_parquet(assets["parquet"], columns=["action"])
            start, end = chunk_relative_rows(episodes, episode)
            action_rows = np.stack(source_actions.iloc[start:end]["action"].to_list())
            actions = np.stack([extract_openvla_action(row) for row in action_rows])
            if len(actions) != int(episode["length"]):
                raise RuntimeError(f"episode {episode_index} action length mismatch")
            np.save(actions_path, actions.astype(np.float32))
        if not video_path.is_file():
            assets = episode_assets(args.source, episode, camera)
            _cut_video(assets["video"], video_path, assets["timestamps"])
        instruction_values = episode["tasks"]
        instruction = str(instruction_values[0] if len(instruction_values) else "").strip()
        entry = {
            "task_class": task_class,
            "episode_index": episode_index,
            "instruction": instruction,
            "length": int(episode["length"]),
            "actions": str(actions_path.resolve()),
            "video": str(video_path.resolve()),
            "source_prefix": str(episode["source_prefix"]),
            "official_successful_demonstration": True,
        }
        manifest_entries.append(entry)
        print(
            f"[{position}/{len(selected)}] {task_class} episode={episode_index} "
            f"frames={entry['length']}",
            flush=True,
        )
    manifest = {
        "schema_version": "vla82_robocasa365_19class_training_v1",
        "source": str(args.source.resolve()),
        "task_class_count": len(TASK_CLASSES),
        "episodes_per_class": args.episodes_per_class,
        "episode_count": len(manifest_entries),
        "openvla_action_mapping": {
            "source_indices": [5, 6, 7, 8, 9, 10, 11],
            "channels": [
                "x", "y", "z", "rx", "ry", "rz", "gripper_close"
            ],
        },
        "episodes": manifest_entries,
    }
    manifest_path = args.output / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
