"""Build a dual-view, episode-safe OpenVLA-OFT cache from RoboCasa365."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.robocasa_oft_contract import (
    episode_split,
    extract_arm_action,
    extract_proprio,
    future_action_chunk,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "datasets" / "vla82_robocasa365_19class" / "training_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "datasets" / "vla82_robocasa365_oft"
PRIMARY_CAMERA = "observation.images.robot0_agentview_left"
WRIST_CAMERA = "observation.images.robot0_eye_in_hand"


def validate_lengths(
    states: int, actions: int, primary_frames: int, wrist_frames: int
) -> None:
    """Reject any episode whose four synchronized streams disagree."""
    if len({states, actions, primary_frames, wrist_frames}) != 1:
        raise ValueError(
            f"episode alignment mismatch: states={states}, actions={actions}, "
            f"primary={primary_frames}, wrist={wrist_frames}"
        )


def build_aligned_arrays(
    states: np.ndarray, actions: np.ndarray
) -> dict[str, np.ndarray]:
    """Convert source arrays to OFT proprioception and future action chunks."""
    state_values = np.asarray(states, dtype=np.float32)
    action_values = np.asarray(actions, dtype=np.float32)
    if len(state_values) != len(action_values) or len(state_values) == 0:
        raise ValueError("state and action rows must be aligned and non-empty")
    proprio = np.stack([extract_proprio(row) for row in state_values])
    arm = np.stack([extract_arm_action(row) for row in action_values])
    chunks = np.stack([future_action_chunk(arm, index) for index in range(len(arm))])
    return {"proprio": proprio, "actions": arm, "action_chunks": chunks}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_frame_count(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    value = completed.stdout.strip().splitlines()[-1]
    if not value.isdigit():
        raise ValueError(f"ffprobe did not return a frame count for {path}: {value}")
    return int(value)


def _save_arrays(directory: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, values in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, np.asarray(values, dtype=np.float32))
        paths[name] = str(path.resolve())
    return paths


def prepare_cache(
    manifest_path: Path, output: Path, *, limit: int | None = None, seed: int = 82
) -> dict[str, Any]:
    import pandas as pd

    from tools.robocasa_episode_archive_v2 import (
        _cut_video,
        chunk_relative_rows,
        episode_assets,
    )

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = Path(source_manifest["source"])
    selected = list(source_manifest["episodes"])
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    metadata_path = (
        source_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    episodes = pd.read_parquet(metadata_path)
    split_ids = episode_split([int(item["episode_index"]) for item in selected], seed=seed)
    split_by_episode = {
        episode_index: split_name
        for split_name, ids in split_ids.items()
        for episode_index in ids
    }
    output.mkdir(parents=True, exist_ok=True)
    parquet_cache: dict[Path, Any] = {}
    results: list[dict[str, Any]] = []

    for position, item in enumerate(selected, start=1):
        episode_index = int(item["episode_index"])
        matches = episodes[episodes["episode_index"] == episode_index]
        if len(matches) != 1:
            raise ValueError(f"episode {episode_index} did not resolve uniquely")
        episode = matches.iloc[0].to_dict()
        length = int(episode["length"])
        episode_dir = output / str(item["task_class"]) / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        primary_path = Path(item["video"])
        if not primary_path.is_file():
            raise FileNotFoundError(primary_path)
        wrist_assets = episode_assets(source_root, episode, WRIST_CAMERA)
        wrist_path = episode_dir / "rgb_eye_in_hand.mp4"
        if not wrist_path.is_file():
            _cut_video(wrist_assets["video"], wrist_path, wrist_assets["timestamps"])

        primary_assets = episode_assets(source_root, episode, PRIMARY_CAMERA)
        parquet_path = Path(primary_assets["parquet"])
        if parquet_path not in parquet_cache:
            parquet_cache[parquet_path] = pd.read_parquet(
                parquet_path, columns=["observation.state", "action"]
            )
        table = parquet_cache[parquet_path]
        start, end = chunk_relative_rows(episodes, episode)
        rows = table.iloc[start:end]
        states = np.asarray(rows["observation.state"].to_list(), dtype=np.float32)
        actions = np.asarray(rows["action"].to_list(), dtype=np.float32)
        arrays = build_aligned_arrays(states, actions)
        array_paths = _save_arrays(episode_dir, arrays)

        primary_frames = video_frame_count(primary_path)
        wrist_frames = video_frame_count(wrist_path)
        validate_lengths(len(states), len(actions), primary_frames, wrist_frames)
        if length != len(states):
            raise ValueError(
                f"episode {episode_index} metadata length={length}, decoded={len(states)}"
            )
        artifacts = {
            "primary_video": str(primary_path.resolve()),
            "wrist_video": str(wrist_path.resolve()),
            **array_paths,
        }
        results.append(
            {
                "task_class": str(item["task_class"]),
                "episode_index": episode_index,
                "instruction": str(item["instruction"]),
                "split": split_by_episode[episode_index],
                "frames": length,
                "artifacts": artifacts,
                "sha256": {
                    name: sha256_file(Path(path)) for name, path in artifacts.items()
                },
            }
        )
        print(
            f"[{position}/{len(selected)}] {item['task_class']} "
            f"episode={episode_index} frames={length}",
            flush=True,
        )

    payload = {
        "schema_version": "robocasa_oft_cache_v1",
        "source_manifest": str(manifest_path.resolve()),
        "source_root": str(source_root.resolve()),
        "seed": seed,
        "episode_count": len(results),
        "split_counts": {
            name: sum(item["split"] == name for item in results)
            for name in ("train", "val", "test")
        },
        "alignment_failures": 0,
        "primary_camera": PRIMARY_CAMERA,
        "wrist_camera": WRIST_CAMERA,
        "proprio_mapping": "state[7:14] + mean(state[14:16])",
        "action_mapping": "action[5:12]",
        "action_chunk_size": 8,
        "episodes": results,
    }
    manifest_output = output / "cache_manifest.json"
    manifest_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=82)
    args = parser.parse_args(argv)
    payload = prepare_cache(
        args.manifest, args.output, limit=args.limit, seed=args.seed
    )
    print((args.output / "cache_manifest.json").resolve())
    print(json.dumps(payload["split_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
