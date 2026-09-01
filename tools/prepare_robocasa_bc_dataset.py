"""Convert RoboCasa LeRobot episodes into episode-safe BC train/val/test files."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    WORKSPACE
    / "third_party"
    / "robocasa"
    / "datasets"
    / "v1.0"
    / "pretrain"
    / "atomic"
    / "PickPlaceCounterToCabinet"
    / "20250819"
    / "lerobot"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "robocasa_bc_pick_place"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parquet_path(dataset: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def make_episode_splits(
    episode_tasks: dict[int, int], train_count: int, seed: int
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    by_task: dict[int, list[int]] = defaultdict(list)
    for episode_index, task_index in episode_tasks.items():
        by_task[task_index].append(episode_index)

    required_train: list[int] = []
    for task_index in sorted(by_task):
        required_train.append(rng.choice(sorted(by_task[task_index])))
    if len(required_train) > train_count:
        raise ValueError(
            f"Cannot cover {len(required_train)} tasks with {train_count} training episodes"
        )

    remaining = sorted(set(episode_tasks) - set(required_train))
    rng.shuffle(remaining)
    fill_count = train_count - len(required_train)
    train = sorted(required_train + remaining[:fill_count])
    held_out = remaining[fill_count:]
    val_count = len(held_out) // 2
    return {
        "train": train,
        "val": sorted(held_out[:val_count]),
        "test": sorted(held_out[val_count:]),
    }


def load_episode(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(
        path,
        columns=[
            "observation.state",
            "action",
            "next.reward",
            "next.done",
            "frame_index",
            "episode_index",
            "task_index",
        ],
    )
    rows = table.num_rows
    return {
        "states": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
        "actions": np.asarray(table["action"].to_pylist(), dtype=np.float32),
        "rewards": np.asarray(table["next.reward"].to_numpy(), dtype=np.float32),
        "dones": np.asarray(table["next.done"].to_numpy(), dtype=np.bool_),
        "frame_indices": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
        "episode_indices": np.asarray(table["episode_index"].to_numpy(), dtype=np.int64),
        "task_indices": np.asarray(table["task_index"].to_numpy(), dtype=np.int64),
        "progress": np.linspace(0.0, 1.0, rows, dtype=np.float32),
    }


def concatenate_episodes(dataset: Path, episodes: list[int]) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for position, episode_index in enumerate(episodes, start=1):
        episode = load_episode(parquet_path(dataset, episode_index))
        for key, values in episode.items():
            collected[key].append(values)
        print(f"[{position}/{len(episodes)}] loaded episode_{episode_index:06d}")
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def main() -> None:
    args = parse_args()
    info = read_json(args.dataset / "meta" / "info.json")
    tasks = read_jsonl(args.dataset / "meta" / "tasks.jsonl")
    total_episodes = int(info["total_episodes"])
    if not 0.5 <= args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in [0.5, 1.0)")

    episode_tasks: dict[int, int] = {}
    for episode_index in range(total_episodes):
        table = pq.read_table(parquet_path(args.dataset, episode_index), columns=["task_index"])
        episode_tasks[episode_index] = int(table["task_index"][0].as_py())

    train_count = round(total_episodes * args.train_ratio)
    splits = make_episode_splits(episode_tasks, train_count, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        print(f"Preparing {split_name}: {len(splits[split_name])} episodes")
        arrays[split_name] = concatenate_episodes(args.dataset, splits[split_name])
        np.savez_compressed(args.output / f"{split_name}.npz", **arrays[split_name])

    train_states = arrays["train"]["states"]
    state_mean = train_states.mean(axis=0)
    state_std = np.maximum(train_states.std(axis=0), 1e-6)
    all_actions = np.concatenate([arrays[name]["actions"] for name in arrays], axis=0)
    task_text_by_index = {int(item["task_index"]): item["task"] for item in tasks}
    metadata = {
        "source_dataset": str(args.dataset.resolve()),
        "seed": args.seed,
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "total_episodes": total_episodes,
        "total_frames": info["total_frames"],
        "num_tasks": len(tasks),
        "state_dim": int(train_states.shape[1]),
        "action_dim": int(all_actions.shape[1]),
        "splits": splits,
        "split_frames": {name: int(values["states"].shape[0]) for name, values in arrays.items()},
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_min": all_actions.min(axis=0).tolist(),
        "action_max": all_actions.max(axis=0).tolist(),
        "task_text_by_index": {str(key): value for key, value in task_text_by_index.items()},
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Prepared {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
        "train/val/test episodes"
    )
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
