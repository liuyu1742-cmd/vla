"""Memory-bounded loader for formal Skill IR expert demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FormalSkillDataset:
    """Expose per-frame 7D actions while keeping held-out seeds isolated."""

    def __init__(self, manifest_path: Path, *, split: str = "train") -> None:
        if split not in {"train", "held_out"}:
            raise ValueError("split must be 'train' or 'held_out'")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "formal_skill_training_manifest_v1":
            raise ValueError("unsupported formal skill manifest schema")
        if self.manifest.get("relation_key") != "organizing::toy":
            raise ValueError("formal dataset relation must be organizing::toy")
        train = self.manifest.get("train")
        held_out = self.manifest.get("held_out")
        if not isinstance(train, list) or not isinstance(held_out, list):
            raise ValueError("manifest train and held_out must be lists")
        train_seeds = {int(item["seed"]) for item in train}
        heldout_seeds = {int(item["seed"]) for item in held_out}
        overlap = train_seeds & heldout_seeds
        if overlap:
            raise ValueError(f"train and held-out seeds overlap: {sorted(overlap)}")

        selected = train if split == "train" else held_out
        self.split = split
        self.episodes: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        for raw_entry in selected:
            entry = dict(raw_entry)
            path = Path(str(entry["episode"]))
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"formal episode is missing: {path}")
            with np.load(path, allow_pickle=False) as episode:
                frames_shape = episode["frames"].shape
                actions_shape = episode["actions"].shape
                phase_shape = episode["phases"].shape
                canonical_shape = episode["canonical_phases"].shape
                relation = str(episode["relation_key"].item())
                instruction = str(episode["instruction"].item())
            if len(frames_shape) != 4 or frames_shape[-1] != 3:
                raise ValueError(f"episode frames have invalid shape: {frames_shape}")
            samples = int(frames_shape[0])
            if actions_shape != (samples, 7):
                raise ValueError(f"episode actions have invalid shape: {actions_shape}")
            if phase_shape != (samples,) or canonical_shape != (samples,):
                raise ValueError("episode phase arrays must align with frames")
            if relation != self.manifest["relation_key"]:
                raise ValueError("episode relation differs from manifest")
            if int(entry.get("samples", samples)) != samples:
                raise ValueError("episode sample count differs from manifest")
            if not instruction:
                raise ValueError("episode instruction is empty")
            episode_index = len(self.episodes)
            entry.update(
                {
                    "episode": path,
                    "samples": samples,
                    "relation_key": relation,
                    "instruction": instruction,
                }
            )
            self.episodes.append(entry)
            self.index.extend((episode_index, frame_index) for frame_index in range(samples))

        self._cache_episode_index: int | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if self._cache_episode_index == episode_index and self._cache is not None:
            return self._cache
        path = self.episodes[episode_index]["episode"]
        with np.load(path, allow_pickle=False) as episode:
            cache = {
                "frames": np.asarray(episode["frames"], dtype=np.uint8),
                "actions": np.asarray(episode["actions"], dtype=np.float32),
                "phases": np.asarray(episode["phases"]),
                "canonical_phases": np.asarray(episode["canonical_phases"]),
            }
        self._cache_episode_index = episode_index
        self._cache = cache
        return cache

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self.index[index]
        entry = self.episodes[episode_index]
        episode = self._load_episode(episode_index)
        return {
            "frame": episode["frames"][frame_index],
            "action": episode["actions"][frame_index],
            "phase": str(episode["phases"][frame_index]),
            "canonical_phase": str(episode["canonical_phases"][frame_index]),
            "instruction": entry["instruction"],
            "relation_key": entry["relation_key"],
            "seed": int(entry["seed"]),
            "frame_index": frame_index,
        }


__all__ = ["FormalSkillDataset"]
