"""Data-shape contract shared by RoboCasa365 and OpenVLA-OFT."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np


STATE_DIM = 16
SOURCE_ACTION_DIM = 12
ARM_ACTION_DIM = 7
PROPRIO_DIM = 8
DEFAULT_CHUNK_SIZE = 8


def extract_proprio(state: Sequence[float]) -> np.ndarray:
    """Return relative EEF pose plus scalar gripper state (8 channels)."""
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (STATE_DIM,):
        raise ValueError("RoboCasa365 state must contain 16 values")
    return np.concatenate((values[7:14], [values[14:16].mean()])).astype(
        np.float32
    )


def extract_arm_action(action: Sequence[float]) -> np.ndarray:
    """Drop mobile-base and mode channels from the official 12-D action."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (SOURCE_ACTION_DIM,):
        raise ValueError("RoboCasa365 action must contain 12 values")
    return values[5:12].copy()


def future_action_chunk(
    actions: np.ndarray, index: int, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> np.ndarray:
    """Return a future chunk, clamping indices to the episode's final action."""
    values = np.asarray(actions, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != ARM_ACTION_DIM
        or not 0 <= index < len(values)
    ):
        raise ValueError("actions must be non-empty Nx7 and index must be valid")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    indices = np.minimum(np.arange(index, index + chunk_size), len(values) - 1)
    return values[indices].copy()


def episode_split(
    episode_ids: Sequence[int], seed: int = 82
) -> dict[str, list[int]]:
    """Create deterministic, disjoint 80/10/10 episode-level splits."""
    ids = sorted({int(value) for value in episode_ids})
    if len(ids) < 3:
        raise ValueError("at least three episodes are required")
    random.Random(seed).shuffle(ids)
    train_end = max(1, round(0.8 * len(ids)))
    train_end = min(train_end, len(ids) - 2)
    held = ids[train_end:]
    val_end = max(1, len(held) // 2)
    return {
        "train": sorted(ids[:train_end]),
        "val": sorted(held[:val_end]),
        "test": sorted(held[val_end:]),
    }
