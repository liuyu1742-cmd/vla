"""Shared action and episode-window contracts for the LIBERO diffusion policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def transform_dataset_gripper(raw: np.ndarray) -> np.ndarray:
    """Convert RLDS gripper actions from ``[-1, 1]`` to OpenVLA's ``[0, 1]``."""

    raw = np.asarray(raw, dtype=np.float32)
    return 1.0 - np.clip(raw, 0.0, 1.0)


def dataset_gripper_to_env(value: np.ndarray) -> np.ndarray:
    """Convert OpenVLA-style gripper values to LIBERO's binary environment action."""

    value = np.asarray(value, dtype=np.float32)
    return np.where(value >= 0.5, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class ActionNormalizer:
    """Per-action robust quantile normalization to the diffusion range ``[-1, 1]``."""

    q01: np.ndarray
    q99: np.ndarray

    @classmethod
    def fit(cls, actions: np.ndarray) -> "ActionNormalizer":
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError("actions must be a non-empty [N, action_dim] array")
        q01 = np.quantile(actions, 0.01, axis=0, method="nearest").astype(
            np.float32
        )
        q99 = np.quantile(actions, 0.99, axis=0, method="nearest").astype(
            np.float32
        )
        return cls(q01=q01, q99=q99)

    @property
    def span(self) -> np.ndarray:
        raw_span = self.q99 - self.q01
        return np.where(raw_span < 1e-6, 1.0, raw_span).astype(np.float32)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        normalized = 2.0 * (actions - self.q01) / self.span - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        clipped = np.clip(actions, -1.0, 1.0)
        return (((clipped + 1.0) * 0.5) * self.span + self.q01).astype(
            np.float32
        )


@dataclass(frozen=True)
class WindowIndex:
    """One padded observation/action window contained within a single episode."""

    episode_index: int
    observation_start: int
    observation_stop: int
    observation_pad_left: int
    action_start: int
    action_stop: int
    action_pad_right: int


def build_window_indices(
    lengths: Sequence[int],
    obs_horizon: int,
    action_horizon: int,
) -> list[WindowIndex]:
    """Build one causal training window per episode step without crossing boundaries."""

    if obs_horizon <= 0 or action_horizon <= 0:
        raise ValueError("obs_horizon and action_horizon must be positive")

    rows: list[WindowIndex] = []
    for episode_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length <= 0:
            raise ValueError("episode lengths must be positive")
        for step in range(length):
            observation_stop = step + 1
            observation_start = max(0, observation_stop - obs_horizon)
            action_start = step
            action_stop = min(length, action_start + action_horizon)
            rows.append(
                WindowIndex(
                    episode_index=episode_index,
                    observation_start=observation_start,
                    observation_stop=observation_stop,
                    observation_pad_left=obs_horizon
                    - (observation_stop - observation_start),
                    action_start=action_start,
                    action_stop=action_stop,
                    action_pad_right=action_horizon
                    - (action_stop - action_start),
                )
            )
    return rows
