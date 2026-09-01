"""Progress guard for combining OpenVLA proposals with a verified safe controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GuardDecision:
    action: np.ndarray
    mode: str
    intervened: bool
    translation_cosine: float | None


def _action(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (7,):
        raise ValueError(f"{name} must contain seven values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def select_progress_guarded_action(
    raw_action: Sequence[float],
    expert_action: Sequence[float],
    *,
    force_expert: bool,
    model_weight: float = 0.15,
    minimum_cosine: float = 0.20,
) -> GuardDecision:
    """Keep a small model residual only when it agrees with safe progress.

    Rotation and gripper channels always come from the verified controller. This
    prevents an uncalibrated VLA proposal from opening the gripper in transit.
    """

    raw = _action(raw_action, "raw_action")
    expert = _action(expert_action, "expert_action")
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be in [0, 1]")
    if not -1.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must be in [-1, 1]")
    if force_expert:
        return GuardDecision(expert.copy(), "forced_expert", True, None)

    raw_norm = float(np.linalg.norm(raw[:3]))
    expert_norm = float(np.linalg.norm(expert[:3]))
    cosine = None
    if raw_norm > 1e-8 and expert_norm > 1e-8:
        cosine = float(np.dot(raw[:3], expert[:3]) / (raw_norm * expert_norm))
    if cosine is None or cosine < minimum_cosine:
        return GuardDecision(expert.copy(), "safety_expert", True, cosine)

    result = expert.copy()
    result[:3] = np.clip(
        (1.0 - model_weight) * expert[:3] + model_weight * raw[:3],
        -1.0,
        1.0,
    )
    return GuardDecision(result, "aligned_model_residual", False, cosine)


__all__ = ["GuardDecision", "select_progress_guarded_action"]
