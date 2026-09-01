"""Validate and adapt OpenVLA continuous actions for a MuJoCo action space."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def adapt_action(
    openvla_action: Sequence[float], action_low: Sequence[float], action_high: Sequence[float]
) -> list[float]:
    """Return a bounded action only when model and simulator dimensions agree."""
    if len(openvla_action) != len(action_low) or len(action_low) != len(action_high):
        raise ValueError(
            "action dimension mismatch: "
            f"model={len(openvla_action)}, low={len(action_low)}, high={len(action_high)}"
        )
    bounded = []
    for value, lower, upper in zip(openvla_action, action_low, action_high):
        if lower > upper:
            raise ValueError(f"invalid action bounds: {lower} > {upper}")
        bounded.append(min(max(float(value), float(lower)), float(upper)))
    return bounded


def to_robocasa_action(openvla_action: Sequence[float]) -> dict[str, np.ndarray]:
    """Map OpenVLA's 7 DoF output to RoboCasa's stationary-base Gym action dict.

    The final two RoboCasa groups are intentionally fixed: a zero base motion
    prevents unvalidated mobile-base control, and the default control-mode value
    follows RoboCasa's own zero-action convention.
    """
    bounded = adapt_action(openvla_action, [-1.0] * 7, [1.0] * 7)
    return {
        "action.end_effector_position": np.asarray(bounded[0:3], dtype=np.float32),
        "action.end_effector_rotation": np.asarray(bounded[3:6], dtype=np.float32),
        "action.gripper_close": np.asarray(bounded[6:7], dtype=np.float32),
        "action.base_motion": np.zeros(4, dtype=np.float32),
        "action.control_mode": np.zeros(1, dtype=np.float32),
    }
