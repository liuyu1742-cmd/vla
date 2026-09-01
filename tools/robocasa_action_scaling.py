"""Calibrated, bounded scaling from OpenVLA output to RoboCasa arm commands."""

from __future__ import annotations

from collections.abc import Sequence


def scale_openvla_action(
    action: Sequence[float], *, translation_gain: float = 1.0, rotation_gain: float = 1.0
) -> list[float]:
    """Scale XYZ and rotation separately, preserve gripper semantics, then clamp."""
    if len(action) != 7:
        raise ValueError(f"OpenVLA action must contain 7 values, got {len(action)}")
    if translation_gain <= 0 or rotation_gain <= 0:
        raise ValueError("action gains must be positive")
    scaled = [
        *(float(value) * translation_gain for value in action[:3]),
        *(float(value) * rotation_gain for value in action[3:6]),
        float(action[6]),
    ]
    return [min(1.0, max(-1.0, value)) for value in scaled]
