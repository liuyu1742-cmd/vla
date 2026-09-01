"""Helpers for an auditable OpenVLA-to-RoboCasa water-cup control loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def make_step_record(
    raw_action: Sequence[float],
    robocasa_action: Mapping[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, object]:
    """Return a JSON-safe record for one real simulator transition."""
    return {
        "raw_action": [float(value) for value in raw_action],
        "robocasa_action": {
            key: value.tolist() if hasattr(value, "tolist") else value for key, value in robocasa_action.items()
        },
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
