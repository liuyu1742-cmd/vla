"""Run a bounded OpenVLA-controlled RoboCasa water-cup episode."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


TASK_NAME = "PickPlaceCounterToCabinet"
CAMERA_NAME = "robot0_agentview_left_image"


def validate_model_action(action: Sequence[float]) -> list[float]:
    """Return an OpenVLA action only when it has the expected seven values."""
    values = [float(value) for value in action]
    if len(values) != 7:
        raise ValueError(f"OpenVLA action must have 7 elements, got {len(values)}")
    return values


def build_report(seed: int, instruction: str) -> dict[str, object]:
    """Create the initial auditable report for one episode."""
    return {
        "seed": seed,
        "task": TASK_NAME,
        "instruction": instruction,
        "camera": CAMERA_NAME,
        "steps": [],
        "runner_completed": False,
        "simulator_success": None,
        "error": None,
    }
