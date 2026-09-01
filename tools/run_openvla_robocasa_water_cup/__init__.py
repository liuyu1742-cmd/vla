"""Compatibility API for the bounded RoboCasa water-cup smoke runner.

The legacy water-cup experiment remains useful for IPC and simulator checks, but
it is not one of the 138 Task-2 relation contracts and therefore cannot contribute
formal skill-coverage evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from tools.skill_transfer.evidence import infrastructure_smoke_metadata


TASK_NAME = "PickPlaceCounterToCabinet"
CAMERA_NAME = "robot0_agentview_left_image"


def validate_model_action(action: Sequence[float]) -> list[float]:
    """Return an OpenVLA action only when it has the expected seven values."""
    values = [float(value) for value in action]
    if len(values) != 7:
        raise ValueError(f"OpenVLA action must have 7 elements, got {len(values)}")
    return values


def build_report(seed: int, instruction: str) -> dict[str, object]:
    """Create an auditable infrastructure-only report for one episode."""
    return {
        **infrastructure_smoke_metadata(),
        "seed": seed,
        "task": TASK_NAME,
        "instruction": instruction,
        "camera": CAMERA_NAME,
        "steps": [],
        "runner_completed": False,
        "simulator_success": None,
        "error": None,
    }


__all__ = ["CAMERA_NAME", "TASK_NAME", "build_report", "validate_model_action"]
