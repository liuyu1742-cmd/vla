"""Stateful phase controller calibrated from successful expert trajectories."""

from __future__ import annotations


PHASES = frozenset({"locate", "grasp", "move", "place"})
VERIFIED_GRASP_ENTRY_DISTANCE = 0.024
GRASP_RECOVERY_DISTANCE = 0.060


class VerifiedPhaseController:
    def __init__(self, *, max_grasp_decisions: int = 20) -> None:
        if max_grasp_decisions < 1:
            raise ValueError("max_grasp_decisions must be positive")
        self.max_grasp_decisions = int(max_grasp_decisions)
        self._ungrasped_decisions = 0

    def __call__(
        self,
        current: str,
        *,
        object_eef_distance: float,
        grasped: bool,
        inside: bool,
        success: bool,
    ) -> str:
        if current not in PHASES:
            raise ValueError(f"unsupported current canonical phase: {current!r}")
        if success:
            self._ungrasped_decisions = 0
            return "done"
        if inside:
            self._ungrasped_decisions = 0
            return "place"
        if grasped:
            self._ungrasped_decisions = 0
            return "move"
        if current == "grasp":
            self._ungrasped_decisions += 1
            if (
                object_eef_distance > GRASP_RECOVERY_DISTANCE
                or self._ungrasped_decisions >= self.max_grasp_decisions
            ):
                self._ungrasped_decisions = 0
                return "locate"
            return "grasp"
        self._ungrasped_decisions = 0
        if current == "move":
            return "locate"
        if current == "locate" and (
            object_eef_distance <= VERIFIED_GRASP_ENTRY_DISTANCE
        ):
            return "grasp"
        return "locate"


__all__ = [
    "GRASP_RECOVERY_DISTANCE",
    "PHASES",
    "VERIFIED_GRASP_ENTRY_DISTANCE",
    "VerifiedPhaseController",
]
