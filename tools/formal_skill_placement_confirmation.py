"""Delay release until storage containment remains stable during closed transport."""

from __future__ import annotations

from tools.formal_skill_verified_phase import VerifiedPhaseController


class PlacementConfirmedPhaseController:
    def __init__(
        self,
        *,
        inside_confirm_decisions: int = 20,
        max_grasp_decisions: int = 20,
    ) -> None:
        if inside_confirm_decisions < 1:
            raise ValueError("inside_confirm_decisions must be positive")
        self.inside_confirm_decisions = int(inside_confirm_decisions)
        self._inside_count = 0
        self._base = VerifiedPhaseController(
            max_grasp_decisions=max_grasp_decisions
        )

    def __call__(
        self,
        current: str,
        *,
        object_eef_distance: float,
        grasped: bool,
        inside: bool,
        success: bool,
    ) -> str:
        if success:
            self._inside_count = 0
            return "done"
        if current == "move" and inside:
            self._inside_count += 1
            return (
                "place"
                if self._inside_count >= self.inside_confirm_decisions
                else "move"
            )
        if current == "place" and inside:
            self._inside_count = 0
            return "place"
        if inside:
            self._inside_count = 0
            return "place"
        self._inside_count = 0
        return self._base(
            current,
            object_eef_distance=object_eef_distance,
            grasped=grasped,
            inside=False,
            success=False,
        )


__all__ = ["PlacementConfirmedPhaseController"]
