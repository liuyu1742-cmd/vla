"""Pick-place phase policy with calibrated contact-entry hysteresis."""

from __future__ import annotations


PHASES = frozenset({"locate", "grasp", "move", "place"})
CONTACT_ENTRY_DISTANCE = 0.028
CONTACT_RECOVERY_DISTANCE = 0.060


class HysteresisPhaseController:
    """Hold a confirmed contact attempt through bounded distance jitter."""

    def __init__(
        self,
        *,
        grasp_entry_distance: float = 0.040,
        grasp_exit_distance: float = CONTACT_RECOVERY_DISTANCE,
        max_grasp_decisions: int = 20,
    ) -> None:
        if not 0.0 < grasp_entry_distance < grasp_exit_distance:
            raise ValueError("grasp thresholds must satisfy 0 < entry < exit")
        if max_grasp_decisions < 1:
            raise ValueError("max_grasp_decisions must be positive")
        self.grasp_entry_distance = float(grasp_entry_distance)
        self.grasp_exit_distance = float(grasp_exit_distance)
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
                object_eef_distance > self.grasp_exit_distance
                or self._ungrasped_decisions >= self.max_grasp_decisions
            ):
                self._ungrasped_decisions = 0
                return "locate"
            return "grasp"
        self._ungrasped_decisions = 0
        if object_eef_distance <= self.grasp_entry_distance:
            return "grasp"
        return "locate"


def next_hysteresis_phase(
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
        return "done"
    if inside:
        return "place"
    if grasped:
        return "move"
    if current == "grasp":
        return (
            "grasp"
            if object_eef_distance <= CONTACT_RECOVERY_DISTANCE
            else "locate"
        )
    if current == "move":
        return (
            "grasp"
            if object_eef_distance <= CONTACT_RECOVERY_DISTANCE
            else "locate"
        )
    if object_eef_distance <= CONTACT_ENTRY_DISTANCE:
        return "grasp"
    return "locate"


__all__ = [
    "CONTACT_ENTRY_DISTANCE",
    "CONTACT_RECOVERY_DISTANCE",
    "HysteresisPhaseController",
    "PHASES",
    "next_hysteresis_phase",
]
