"""Predicate phase policy using the measured solid-toy contact tolerance."""

from __future__ import annotations


PHASES = frozenset({"locate", "grasp", "move", "place"})
VERIFIED_CONTACT_DISTANCE = 0.028


def next_calibrated_phase(
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
    if current in {"grasp", "move"} and object_eef_distance > 0.06:
        return "locate"
    if object_eef_distance <= VERIFIED_CONTACT_DISTANCE:
        return "grasp"
    return "locate"


__all__ = ["PHASES", "VERIFIED_CONTACT_DISTANCE", "next_calibrated_phase"]
