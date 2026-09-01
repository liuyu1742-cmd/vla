"""Regression tests for the formal pick-place phase state machine."""

from __future__ import annotations


def test_grasp_phase_holds_through_threshold_jitter() -> None:
    """Closing the gripper must not be cancelled by a 4 cm threshold bounce."""
    from tools.formal_skill_hysteresis_phase import HysteresisPhaseController

    controller = HysteresisPhaseController(
        grasp_entry_distance=0.04,
        grasp_exit_distance=0.06,
        max_grasp_decisions=20,
    )

    phase = controller(
        "locate",
        object_eef_distance=0.0395,
        grasped=False,
        inside=False,
        success=False,
    )
    assert phase == "grasp"

    phase = controller(
        phase,
        object_eef_distance=0.0535,
        grasped=False,
        inside=False,
        success=False,
    )
    assert phase == "grasp"
