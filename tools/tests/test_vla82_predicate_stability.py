"""Placement windows must include delayed stable support after release bounce."""

from __future__ import annotations

import numpy as np
import unittest

from tools.vla82_full_sim.environment import ContactEvidence, PhysicsSnapshot, TargetGeometry
from tools.vla82_full_sim import predicates


def _snapshot(
    step: int, position: tuple[float, float, float], *,
    target_contact: bool, gripper_contact: bool = False,
    target_relation: str = "inside", target_id: str = "drawer",
) -> PhysicsSnapshot:
    target = TargetGeometry(
        target_id=target_id,
        fixture_id=f"{target_id}_fixture",
        min_corner=(0.0, 0.0, 0.0),
        max_corner=(1.0, 1.0, 1.0),
        support_id=target_id,
        spatial_relation=target_relation,
    )
    evidence = tuple(
        item for item in (
            ContactEvidence("obj_geom", f"{target_id}_floor", "obj", target_id, f"{target_id}_fixture", None, -0.001)
            if target_contact else None,
            ContactEvidence("gripper0_right_finger1_pad_collision", "obj_geom", "obj", None, None, None, -0.001)
            if gripper_contact else None,
        ) if item is not None
    )
    return PhysicsSnapshot(
        step=step,
        robot_qpos=np.zeros(7),
        gripper_qpos=np.zeros(2),
        body_poses={"obj": np.asarray((*position, 1.0, 0.0, 0.0, 0.0))},
        joint_positions={},
        contacts=(),
        dirt_fraction=1.0,
        spray_coverage=0.0,
        dispensed_amount=0.0,
        contact_evidence=evidence,
        target_geometries={target_id: target},
    )


class PlacementStabilityWindowTest(unittest.TestCase):
    def test_stack_accepts_authoritative_on_relation_with_exact_stable_support(self) -> None:
        states = (
            _snapshot(0, (0.5, 0.5, 1.1), target_contact=False, gripper_contact=True,
                      target_relation="on", target_id="folder"),
            _snapshot(1, (0.5, 0.5, 1.0), target_contact=True, gripper_contact=True,
                      target_relation="on", target_id="folder"),
            _snapshot(2, (0.5, 0.5, 1.0), target_contact=True,
                      target_relation="on", target_id="folder"),
            _snapshot(3, (0.5, 0.5, 1.0), target_contact=True,
                      target_relation="on", target_id="folder"),
        )
        window = predicates.PhaseWindow("stack", 0, 3, states)
        ok, metrics, errors = predicates._strict_phase_object_result(
            "stack", "obj", window, states[-1].target_geometries["folder"],
        )
        self.assertTrue(ok, errors)
        self.assertTrue(metrics["stable_support"])

    def test_spray_bottle_requires_an_upright_final_pose(self) -> None:
        required = getattr(predicates, "requires_upright_final_pose", None)
        self.assertIsNotNone(required)
        self.assertTrue(required("VLA82-004"))
        self.assertTrue(required("VLA82-017"))
        self.assertFalse(required("VLA82-019"))

    def test_strict_place_rejects_stable_target_contact_after_a_long_drop(self) -> None:
        dropped = (
            _snapshot(0, (0.5, 0.5, 1.5), target_contact=False, gripper_contact=True),
            *(
                _snapshot(step, (0.5, 0.5, 1.5 - .05 * step), target_contact=False)
                for step in range(1, 9)
            ),
            _snapshot(9, (0.5, 0.5, 0.8), target_contact=True),
            _snapshot(10, (0.5, 0.5, 0.8), target_contact=True),
        )
        window = predicates.PhaseWindow("place", 0, 10, dropped)
        ok, _metrics, errors = predicates._strict_phase_object_result(
            "place", "obj", window, dropped[-1].target_geometries["drawer"],
        )
        self.assertFalse(ok)
        self.assertIn("controlled_release_missing:drawer", errors)

    def test_release_after_long_uncontrolled_drop_is_rejected(self) -> None:
        controlled = getattr(predicates, "controlled_release_is_recently_held", None)
        self.assertIsNotNone(controlled)
        dropped = (
            _snapshot(0, (0.5, 0.5, 1.5), target_contact=False, gripper_contact=True),
            *(
                _snapshot(step, (0.5, 0.5, 1.5 - .05 * step), target_contact=False)
                for step in range(1, 9)
            ),
            _snapshot(9, (0.5, 0.5, 0.8), target_contact=True),
            _snapshot(10, (0.5, 0.5, 0.8), target_contact=True),
        )
        self.assertFalse(controlled(dropped, release_index=9, object_id="obj", max_gap=6))

    def test_release_immediately_after_gripper_opens_is_controlled(self) -> None:
        controlled = getattr(predicates, "controlled_release_is_recently_held", None)
        self.assertIsNotNone(controlled)
        placed = (
            _snapshot(0, (0.5, 0.5, 1.1), target_contact=False),
            _snapshot(1, (0.5, 0.5, 1.02), target_contact=False, gripper_contact=True),
            _snapshot(2, (0.5, 0.5, 0.8), target_contact=True),
            _snapshot(3, (0.5, 0.5, 0.8), target_contact=True),
        )
        self.assertTrue(controlled(placed, release_index=2, object_id="obj", max_gap=6))

    def test_target_guided_release_remains_controlled_while_object_settles_in_basin(self) -> None:
        controlled = getattr(predicates, "controlled_release_from_target_contact", None)
        self.assertIsNotNone(controlled)
        settling = (
            _snapshot(0, (0.5, 0.5, 1.1), target_contact=False),
            _snapshot(1, (0.5, 0.5, 1.0), target_contact=True, gripper_contact=True),
            _snapshot(2, (0.5, 0.5, 0.95), target_contact=True),
            *(
                _snapshot(step, (0.5, 0.5, 0.9), target_contact=True)
                for step in range(3, 11)
            ),
        )
        self.assertTrue(controlled(
            settling,
            release_index=9,
            object_id="obj",
            target_id="drawer",
        ))

    def test_place_window_waits_for_delayed_stable_in_target_pair(self) -> None:
        timeline = (
            _snapshot(0, (0.5, 0.5, 1.2), target_contact=False),
            _snapshot(1, (0.5, 0.5, 1.05), target_contact=True),
            _snapshot(2, (0.5, 0.5, 1.04), target_contact=False),
            _snapshot(3, (0.5, 0.5, 0.8), target_contact=True),
            _snapshot(4, (0.5, 0.5, 0.8), target_contact=True),
        )
        finder = getattr(predicates, "stable_release_end_index", None)
        self.assertIsNotNone(finder)
        self.assertEqual(
            finder(
                timeline,
                start_index=1,
                object_id="obj",
                target=timeline[-1].target_geometries["drawer"],
                phase="place",
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
