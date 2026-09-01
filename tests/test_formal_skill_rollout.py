from __future__ import annotations

import unittest

import numpy as np

from tools.formal_skill_rollout import guarded_action, next_canonical_phase


class FormalSkillRolloutTests(unittest.TestCase):
    def test_phase_progression_uses_robot_and_task_predicates(self) -> None:
        self.assertEqual(
            next_canonical_phase(
                "locate", object_eef_distance=0.20, grasped=False, inside=False, success=False
            ),
            "locate",
        )
        self.assertEqual(
            next_canonical_phase(
                "locate", object_eef_distance=0.03, grasped=False, inside=False, success=False
            ),
            "grasp",
        )
        self.assertEqual(
            next_canonical_phase(
                "grasp", object_eef_distance=0.02, grasped=True, inside=False, success=False
            ),
            "move",
        )
        self.assertEqual(
            next_canonical_phase(
                "move", object_eef_distance=0.10, grasped=True, inside=True, success=False
            ),
            "place",
        )
        self.assertEqual(
            next_canonical_phase(
                "place", object_eef_distance=0.10, grasped=False, inside=True, success=True
            ),
            "done",
        )

    def test_grasp_loss_recovers_to_locate(self) -> None:
        self.assertEqual(
            next_canonical_phase(
                "move", object_eef_distance=0.12, grasped=False, inside=False, success=False
            ),
            "locate",
        )

    def test_guard_forces_safe_gripper_and_zero_untrained_rotation(self) -> None:
        raw = [2.0, -2.0, 0.8, 0.9, -0.9, 0.5, -0.2]
        locate = guarded_action(raw, "locate")
        move = guarded_action(raw, "move")
        place = guarded_action(raw, "place")

        np.testing.assert_allclose(locate, [1.0, -1.0, 0.8, 0, 0, 0, -1.0])
        np.testing.assert_allclose(move, [0.3, -0.3, 0.3, 0, 0, 0, 1.0])
        np.testing.assert_allclose(place, [0.5, -0.5, 0.5, 0, 0, 0, -1.0])

    def test_guard_rejects_invalid_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "seven"):
            guarded_action([0.0], "locate")


if __name__ == "__main__":
    unittest.main()
