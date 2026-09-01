import unittest

import numpy as np

from tools.formal_skill_execution_guard import PickPlaceExecutionGuard


class PickPlaceExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = PickPlaceExecutionGuard(
            grasp_hold_decisions=3,
            release_hold_decisions=2,
        )
        self.raw = [0.4, -0.5, 0.3, 0.0, 0.0, 0.0, -1.0]

    def test_grasp_hold_continues_after_predicate_switches_to_move(self):
        actions = [
            self.guard.apply(self.raw, "grasp"),
            self.guard.apply(self.raw, "move"),
            self.guard.apply(self.raw, "move"),
        ]

        for action in actions:
            np.testing.assert_allclose(action, [0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(
            self.guard.apply(self.raw, "move"),
            [0.3, -0.3, 0.3, 0, 0, 0, 1],
            atol=1e-6,
        )

    def test_failed_grasp_returning_to_locate_cancels_hold(self):
        self.guard.apply(self.raw, "grasp")
        locate = self.guard.apply(self.raw, "locate")

        np.testing.assert_allclose(
            locate, [0.4, -0.5, 0.3, 0, 0, 0, -1], atol=1e-6
        )

    def test_reentering_grasp_restarts_full_hold(self):
        self.guard.apply(self.raw, "grasp")
        self.guard.apply(self.raw, "locate")

        np.testing.assert_allclose(
            self.guard.apply(self.raw, "grasp"), [0, 0, 0, 0, 0, 0, 1]
        )

    def test_place_opens_without_translation_before_retreat(self):
        first = self.guard.apply(self.raw, "place")
        second = self.guard.apply(self.raw, "place")
        retreat = self.guard.apply(self.raw, "place")

        np.testing.assert_allclose(first, [0, 0, 0, 0, 0, 0, -1])
        np.testing.assert_allclose(second, [0, 0, 0, 0, 0, 0, -1])
        np.testing.assert_allclose(
            retreat, [0.4, -0.5, 0.3, 0, 0, 0, -1], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
