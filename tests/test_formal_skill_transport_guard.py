import unittest

import numpy as np

from tools.formal_skill_transport_guard import CalibratedPickPlaceGuard


class FormalSkillTransportGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = CalibratedPickPlaceGuard(
            grasp_hold_decisions=2,
            lift_decisions=3,
            release_hold_decisions=2,
        )
        self.raw = [0.4, -0.5, 0.1, 0.0, 0.0, 0.0, -1.0]

    def test_close_hold_is_followed_by_vertical_lift_before_model_move(self):
        sequence = [
            self.guard.apply(self.raw, "grasp"),
            self.guard.apply(self.raw, "move"),
            self.guard.apply(self.raw, "move"),
            self.guard.apply(self.raw, "move"),
            self.guard.apply(self.raw, "move"),
        ]

        np.testing.assert_allclose(sequence[0], [0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(sequence[1], [0, 0, 0, 0, 0, 0, 1])
        for action in sequence[2:]:
            np.testing.assert_allclose(action, [0, 0, 0.3, 0, 0, 0, 1])
        np.testing.assert_allclose(
            self.guard.apply(self.raw, "move"),
            [0.3, -0.3, 0.1, 0, 0, 0, 1],
            atol=1e-6,
        )

    def test_grasp_loss_rearms_close_and_lift_sequence(self):
        self.guard.apply(self.raw, "grasp")
        self.guard.apply(self.raw, "move")
        self.guard.apply(self.raw, "move")

        np.testing.assert_allclose(
            self.guard.apply(self.raw, "grasp"), [0, 0, 0, 0, 0, 0, 1]
        )

    def test_locate_cancels_pending_transport_sequence(self):
        self.guard.apply(self.raw, "grasp")
        locate = self.guard.apply(self.raw, "locate")

        np.testing.assert_allclose(
            locate, [0.4, -0.5, 0.1, 0, 0, 0, -1], atol=1e-6
        )

    def test_release_remains_stationary_before_retreat(self):
        np.testing.assert_allclose(
            self.guard.apply(self.raw, "place"), [0, 0, 0, 0, 0, 0, -1]
        )
        np.testing.assert_allclose(
            self.guard.apply(self.raw, "place"), [0, 0, 0, 0, 0, 0, -1]
        )
        np.testing.assert_allclose(
            self.guard.apply(self.raw, "place"),
            [0.4, -0.5, 0.1, 0, 0, 0, -1],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
