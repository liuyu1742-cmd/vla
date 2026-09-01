import unittest

import numpy as np

from tools.formal_skill_release_guard import ReleaseHoldGuard


class FormalSkillReleaseGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = ReleaseHoldGuard(hold_decisions=3)
        self.raw = [0.4, -0.5, 0.3, 0.0, 0.0, 0.0, 1.0]

    def test_place_holds_position_open_for_configured_decisions(self):
        held = [self.guard.apply(self.raw, "place") for _ in range(3)]

        for action in held:
            np.testing.assert_allclose(action, [0, 0, 0, 0, 0, 0, -1])
        released = self.guard.apply(self.raw, "place")
        np.testing.assert_allclose(
            released, [0.4, -0.5, 0.3, 0, 0, 0, -1], atol=1e-6
        )

    def test_leaving_place_resets_release_hold(self):
        self.guard.apply(self.raw, "place")
        self.guard.apply(self.raw, "locate")

        np.testing.assert_allclose(
            self.guard.apply(self.raw, "place"), [0, 0, 0, 0, 0, 0, -1]
        )

    def test_other_phases_keep_existing_guard_contract(self):
        action = self.guard.apply(self.raw, "move")

        np.testing.assert_allclose(
            action, [0.3, -0.3, 0.3, 0, 0, 0, 1], atol=1e-6
        )

    def test_invalid_hold_count_is_rejected(self):
        with self.assertRaises(ValueError):
            ReleaseHoldGuard(hold_decisions=0)


if __name__ == "__main__":
    unittest.main()
