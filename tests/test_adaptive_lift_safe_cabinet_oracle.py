import unittest

import numpy as np

from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.adaptive_lift_safe_cabinet import (
    AdaptiveLiftSafeCabinetPickPlaceOracle,
)


class AdaptiveLiftSafeCabinetOracleTests(unittest.TestCase):
    def test_grasped_object_uses_reachable_counter_clearance(self) -> None:
        oracle = AdaptiveLiftSafeCabinetPickPlaceOracle(
            np.asarray([0.4, 0.0, 1.1]),
            np.asarray([0.5, 0.0, 1.1]),
            np.asarray([0.3, 0.0, 1.1]),
        )
        oracle.phase = "close_gripper"
        oracle.phase_steps = oracle.CLOSE_HOLD_STEPS
        eef = np.asarray([0.0, 0.0, 0.97])

        decision = oracle.decide(
            PickPlaceSnapshot(eef, np.asarray([0.0, 0.0, 0.95]), True)
        )

        self.assertEqual(decision.phase, "lift_object")
        self.assertAlmostEqual(oracle.lift_target[2] - eef[2], 0.14, places=6)
        self.assertEqual(float(decision.action[2]), oracle.CLOSED_TRANSLATION_LIMIT)


if __name__ == "__main__":
    unittest.main()
