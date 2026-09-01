import unittest

import numpy as np

from tools.water_cup_dagger_oracle import OracleSnapshot, WaterCupDaggerOracle


class WaterCupDaggerOracleTests(unittest.TestCase):
    def make_oracle(self) -> WaterCupDaggerOracle:
        return WaterCupDaggerOracle(
            destination_front=np.array([0.8, 0.0, 1.1]),
            destination_center=np.array([0.9, 0.0, 1.1]),
            destination_retreat=np.array([0.7, 0.0, 1.1]),
        )

    def snapshot(self, eef, obj, grasped=False) -> OracleSnapshot:
        return OracleSnapshot(
            eef_position=np.asarray(eef, dtype=float),
            object_position=np.asarray(obj, dtype=float),
            grasped=grasped,
        )

    def test_approach_keeps_gripper_open(self) -> None:
        decision = self.make_oracle().decide(
            self.snapshot([0.0, 0.0, 0.0], [0.10, 0.20, 0.0])
        )

        self.assertEqual(decision.phase, "approach_object")
        self.assertEqual(decision.action.shape, (7,))
        self.assertGreater(decision.action[2], 0.0)
        self.assertEqual(decision.action[6], 0.0)
        self.assertFalse(decision.force_expert)

    def test_close_is_entered_only_after_descent_reaches_contact(self) -> None:
        oracle = self.make_oracle()
        obj = np.array([0.10, 0.20, 0.50])

        descend = oracle.decide(self.snapshot(obj + [0.0, 0.0, 0.18], obj))
        self.assertEqual(descend.phase, "descend_to_object")
        self.assertEqual(descend.action[6], 0.0)

        close = oracle.decide(self.snapshot(obj + [0.0, 0.0, 0.002], obj))
        self.assertEqual(close.phase, "close_gripper")
        self.assertEqual(close.action[6], 1.0)
        self.assertTrue(close.force_expert)

    def test_failed_grasp_reopens_and_returns_to_approach(self) -> None:
        oracle = self.make_oracle()
        obj = np.array([0.10, 0.20, 0.50])
        contact = self.snapshot(obj + [0.0, 0.0, 0.002], obj, grasped=False)

        oracle.decide(self.snapshot(obj + [0.0, 0.0, 0.18], obj))
        for _ in range(35):
            decision = oracle.decide(contact)
            self.assertEqual(decision.action[6], 1.0)

        recovery = oracle.decide(contact)
        self.assertEqual(recovery.phase, "approach_object")
        self.assertEqual(recovery.action[6], 0.0)
        self.assertEqual(oracle.grasp_attempts, 1)


if __name__ == "__main__":
    unittest.main()
