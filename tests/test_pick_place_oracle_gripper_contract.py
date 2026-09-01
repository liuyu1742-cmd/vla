"""Regression for explicit Panda gripper commands in expert demonstrations."""

from __future__ import annotations

import unittest

import numpy as np

from tools.pick_place_oracle.open_gripper import OpenGripperPickPlaceOracle
from tools.pick_place_oracle import PickPlaceSnapshot


class PickPlaceOracleGripperContractTests(unittest.TestCase):
    def test_motion_uses_negative_one_for_fully_open_gripper(self) -> None:
        oracle = OpenGripperPickPlaceOracle(
            np.array([0.8, 0.0, 1.1]),
            np.array([0.9, 0.0, 1.1]),
            np.array([0.7, 0.0, 1.1]),
        )
        decision = oracle.decide(
            PickPlaceSnapshot(
                eef_position=np.zeros(3),
                object_position=np.array([0.1, 0.2, 0.5]),
                grasped=False,
            )
        )
        self.assertEqual(-1.0, decision.action[6])

    def test_closed_gripper_remains_positive_one(self) -> None:
        oracle = OpenGripperPickPlaceOracle(
            np.array([0.8, 0.0, 1.1]),
            np.array([0.9, 0.0, 1.1]),
            np.array([0.7, 0.0, 1.1]),
        )
        obj = np.array([0.1, 0.2, 0.5])
        oracle.decide(PickPlaceSnapshot(obj + [0, 0, 0.18], obj, False))
        decision = oracle.decide(PickPlaceSnapshot(obj + [0, 0, 0.025], obj, False))
        self.assertEqual(1.0, decision.action[6])


if __name__ == "__main__":
    unittest.main()
