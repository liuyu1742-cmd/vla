"""Regression for top-grasping a solid primitive instead of an open cup."""

from __future__ import annotations

import unittest

import numpy as np

from tools.pick_place_oracle import PickPlaceOracle, PickPlaceSnapshot


class SolidObjectPickPlaceOracleTests(unittest.TestCase):
    def test_reachable_solid_object_height_enters_close_gripper(self) -> None:
        oracle = PickPlaceOracle(
            np.array([0.8, 0.0, 1.1]),
            np.array([0.9, 0.0, 1.1]),
            np.array([0.7, 0.0, 1.1]),
        )
        obj = np.array([0.1, 0.2, 0.5])
        oracle.decide(PickPlaceSnapshot(obj + [0, 0, 0.18], obj, False))

        decision = oracle.decide(
            PickPlaceSnapshot(obj + [0, 0, 0.025], obj, False)
        )

        self.assertEqual("close_gripper", decision.phase)
        self.assertEqual("grasp", decision.canonical_action)
        self.assertEqual(1.0, decision.action[6])


if __name__ == "__main__":
    unittest.main()
