"""Regression for completing closed-gripper transit within a bounded episode."""

from __future__ import annotations

import unittest

import numpy as np

from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.fast_transit import FastTransitPickPlaceOracle


class PickPlaceOracleTransitSpeedTests(unittest.TestCase):
    def test_closed_transit_allows_half_scale_translation(self) -> None:
        oracle = FastTransitPickPlaceOracle(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.1, 0.0, 0.0]),
            np.array([0.9, 0.0, 0.0]),
        )
        oracle.phase = "approach_destination"
        snapshot = PickPlaceSnapshot(np.zeros(3), np.zeros(3), True)

        decision = oracle._motion(snapshot, np.ones(3), closed=True)

        self.assertEqual([0.5, 0.5, 0.5], decision.action[:3].tolist())
        self.assertEqual(1.0, decision.action[6])


if __name__ == "__main__":
    unittest.main()
