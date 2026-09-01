"""Regression for matching the formal toy geometry to the Panda gripper."""

from __future__ import annotations

import math
import unittest

from tools.skill_transfer.robocasa_assets import (
    ACTIVE_TOY_MODEL_XML,
    inspect_toy_asset,
)


class Task2ToyGraspGeometryTests(unittest.TestCase):
    def test_active_toy_random_yaw_fits_eight_centimeter_opening(self) -> None:
        asset = inspect_toy_asset(ACTIVE_TOY_MODEL_XML)
        size = asset["full_size"]
        worst_planar_projection = math.hypot(size[0], size[1])
        self.assertLessEqual(worst_planar_projection, 0.08)
        self.assertEqual([0.05, 0.05, 0.05], size)


if __name__ == "__main__":
    unittest.main()
