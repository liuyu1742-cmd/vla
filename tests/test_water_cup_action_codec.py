"""Regression tests for the water-cup OpenVLA action decoding contract."""

from __future__ import annotations

import unittest

from tools.water_cup_action_codec import install_water_cup_action_stats


class _Model:
    norm_stats: dict


class WaterCupActionCodecTests(unittest.TestCase):
    def test_installed_stats_preserve_normalized_robot_actions(self) -> None:
        model = _Model()
        install_water_cup_action_stats(model)

        stats = model.norm_stats["water_cup_robocasa"]["action"]
        self.assertEqual(stats["q01"], [-1.0] * 7)
        self.assertEqual(stats["q99"], [1.0] * 7)
        self.assertEqual(stats["mask"], [True] * 7)


if __name__ == "__main__":
    unittest.main()
