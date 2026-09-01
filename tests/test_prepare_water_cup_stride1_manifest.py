"""Tests for the stride-1 water-cup train/held-out split."""

from __future__ import annotations

import unittest

from tools.prepare_water_cup_stride1_manifest import split_successful


class Stride1ManifestTests(unittest.TestCase):
    def test_excludes_failures_and_keeps_seed_two_held_out(self) -> None:
        episodes = [
            {"seed": 0, "success": True},
            {"seed": 2, "success": True},
            {"seed": 9, "success": False},
        ]
        train, held_out = split_successful(episodes, held_out_seed=2)
        self.assertEqual([item["seed"] for item in train], [0])
        self.assertEqual([item["seed"] for item in held_out], [2])


if __name__ == "__main__":
    unittest.main()
