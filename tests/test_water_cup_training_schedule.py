"""Tests for epoch-based water-cup OpenVLA training."""

from __future__ import annotations

import unittest

from tools.water_cup_training_schedule import training_updates


class WaterCupTrainingScheduleTests(unittest.TestCase):
    def test_uses_every_sample_in_each_epoch(self) -> None:
        self.assertEqual(training_updates(samples=552, batch_size=1, epochs=12), 6624)

    def test_rounds_up_partial_batch(self) -> None:
        self.assertEqual(training_updates(samples=5, batch_size=2, epochs=3), 9)


if __name__ == "__main__":
    unittest.main()
