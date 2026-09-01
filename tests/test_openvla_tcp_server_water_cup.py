"""Tests for the water-cup-specific OpenVLA inference server."""

from __future__ import annotations

import unittest

from tools.openvla_tcp_server_water_cup import configure_water_cup_model


class _Model:
    norm_stats: dict


class WaterCupServerTests(unittest.TestCase):
    def test_configure_model_returns_its_dataset_key(self) -> None:
        model = _Model()
        self.assertEqual(configure_water_cup_model(model), "water_cup_robocasa")
        self.assertIn("water_cup_robocasa", model.norm_stats)


if __name__ == "__main__":
    unittest.main()
