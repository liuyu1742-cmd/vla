"""Tests for the bounded OpenVLA RoboCasa water-cup runner."""

from __future__ import annotations

import unittest


class OpenVlaRoboCasaWaterCupTests(unittest.TestCase):
    def test_rejects_non_seven_dimensional_action(self) -> None:
        from tools.run_openvla_robocasa_water_cup import validate_model_action

        with self.assertRaisesRegex(ValueError, "7"):
            validate_model_action([0.0] * 6)

    def test_report_separates_runner_and_simulator_status(self) -> None:
        from tools.run_openvla_robocasa_water_cup import build_report

        report = build_report(0, "pick up the glass cup and place it in the cabinet")

        self.assertFalse(report["runner_completed"])
        self.assertIsNone(report["simulator_success"])


if __name__ == "__main__":
    unittest.main()
