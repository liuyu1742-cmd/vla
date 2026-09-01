"""Tests for the standalone OpenVLA RoboCasa water-cup loop."""

import unittest


class OpenVlaRoboCasaWaterCupLoopTests(unittest.TestCase):
    def test_step_record_keeps_raw_and_adapted_actions(self):
        from tools.openvla_robocasa_water_cup_loop import make_step_record
        from tools.openvla_simulator_adapter import to_robocasa_action

        record = make_step_record([0.0] * 7, to_robocasa_action([0.0] * 7), 0.0, False, False)

        self.assertEqual(record["raw_action"], [0.0] * 7)
        self.assertIn("action.end_effector_position", record["robocasa_action"])


if __name__ == "__main__":
    unittest.main()
