import unittest


class RoboCasaCalibrationTests(unittest.TestCase):
    def test_appends_fixed_base_and_control_dimensions(self):
        from tools.calibrate_robocasa_action_space import to_native_action

        action = to_native_action([0.1] * 7)

        self.assertEqual(len(action), 12)
        self.assertEqual(action[7:], [0.0] * 5)
