import unittest

from tools.openvla_simulator_adapter import to_robocasa_native_action


class RoboCasaNativeActionTests(unittest.TestCase):
    def test_openvla_action_is_padded_to_raw_robosuite_dimension(self):
        action = to_robocasa_native_action([0.1, -0.2, 2.0, 0.3, 0.4, -0.5, -3.0])
        self.assertEqual(action, [0.1, -0.2, 1.0, 0.3, 0.4, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
