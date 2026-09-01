import unittest

from tools.robocasa_action_scaling import scale_openvla_action


class RoboCasaActionScalingTests(unittest.TestCase):
    def test_translation_and_rotation_are_scaled_independently(self):
        action = scale_openvla_action([0.02, -0.03, 0.04, 0.1, -0.1, 0.2, 0.8], translation_gain=5.0, rotation_gain=2.0)
        self.assertEqual(action, [0.1, -0.15, 0.2, 0.2, -0.2, 0.4, 0.8])

    def test_scaling_never_exceeds_native_action_limits(self):
        action = scale_openvla_action([0.5] * 7, translation_gain=10.0, rotation_gain=10.0)
        self.assertEqual(action, [1.0] * 7)


if __name__ == "__main__":
    unittest.main()
