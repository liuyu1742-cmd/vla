import unittest


class RotationGripperCalibrationTests(unittest.TestCase):
    def test_rotation_and_gripper_actions_are_native_12d(self):
        from tools.calibrate_robocasa_rotation_gripper import native_probe

        self.assertEqual(len(native_probe(3, 0.1)), 12)
        self.assertEqual(native_probe(6, -1.0)[6], -1.0)
