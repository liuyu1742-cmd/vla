import unittest

from tools.robocasa_raw_executor import raw_robocasa_action


class RawRoboCasaExecutorTests(unittest.TestCase):
    def test_openvla_action_becomes_bounded_native_vector(self):
        action = raw_robocasa_action([0.1, -0.2, 2.0, 0.3, 0.4, -0.5, -3.0])
        self.assertEqual(action, [0.1, -0.2, 1.0, 0.3, 0.4, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
