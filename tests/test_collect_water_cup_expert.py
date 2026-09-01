import unittest

from tools.collect_water_cup_expert import flatten_gym_action


class CollectWaterCupExpertTests(unittest.TestCase):
    def test_flatten_gym_action_preserves_openvla_channels(self):
        action = flatten_gym_action({
            "action.end_effector_position": [0.1, 0.2, 0.3],
            "action.end_effector_rotation": [0.4, 0.5, 0.6],
            "action.gripper_close": [1.0],
        })
        self.assertEqual(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])


if __name__ == "__main__":
    unittest.main()
