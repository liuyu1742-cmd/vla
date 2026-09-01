import unittest

from tools.openvla_gym_lora_rollout import action_for_gym


class GymLoRARolloutTests(unittest.TestCase):
    def test_action_maps_seven_openvla_channels_to_gym_groups(self):
        action = action_for_gym([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
        self.assertEqual(action["action.end_effector_position"].tolist(), [0.1, 0.2, 0.3])
        self.assertEqual(action["action.gripper_close"].tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
