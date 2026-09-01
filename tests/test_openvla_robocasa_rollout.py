import unittest

from tools.openvla_robocasa_rollout import build_request, choose_phase


class OpenVLARoboCasaRolloutTests(unittest.TestCase):
    def test_request_uses_phase_specific_instruction(self):
        request = build_request("C:/frame.png", "pick")
        self.assertEqual(request["image_path"], "C:/frame.png")
        self.assertEqual(request["instruction"], "pick up the glass cup")

    def test_phase_switch_requires_verified_grasp(self):
        self.assertEqual(choose_phase("pick", {"grasped": False}), "pick")
        self.assertEqual(choose_phase("pick", {"grasped": True}), "place")


if __name__ == "__main__":
    unittest.main()
