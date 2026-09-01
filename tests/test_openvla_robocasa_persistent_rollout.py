import unittest

from tools.openvla_robocasa_persistent_rollout import rollout_instruction


class PersistentRolloutTests(unittest.TestCase):
    def test_pick_phase_uses_pick_instruction(self):
        self.assertEqual(rollout_instruction("pick"), "pick up the glass cup")


if __name__ == "__main__":
    unittest.main()
