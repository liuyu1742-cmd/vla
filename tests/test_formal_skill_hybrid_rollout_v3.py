import unittest

from tools import formal_skill_hybrid_rollout_v3
from tools.formal_skill_final_transport import activate_final_transport


class FormalSkillHybridRolloutV3Tests(unittest.TestCase):
    def test_wrapper_installs_locate_recovery_and_final_transport(self):
        self.assertEqual(formal_skill_hybrid_rollout_v3.PATCH_COUNT, 5)
        self.assertIs(
            formal_skill_hybrid_rollout_v3._IMPLEMENTATION.activate_final_placement,
            activate_final_transport,
        )
        self.assertIn(
            "locate_recovery_started_at",
            formal_skill_hybrid_rollout_v3.PATCHED_SOURCE,
        )
        self.assertIn(
            "state_aware_locate_recovery",
            formal_skill_hybrid_rollout_v3.PATCHED_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
