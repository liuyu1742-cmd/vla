import unittest

import numpy as np

from tools.formal_skill_locate_recovery import LocateStagnationRecovery


class FormalSkillLocateRecoveryActionTests(unittest.TestCase):
    def test_active_recovery_rejects_zero_model_translation(self):
        recovery = LocateStagnationRecovery(
            window_decisions=2, minimum_progress=0.01
        )
        for distance in (0.210, 0.209, 0.208):
            recovery.observe("locate", distance)

        decision = recovery.select_action(
            raw_action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            expert_action=[-0.31, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            force_expert=False,
        )

        self.assertEqual(decision.mode, "safety_expert")
        self.assertTrue(decision.intervened)
        np.testing.assert_allclose(
            decision.action,
            [-0.31, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        )


if __name__ == "__main__":
    unittest.main()
