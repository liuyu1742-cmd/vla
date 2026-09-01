import unittest

import numpy as np

from tools.water_cup_dagger_oracle import OracleDecision
from tools.water_cup_guarded_control import supervise_action


class WaterCupGuardedControlTests(unittest.TestCase):
    def decision(self, action, phase="approach_object", forced=False):
        return OracleDecision(np.asarray(action, dtype=np.float32), phase, forced)

    def test_matching_policy_action_is_preserved(self) -> None:
        oracle = self.decision([0.4, 0.2, -0.3, 0, 0, 0, 0])
        policy = np.array([0.35, 0.15, -0.25, 0, 0, 0, 0], dtype=np.float32)
        result = supervise_action(policy, oracle, max_translation_error=0.15)
        np.testing.assert_allclose(result.action, policy)
        self.assertEqual(result.source, "openvla")

    def test_premature_close_uses_recovery_action(self) -> None:
        oracle = self.decision([0.2, 0.9, -1.0, 0, 0, 0, 0])
        policy = np.array([0.05, -0.2, 0.2, 0, 0, 0, 1], dtype=np.float32)
        result = supervise_action(policy, oracle, max_translation_error=0.2)
        np.testing.assert_allclose(result.action, oracle.action)
        self.assertEqual(result.source, "supervisor_gripper")

    def test_large_translation_error_uses_recovery_action(self) -> None:
        oracle = self.decision([0.8, 0.8, -0.8, 0, 0, 0, 0])
        policy = np.array([-0.8, 0.8, -0.8, 0, 0, 0, 0], dtype=np.float32)
        result = supervise_action(policy, oracle, max_translation_error=0.2)
        np.testing.assert_allclose(result.action, oracle.action)
        self.assertEqual(result.source, "supervisor_translation")

    def test_forced_oracle_phase_always_wins(self) -> None:
        oracle = self.decision([0, 0, 0, 0, 0, 0, 1], "close_gripper", True)
        result = supervise_action(np.zeros(7), oracle, max_translation_error=1.0)
        np.testing.assert_allclose(result.action, oracle.action)
        self.assertEqual(result.source, "supervisor_forced")


if __name__ == "__main__":
    unittest.main()
