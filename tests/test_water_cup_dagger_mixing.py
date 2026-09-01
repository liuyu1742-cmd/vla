import unittest

import numpy as np

from tools.water_cup_dagger_mixing import choose_executed_action
from tools.water_cup_dagger_oracle import OracleDecision


class _Rng:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class WaterCupDaggerMixingTests(unittest.TestCase):
    def decision(self, phase="approach_object", forced=False):
        return OracleDecision(
            action=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]),
            phase=phase,
            force_expert=forced,
        )

    def test_forced_phase_always_executes_oracle(self) -> None:
        oracle = self.decision(phase="close_gripper", forced=True)
        oracle = OracleDecision(np.r_[np.zeros(6), 1.0], oracle.phase, True)
        result = choose_executed_action(
            np.full(7, -0.5), oracle, beta=0.0, rng=_Rng(0.99)
        )
        np.testing.assert_allclose(result.executed_action, oracle.action)
        np.testing.assert_allclose(result.oracle_label, oracle.action)
        self.assertEqual(result.source, "oracle_forced")

    def test_movement_uses_deterministic_oracle_probability(self) -> None:
        policy = np.full(7, -0.25)
        oracle = self.decision(phase="lift_object")
        selected = choose_executed_action(policy, oracle, beta=0.5, rng=_Rng(0.2))
        rejected = choose_executed_action(policy, oracle, beta=0.5, rng=_Rng(0.8))
        np.testing.assert_allclose(selected.executed_action, oracle.action)
        np.testing.assert_allclose(rejected.executed_action, policy)
        self.assertEqual(selected.source, "oracle_mixed")
        self.assertEqual(rejected.source, "policy")

    def test_policy_cannot_close_before_contact(self) -> None:
        policy = np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0])
        result = choose_executed_action(
            policy,
            self.decision(phase="approach_object"),
            beta=0.0,
            rng=_Rng(0.8),
        )
        np.testing.assert_allclose(result.executed_action[:6], policy[:6])
        self.assertEqual(result.executed_action[6], 0.0)
        self.assertTrue(result.gripper_gated)
        self.assertEqual(result.source, "policy_gated")


if __name__ == "__main__":
    unittest.main()
