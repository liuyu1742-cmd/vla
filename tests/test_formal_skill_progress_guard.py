import unittest

import numpy as np

from tools.formal_skill_progress_guard import select_progress_guarded_action


class FormalSkillProgressGuardTests(unittest.TestCase):
    def test_rejects_translation_that_opposes_verified_expert(self):
        decision = select_progress_guarded_action(
            raw_action=[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            expert_action=[1.0, 0.5, -1.0, 0.0, 0.0, 0.0, -1.0],
            force_expert=False,
        )

        self.assertEqual(decision.mode, "safety_expert")
        self.assertTrue(decision.intervened)
        np.testing.assert_allclose(
            decision.action, [1.0, 0.5, -1.0, 0.0, 0.0, 0.0, -1.0]
        )

    def test_keeps_bounded_model_residual_when_translation_is_aligned(self):
        decision = select_progress_guarded_action(
            raw_action=[0.8, 0.4, -0.8, 0.0, 0.0, 0.0, -1.0],
            expert_action=[1.0, 0.5, -1.0, 0.0, 0.0, 0.0, -1.0],
            force_expert=False,
            model_weight=0.15,
        )

        self.assertEqual(decision.mode, "aligned_model_residual")
        self.assertFalse(decision.intervened)
        np.testing.assert_allclose(
            decision.action[:3], [0.97, 0.485, -0.97], atol=1e-6
        )
        self.assertEqual(float(decision.action[6]), -1.0)

    def test_forced_expert_hold_never_uses_model_translation(self):
        decision = select_progress_guarded_action(
            raw_action=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            expert_action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            force_expert=True,
        )

        self.assertEqual(decision.mode, "forced_expert")
        self.assertTrue(decision.intervened)
        np.testing.assert_allclose(
            decision.action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )

    def test_rejects_bad_shapes_and_non_finite_values(self):
        with self.assertRaises(ValueError):
            select_progress_guarded_action([0.0], np.zeros(7), force_expert=False)
        with self.assertRaises(ValueError):
            select_progress_guarded_action(
                np.full(7, np.nan), np.zeros(7), force_expert=False
            )


if __name__ == "__main__":
    unittest.main()
