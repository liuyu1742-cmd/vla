import collections
import unittest

from tools.formal_skill_phase_balancing import balanced_phase_epoch_order


class FormalSkillPhaseBalancingTests(unittest.TestCase):
    def test_epoch_has_equal_phase_quotas_despite_imbalanced_input(self):
        phases = ["locate"] * 2 + ["grasp"] * 3 + ["move"] * 20 + ["place"] * 5
        order = balanced_phase_epoch_order(phases, seed=23, epoch=0)
        counts = collections.Counter(phases[index] for index in order)

        self.assertEqual(len(order), len(phases))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(set(counts), {"locate", "grasp", "move", "place"})

    def test_minority_phase_is_oversampled_and_resume_is_reproducible(self):
        phases = ["locate"] + ["grasp"] * 2 + ["move"] * 9 + ["place"] * 4
        full = balanced_phase_epoch_order(phases, seed=4, epoch=2)
        resumed = balanced_phase_epoch_order(phases, seed=4, epoch=2, start=7)

        self.assertEqual(resumed, full[7:])
        self.assertGreater(full.count(0), 1)

    def test_missing_required_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            balanced_phase_epoch_order(
                ["locate", "grasp", "move"], seed=1, epoch=0
            )


if __name__ == "__main__":
    unittest.main()
