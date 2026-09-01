import unittest

from tools.formal_skill_training_order import epoch_sample_order


class FormalSkillTrainingOrderTests(unittest.TestCase):
    def test_each_epoch_is_a_complete_deterministic_permutation(self):
        first = epoch_sample_order(32, seed=23, epoch=0)
        repeated = epoch_sample_order(32, seed=23, epoch=0)

        self.assertEqual(first, repeated)
        self.assertEqual(sorted(first), list(range(32)))
        self.assertNotEqual(first, list(range(32)))

    def test_epochs_do_not_reuse_episode_contiguous_order(self):
        self.assertNotEqual(
            epoch_sample_order(64, seed=23, epoch=0),
            epoch_sample_order(64, seed=23, epoch=1),
        )

    def test_resume_offset_is_exact_suffix_of_same_epoch(self):
        full = epoch_sample_order(21, seed=7, epoch=3)
        resumed = epoch_sample_order(21, seed=7, epoch=3, start=8)

        self.assertEqual(resumed, full[8:])

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            epoch_sample_order(0, seed=1, epoch=0)
        with self.assertRaises(ValueError):
            epoch_sample_order(3, seed=1, epoch=-1)
        with self.assertRaises(ValueError):
            epoch_sample_order(3, seed=1, epoch=0, start=4)


if __name__ == "__main__":
    unittest.main()
