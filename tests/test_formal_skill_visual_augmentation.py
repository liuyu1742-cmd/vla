import unittest

import numpy as np

from tools.formal_skill_visual_augmentation import augment_training_frame


class FormalSkillVisualAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[:32, :32]
        self.frame = np.stack(
            (
                (x * 7) % 256,
                (y * 9) % 256,
                ((x + y) * 5) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)

    def test_epoch_zero_preserves_the_original_frame(self) -> None:
        augmented = augment_training_frame(
            self.frame, seed=23, epoch=0, sample_index=17
        )

        np.testing.assert_array_equal(augmented, self.frame)
        self.assertIsNot(augmented, self.frame)

    def test_later_epoch_is_deterministic_and_changes_appearance(self) -> None:
        first = augment_training_frame(
            self.frame, seed=23, epoch=1, sample_index=17
        )
        second = augment_training_frame(
            self.frame, seed=23, epoch=1, sample_index=17
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, self.frame.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertFalse(np.array_equal(first, self.frame))

    def test_invalid_frame_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "H,W,3"):
            augment_training_frame(
                np.zeros((32, 32), dtype=np.uint8),
                seed=23,
                epoch=1,
                sample_index=0,
            )


if __name__ == "__main__":
    unittest.main()
