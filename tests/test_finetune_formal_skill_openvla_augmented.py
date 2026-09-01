import unittest

import numpy as np

import tools.finetune_formal_skill_openvla_augmented as augmented


class AugmentedFormalTrainerTests(unittest.TestCase):
    def _dataset(self):
        dataset = augmented.AugmentedCachedPhaseDataset.__new__(
            augmented.AugmentedCachedPhaseDataset
        )
        dataset.index = [(0, 0)]
        dataset.episodes = [
            {
                "instruction": "put away the toy in the cabinet",
                "relation_key": "organizing::toy",
                "seed": 12,
            }
        ]
        frame = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        dataset._episode_caches = {
            0: {
                "frames": frame[None, ...],
                "actions": np.zeros((1, 7), dtype=np.float32),
                "phases": np.asarray(["approach_object"]),
                "canonical_phases": np.asarray(["locate"]),
            }
        }
        return dataset, frame

    def test_epoch_order_controls_deterministic_frame_augmentation(self) -> None:
        dataset, frame = self._dataset()

        augmented.augmented_epoch_sample_order(
            4, seed=23, epoch=0, start=0
        )
        np.testing.assert_array_equal(dataset[0]["frame"], frame)

        augmented.augmented_epoch_sample_order(
            4, seed=23, epoch=1, start=0
        )
        changed = dataset[0]["frame"]
        self.assertEqual(changed.dtype, np.uint8)
        self.assertFalse(np.array_equal(changed, frame))


if __name__ == "__main__":
    unittest.main()
