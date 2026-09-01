import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.collect_water_cup_dagger import save_episode, validate_training_seed


class CollectWaterCupDaggerTests(unittest.TestCase):
    def test_seed_two_is_rejected_from_recovery_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "held-out seed 2"):
            validate_training_seed(2)

    def test_saved_episode_uses_oracle_actions_as_training_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
            oracle = np.array([[0.1] * 7, [0.2] * 7], dtype=np.float32)
            policy = np.array([[0.3] * 7, [0.4] * 7], dtype=np.float32)
            executed = np.array([[0.5] * 7, [0.6] * 7], dtype=np.float32)

            episode_path, manifest_path = save_episode(
                root,
                seed=0,
                frames=frames,
                oracle_actions=oracle,
                policy_actions=policy,
                executed_actions=executed,
                report={"success": False, "beta": 0.5},
            )

            with np.load(episode_path) as data:
                np.testing.assert_allclose(data["actions"], oracle)
                np.testing.assert_allclose(data["oracle_actions"], oracle)
                np.testing.assert_allclose(data["policy_actions"], policy)
                self.assertEqual(data["frames"].shape[0], data["actions"].shape[0])
                self.assertEqual(data["actions"].shape[1], 7)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"], "dagger_recovery")
            self.assertEqual(manifest["samples"], 2)
            self.assertEqual(manifest["instruction"], "pick up the glass cup and place it in the cabinet")


if __name__ == "__main__":
    unittest.main()
