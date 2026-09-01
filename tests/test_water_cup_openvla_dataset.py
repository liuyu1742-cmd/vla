import tempfile
import unittest
from pathlib import Path

import numpy as np


class WaterCupOpenVLADatasetTests(unittest.TestCase):
    def test_split_uses_held_out_seed(self):
        from tools.finetune_water_cup_openvla import episode_paths

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (0, 1, 2):
                np.savez(root / f"episode_seed_{seed:03d}.npz", frames=np.zeros((1, 2, 2, 3), dtype=np.uint8), actions=np.zeros((1, 7), dtype=np.float32))
            train = episode_paths(root, held_out_seed=2, train=True)
            test = episode_paths(root, held_out_seed=2, train=False)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 1)
            self.assertIn("002", test[0].name)


if __name__ == "__main__":
    unittest.main()
