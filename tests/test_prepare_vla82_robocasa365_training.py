import unittest

import numpy as np

from tools.prepare_vla82_robocasa365_training import (
    extract_openvla_action,
    select_balanced_episodes,
)


class PrepareVLA82RoboCasa365TrainingTests(unittest.TestCase):
    def test_extracts_arm_channels_from_official_lerobot_order(self):
        action = np.arange(12, dtype=np.float32)
        self.assertEqual(
            extract_openvla_action(action).tolist(),
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        )

    def test_selects_equal_episode_count_per_task_class(self):
        rows = [
            {"episode_index": 1, "source_prefix": "pretrain/atomic/A/date"},
            {"episode_index": 2, "source_prefix": "pretrain/atomic/A/date"},
            {"episode_index": 3, "source_prefix": "pretrain/atomic/B/date"},
            {"episode_index": 4, "source_prefix": "pretrain/atomic/B/date"},
        ]
        selected = select_balanced_episodes(rows, ["A", "B"], episodes_per_class=1)
        self.assertEqual(
            [(item["task_class"], item["episode_index"]) for item in selected],
            [("A", 1), ("B", 3)],
        )


if __name__ == "__main__":
    unittest.main()
