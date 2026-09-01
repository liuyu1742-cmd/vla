import json
import tempfile
import unittest
from pathlib import Path

from tools.finetune_vla82_robocasa365 import balanced_sample_weights, planned_training_summary


class FinetuneVLA82RoboCasa365Tests(unittest.TestCase):
    def test_balancing_upweights_rare_task_phase_cells(self):
        weights = balanced_sample_weights(
            [
                ("task_a", "moving"),
                ("task_a", "moving"),
                ("task_a", "holding"),
                ("task_b", "moving"),
            ]
        )
        self.assertGreater(weights[2], weights[0])
        self.assertGreater(weights[3], weights[0])

    def test_summary_counts_cached_samples_and_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "cache.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "vla82_robocasa365_cache_v1",
                        "task_class_count": 19,
                        "episode_count": 19,
                        "sample_count": 553,
                    }
                ),
                encoding="utf-8",
            )
            summary = planned_training_summary(manifest, epochs=2)
        self.assertEqual(summary["planned_updates"], 1106)
        self.assertEqual(summary["task_class_count"], 19)


if __name__ == "__main__":
    unittest.main()
