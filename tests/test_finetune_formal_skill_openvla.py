from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.finetune_formal_skill_openvla import (
    action_statistics,
    build_instruction,
    normalize_action,
    planned_training_summary,
)


class FormalSkillTrainingTests(unittest.TestCase):
    def test_action_statistics_and_normalization_use_training_quantiles(self) -> None:
        actions = np.asarray(
            [
                [-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        stats = action_statistics(actions, lower=0.0, upper=1.0)
        self.assertEqual(stats["mask"], [True, True, True, False, False, False, True])
        normalized = normalize_action(actions[1], stats)
        np.testing.assert_allclose(
            normalized,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
        )

    def test_prompt_is_conditioned_on_skill_phase(self) -> None:
        prompt = build_instruction(
            "pick up the toy and place it in the cabinet", "grasp"
        )
        self.assertIn("pick up the toy and place it in the cabinet", prompt)
        self.assertIn("current skill phase is grasp", prompt)

    def test_plan_uses_only_train_split_and_records_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_manifest.json"
            manifest = {
                "schema_version": "formal_skill_training_manifest_v1",
                "relation_key": "organizing::toy",
                "skill_ir_sha256": "a" * 64,
                "training_episode_count": 2,
                "heldout_episode_count": 1,
                "training_sample_count": 5,
                "heldout_sample_count": 4,
                "train": [{"seed": 0}, {"seed": 1}],
                "held_out": [{"seed": 101}],
            }
            encoded = json.dumps(manifest).encode("utf-8")
            path.write_bytes(encoded)
            summary = planned_training_summary(path, epochs=3, batch_size=2)

            self.assertEqual(summary["train_seeds"], [0, 1])
            self.assertEqual(summary["heldout_seeds"], [101])
            self.assertEqual(summary["planned_updates"], 9)
            self.assertEqual(summary["manifest_sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertNotIn(101, summary["train_seeds"])


if __name__ == "__main__":
    unittest.main()
