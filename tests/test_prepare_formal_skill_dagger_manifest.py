from __future__ import annotations

import unittest

from tools.prepare_formal_skill_dagger_manifest import build_dagger_manifest


class PrepareFormalSkillDaggerManifestTests(unittest.TestCase):
    def nominal(self) -> dict:
        return {
            "schema_version": "formal_skill_training_manifest_v1",
            "relation_key": "organizing::toy",
            "skill_ir_sha256": "skill-hash",
            "minimum_training_episodes": 1,
            "training_episode_count": 1,
            "heldout_episode_count": 1,
            "training_sample_count": 10,
            "heldout_sample_count": 5,
            "train": [
                {
                    "seed": 0,
                    "samples": 10,
                    "episode": "nominal.npz",
                    "report": "nominal.json",
                }
            ],
            "held_out": [
                {
                    "seed": 101,
                    "samples": 5,
                    "episode": "heldout.npz",
                    "report": "heldout.json",
                }
            ],
        }

    def test_aggregates_nominal_and_weighted_recovery(self) -> None:
        recovery = [
            {
                "seed": 42,
                "samples": 4,
                "source": "dagger_recovery",
                "episode": "recovery.npz",
                "report": "recovery.json",
            }
        ]

        result = build_dagger_manifest(
            self.nominal(), recovery, recovery_repeat=2
        )

        self.assertEqual(3, result["training_episode_count"])
        self.assertEqual(18, result["training_sample_count"])
        self.assertEqual([101], [item["seed"] for item in result["held_out"]])
        self.assertEqual(
            ["nominal_expert", "dagger_recovery", "dagger_recovery"],
            [item["source"] for item in result["train"]],
        )
        self.assertEqual([0, 0, 1], [item["repeat_index"] for item in result["train"]])
        self.assertEqual(4, result["dagger"]["unique_recovery_samples"])
        self.assertEqual(8, result["dagger"]["weighted_recovery_samples"])

    def test_rejects_recovery_on_heldout_seed(self) -> None:
        recovery = [
            {
                "seed": 101,
                "samples": 4,
                "source": "dagger_recovery",
                "episode": "forbidden.npz",
                "report": "forbidden.json",
            }
        ]

        with self.assertRaisesRegex(ValueError, "held-out"):
            build_dagger_manifest(
                self.nominal(), recovery, recovery_repeat=2
            )

    def test_requires_at_least_one_recovery_episode(self) -> None:
        with self.assertRaisesRegex(ValueError, "recovery"):
            build_dagger_manifest(
                self.nominal(), [], recovery_repeat=2
            )


if __name__ == "__main__":
    unittest.main()
