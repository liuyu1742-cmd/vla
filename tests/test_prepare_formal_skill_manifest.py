"""Tests for leakage-free formal skill training and held-out manifests."""

from __future__ import annotations

import unittest

from tools.prepare_formal_skill_manifest import build_training_manifest


def episode(seed: int, *, success: bool = True) -> dict[str, object]:
    return {
        "seed": seed,
        "relation_key": "organizing::toy",
        "success": success,
        "counts_toward_task2_coverage": success,
        "samples": 10 + seed,
        "episode": f"seed_{seed:03d}/episode.npz",
        "report": f"seed_{seed:03d}/report.json",
        "episode_sha256": f"episode-{seed}",
        "report_sha256": f"report-{seed}",
        "skill_ir_sha256": "same-skill-ir",
    }


class PrepareFormalSkillManifestTests(unittest.TestCase):
    def test_builds_disjoint_training_and_heldout_splits(self) -> None:
        records = [episode(seed) for seed in (0, 1, 101, 102, 103)]
        manifest = build_training_manifest(
            records,
            heldout_seeds=[101, 102, 103],
            min_training=2,
        )
        self.assertEqual([0, 1], [item["seed"] for item in manifest["train"]])
        self.assertEqual(
            [101, 102, 103], [item["seed"] for item in manifest["held_out"]]
        )
        self.assertEqual(2, manifest["training_episode_count"])
        self.assertEqual(3, manifest["heldout_episode_count"])

    def test_failed_episode_is_never_included(self) -> None:
        records = [episode(0), episode(1, success=False), episode(101)]
        manifest = build_training_manifest(
            records, heldout_seeds=[101], min_training=1
        )
        self.assertEqual([0], [item["seed"] for item in manifest["train"]])

    def test_duplicate_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            build_training_manifest(
                [episode(0), episode(0), episode(101)],
                heldout_seeds=[101],
                min_training=1,
            )

    def test_missing_heldout_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "held-out"):
            build_training_manifest(
                [episode(0), episode(101)],
                heldout_seeds=[101, 102],
                min_training=1,
            )

    def test_too_few_training_episodes_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "training episodes"):
            build_training_manifest(
                [episode(0), episode(101)],
                heldout_seeds=[101],
                min_training=2,
            )


if __name__ == "__main__":
    unittest.main()
