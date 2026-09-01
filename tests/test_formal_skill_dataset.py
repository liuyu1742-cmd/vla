"""Tests for manifest-driven formal Skill IR demonstration loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.formal_skill_dataset import FormalSkillDataset


class FormalSkillDatasetTests(unittest.TestCase):
    def make_episode(self, root: Path, seed: int, samples: int) -> Path:
        path = root / f"seed_{seed:03d}.npz"
        np.savez_compressed(
            path,
            frames=np.full((samples, 6, 8, 3), seed, dtype=np.uint8),
            actions=np.full((samples, 7), seed / 100.0, dtype=np.float32),
            phases=np.asarray(["approach_object"] * samples),
            canonical_phases=np.asarray(["locate"] * samples),
            relation_key=np.asarray("organizing::toy"),
            instruction=np.asarray("organize the toy into storage"),
        )
        return path

    def test_train_split_excludes_heldout_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = self.make_episode(root, 0, 2)
            heldout_path = self.make_episode(root, 101, 1)
            manifest = {
                "schema_version": "formal_skill_training_manifest_v1",
                "relation_key": "organizing::toy",
                "heldout_seeds": [101],
                "train": [{"seed": 0, "samples": 2, "episode": str(train_path)}],
                "held_out": [
                    {"seed": 101, "samples": 1, "episode": str(heldout_path)}
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            train = FormalSkillDataset(manifest_path, split="train")
            heldout = FormalSkillDataset(manifest_path, split="held_out")

            self.assertEqual(2, len(train))
            self.assertEqual(1, len(heldout))
            sample = train[1]
            self.assertEqual(0, sample["seed"])
            self.assertEqual("organizing::toy", sample["relation_key"])
            self.assertEqual("locate", sample["canonical_phase"])
            self.assertEqual("approach_object", sample["phase"])
            self.assertEqual((7,), sample["action"].shape)
            self.assertEqual((6, 8, 3), sample["frame"].shape)

    def test_manifest_seed_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_path = self.make_episode(root, 101, 1)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "formal_skill_training_manifest_v1",
                        "relation_key": "organizing::toy",
                        "heldout_seeds": [101],
                        "train": [
                            {"seed": 101, "samples": 1, "episode": str(episode_path)}
                        ],
                        "held_out": [
                            {"seed": 101, "samples": 1, "episode": str(episode_path)}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                FormalSkillDataset(manifest_path, split="train")

    def test_action_dimension_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_path = root / "bad.npz"
            np.savez_compressed(
                episode_path,
                frames=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                actions=np.zeros((1, 6), dtype=np.float32),
                phases=np.asarray(["approach_object"]),
                canonical_phases=np.asarray(["locate"]),
                relation_key=np.asarray("organizing::toy"),
                instruction=np.asarray("instruction"),
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "formal_skill_training_manifest_v1",
                        "relation_key": "organizing::toy",
                        "heldout_seeds": [],
                        "train": [
                            {"seed": 0, "samples": 1, "episode": str(episode_path)}
                        ],
                        "held_out": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "actions"):
                FormalSkillDataset(manifest_path, split="train")


if __name__ == "__main__":
    unittest.main()
