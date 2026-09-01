"""Tests for resume-safe storage-box expert collection workers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.run_storage_box_collection_batch import _is_successful, build_command


class RunStorageBoxCollectionBatchTests(unittest.TestCase):
    def test_success_check_requires_native_storage_box_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_dir = root / "seed_007"
            episode_dir.mkdir()
            (episode_dir / "episode.npz").write_bytes(b"episode")
            (episode_dir / "report.json").write_text(
                json.dumps(
                    {
                        "relation_key": "organizing::storage_box",
                        "success": True,
                        "counts_toward_task2_coverage": True,
                        "success_predicates": {"simulator_success": True},
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(_is_successful(root, 7))

            report = json.loads((episode_dir / "report.json").read_text())
            report["success_predicates"]["simulator_success"] = False
            (episode_dir / "report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            self.assertFalse(_is_successful(root, 7))

    def test_command_uses_storage_box_collector(self) -> None:
        command = build_command(
            python="python",
            skill_ir=Path("skill.json"),
            seed=12,
            max_steps=1200,
            output_root=Path("episodes"),
        )

        self.assertEqual(
            ["python", "-u", "-m", "tools.collect_storage_box_expert"],
            command[:4],
        )
        self.assertIn("12", command)
        self.assertIn("1200", command)


if __name__ == "__main__":
    unittest.main()
