from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.finetune_formal_skill_openvla_resumable import (
    find_latest_checkpoint,
    resume_position,
)


class FormalSkillResumableTrainingTests(unittest.TestCase):
    def test_resume_position_maps_updates_to_epoch_and_offset(self) -> None:
        self.assertEqual(resume_position(0, 4042), (0, 0))
        self.assertEqual(resume_position(500, 4042), (0, 500))
        self.assertEqual(resume_position(4042, 4042), (1, 0))
        self.assertEqual(resume_position(4542, 4042), (1, 500))

    def test_latest_complete_matching_checkpoint_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step, manifest_hash in ((500, "a" * 64), (1000, "a" * 64), (1500, "b" * 64)):
                checkpoint = root / f"checkpoint_step_{step:06d}"
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
                (checkpoint / "optimizer.pt").write_bytes(b"state")
                (checkpoint / "checkpoint_state.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "formal_skill_training_checkpoint_v1",
                            "completed_updates": step,
                            "manifest_sha256": manifest_hash,
                            "planned_updates": 8084,
                        }
                    ),
                    encoding="utf-8",
                )
            selected = find_latest_checkpoint(
                root, manifest_sha256="a" * 64, planned_updates=8084
            )
            self.assertIsNotNone(selected)
            self.assertEqual(selected.name, "checkpoint_step_001000")

    def test_incomplete_checkpoint_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_step_000500"
            checkpoint.mkdir()
            (checkpoint / "checkpoint_state.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(
                find_latest_checkpoint(root, manifest_sha256="a" * 64, planned_updates=8084)
            )


if __name__ == "__main__":
    unittest.main()
