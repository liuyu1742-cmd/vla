"""Pure persistence tests for formal expert demonstrations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.collect_formal_skill_expert import (
    build_formal_report,
    save_formal_episode,
    validate_skill_ir_for_collection,
)


class CollectFormalSkillExpertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skill_ir = {
            "schema_version": "household_skill_ir_v2",
            "relation_key": "organizing::toy",
            "task_id": "organizing",
            "object_id": "toy",
            "instruction": "organize the toy into storage",
            "canonical_actions": [
                "locate(toy)",
                "grasp(toy)",
                "move(storage)",
                "place(toy)",
            ],
        }

    def test_collection_requires_exact_first_formal_relation(self) -> None:
        validate_skill_ir_for_collection(self.skill_ir)
        invalid = dict(self.skill_ir, relation_key="organizing::book")
        with self.assertRaisesRegex(ValueError, "organizing::toy"):
            validate_skill_ir_for_collection(invalid)

    def test_failed_report_never_counts_toward_coverage(self) -> None:
        report = build_formal_report(
            self.skill_ir,
            seed=0,
            success=False,
            executed_steps=12,
            predicates={"placed_in_storage": False},
        )
        self.assertEqual("formal_skill_execution", report["evidence_scope"])
        self.assertFalse(report["counts_toward_task2_coverage"])
        self.assertEqual(0, report["canonical_action_progress"])

    def test_episode_contains_relation_phase_state_and_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_formal_report(
                self.skill_ir,
                seed=4,
                success=True,
                executed_steps=2,
                predicates={"placed_in_storage": True, "gripper_released": True},
            )
            episode_path, report_path = save_formal_episode(
                root,
                seed=4,
                frames=np.zeros((2, 8, 8, 3), dtype=np.uint8),
                actions=np.zeros((2, 7), dtype=np.float32),
                phases=["approach_object", "release_object"],
                canonical_phases=["locate", "place"],
                eef_positions=np.zeros((2, 3), dtype=np.float32),
                object_positions=np.ones((2, 3), dtype=np.float32),
                grasped=[False, True],
                report=report,
            )
            with np.load(episode_path) as episode:
                self.assertEqual((2, 7), episode["actions"].shape)
                self.assertEqual("organizing::toy", str(episode["relation_key"]))
                self.assertEqual(["locate", "place"], episode["canonical_phases"].tolist())
                self.assertEqual((2, 3), episode["eef_positions"].shape)
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(saved_report["success"])


if __name__ == "__main__":
    unittest.main()
