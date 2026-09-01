"""Tests for the shared Task-2-to-OpenVLA Skill IR."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


TOY_ACTIONS = [
    "locate(toy)",
    "grasp(toy)",
    "move(storage)",
    "place(toy)",
]


def toy_record() -> dict[str, object]:
    return {
        "relation_key": "organizing::toy",
        "task_id": "organizing",
        "task_name": "整理任务",
        "object_id": "toy",
        "object_name": "玩具",
        "instruction": "整理任务：整理并归位玩具（toy）。",
        "target": "none",
        "canonical_actions": list(TOY_ACTIONS),
    }


class HouseholdSkillIRV2Tests(unittest.TestCase):
    def test_load_contract_indexes_unique_relations(self) -> None:
        from tools.skill_transfer.skill_ir import load_contract

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps([toy_record()]), encoding="utf-8")

            contract = load_contract(path)

        self.assertEqual(["organizing::toy"], list(contract))

    def test_builds_skill_ir_with_null_unproven_evidence(self) -> None:
        from tools.skill_transfer.skill_ir import build_skill_ir

        contract = {"organizing::toy": toy_record()}
        skill_ir = build_skill_ir(
            "organizing::toy", contract, contract_version="task2_snapshot_test"
        )

        self.assertEqual("household_skill_ir_v2", skill_ir["schema_version"])
        self.assertEqual("organizing::toy", skill_ir["relation_key"])
        self.assertEqual(TOY_ACTIONS, skill_ir["canonical_actions"])
        self.assertIsNone(skill_ir["source_video"])
        self.assertIsNone(skill_ir["perception_evidence"])
        self.assertIsNone(skill_ir["parser_evidence"])
        self.assertIsNone(skill_ir["execution_evidence"])

    def test_rejects_unknown_relation(self) -> None:
        from tools.skill_transfer.skill_ir import build_skill_ir

        with self.assertRaisesRegex(ValueError, "unknown relation"):
            build_skill_ir("organizing::book", {}, contract_version="test")

    def test_rejects_task_or_object_mismatch_with_relation_key(self) -> None:
        from tools.skill_transfer.skill_ir import validate_skill_ir

        contract = {"organizing::toy": toy_record()}
        skill_ir = {
            "schema_version": "household_skill_ir_v2",
            "contract_version": "test",
            **toy_record(),
            "task_id": "object_fetching",
            "source_video": None,
            "perception_evidence": None,
            "parser_evidence": None,
            "execution_request": None,
            "execution_evidence": None,
        }
        with self.assertRaisesRegex(ValueError, "task_id"):
            validate_skill_ir(skill_ir, contract)

    def test_rejects_changed_canonical_action_order(self) -> None:
        from tools.skill_transfer.skill_ir import build_skill_ir, validate_skill_ir

        contract = {"organizing::toy": toy_record()}
        skill_ir = build_skill_ir(
            "organizing::toy", contract, contract_version="test"
        )
        skill_ir["canonical_actions"] = list(reversed(TOY_ACTIONS))

        with self.assertRaisesRegex(ValueError, "canonical_actions"):
            validate_skill_ir(skill_ir, contract)

    def test_rejects_all_zero_keypoint_evidence(self) -> None:
        from tools.skill_transfer.skill_ir import build_skill_ir, validate_skill_ir

        contract = {"organizing::toy": toy_record()}
        skill_ir = build_skill_ir(
            "organizing::toy", contract, contract_version="test"
        )
        skill_ir["perception_evidence"] = {
            "keypoints": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        }

        with self.assertRaisesRegex(ValueError, "all-zero keypoints"):
            validate_skill_ir(skill_ir, contract)

    def test_success_requires_all_predicates_and_action_progress(self) -> None:
        from tools.skill_transfer.skill_ir import build_skill_ir, validate_skill_ir

        contract = {"organizing::toy": toy_record()}
        skill_ir = build_skill_ir(
            "organizing::toy", contract, contract_version="test"
        )
        skill_ir["execution_evidence"] = {
            "success": True,
            "canonical_action_progress": 3,
            "success_predicates": {"inside_storage": True},
        }

        with self.assertRaisesRegex(ValueError, "canonical_action_progress"):
            validate_skill_ir(skill_ir, contract)

    def test_write_skill_ir_validates_before_writing(self) -> None:
        from tools.skill_transfer.skill_ir import build_skill_ir, write_skill_ir

        contract = {"organizing::toy": toy_record()}
        skill_ir = build_skill_ir(
            "organizing::toy", contract, contract_version="test"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "skill_ir.json"
            write_skill_ir(output, skill_ir, contract)
            stored = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual("organizing::toy", stored["relation_key"])


if __name__ == "__main__":
    unittest.main()
