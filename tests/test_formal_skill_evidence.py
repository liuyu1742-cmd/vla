"""Tests for separating infrastructure smoke runs from formal skill evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.skill_transfer.evidence import (
    FormalEvidenceError,
    infrastructure_smoke_metadata,
    validate_formal_skill_evidence,
)


class FormalSkillEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "organizing::toy": {
                "relation_key": "organizing::toy",
                "canonical_actions": [
                    "locate(toy)",
                    "grasp(toy)",
                    "move(storage)",
                    "place(toy)",
                ],
            }
        }

    def test_water_cup_runs_are_explicitly_infrastructure_smoke(self) -> None:
        self.assertEqual(
            {
                "evidence_scope": "infrastructure_smoke",
                "relation_key": None,
                "counts_toward_task2_coverage": False,
            },
            infrastructure_smoke_metadata(),
        )

    def test_successful_smoke_report_is_rejected_as_formal_evidence(self) -> None:
        report = {"success": True, **infrastructure_smoke_metadata()}
        with self.assertRaisesRegex(FormalEvidenceError, "infrastructure_smoke"):
            validate_formal_skill_evidence(report, self.contract)

    def test_unknown_relation_is_rejected(self) -> None:
        report = self._formal_report(relation_key="cleaning::toy")
        with self.assertRaisesRegex(FormalEvidenceError, "not in Task-2 contract"):
            validate_formal_skill_evidence(report, self.contract)

    def test_action_mismatch_is_rejected(self) -> None:
        report = self._formal_report(canonical_actions=["locate(toy)"])
        with self.assertRaisesRegex(FormalEvidenceError, "canonical actions"):
            validate_formal_skill_evidence(report, self.contract)

    def test_success_requires_all_relation_predicates(self) -> None:
        report = self._formal_report(success_predicates={"placed_in_storage": False})
        with self.assertRaisesRegex(FormalEvidenceError, "success predicates"):
            validate_formal_skill_evidence(report, self.contract)

    def test_valid_formal_report_counts_toward_coverage(self) -> None:
        validated = validate_formal_skill_evidence(self._formal_report(), self.contract)
        self.assertTrue(validated["counts_toward_task2_coverage"])
        self.assertEqual("organizing::toy", validated["relation_key"])

    def _formal_report(self, **overrides: object) -> dict[str, object]:
        report: dict[str, object] = {
            "evidence_scope": "formal_skill_execution",
            "relation_key": "organizing::toy",
            "counts_toward_task2_coverage": True,
            "success": True,
            "canonical_actions": list(
                self.contract["organizing::toy"]["canonical_actions"]
            ),
            "canonical_action_progress": 4,
            "success_predicates": {"placed_in_storage": True},
        }
        report.update(overrides)
        return report


if __name__ == "__main__":
    unittest.main()
