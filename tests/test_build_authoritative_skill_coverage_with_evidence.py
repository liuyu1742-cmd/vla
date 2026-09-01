"""Tests that authoritative registry rebuilds replay audited evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_build_authoritative_skill_coverage import write_registry_workbook
from tools.build_authoritative_skill_coverage import main


class BuildAuthoritativeCoverageWithEvidenceTests(unittest.TestCase):
    def test_rebuild_replays_explicit_evidence_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scope.xlsx"
            output = root / "generated"
            evidence = root / "acceptance_summary.json"
            ledger = root / "evidence_ledger.json"
            write_registry_workbook(source)
            evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
            evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
            ledger.write_text(
                json.dumps(
                    {
                        "schema_version": "skill_coverage_evidence_ledger_v1",
                        "records": [
                            {
                                "relation_key": (
                                    "appliance_management::remote_control"
                                ),
                                "dataset_task_id": "appliance_management",
                                "service_task_id": "appliance_management",
                                "object_id": "remote_control",
                                "validation_state": "passed",
                                "evidence_level": "L3",
                                "evaluation_mode": "hybrid_closed_loop",
                                "pure_autonomous_vla": False,
                                "bundle_path": str(evidence),
                                "bundle_sha256": evidence_hash,
                                "heldout_seeds": [101, 102, 103],
                                "successful_seeds": 3,
                                "evaluated_seeds": 3,
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--xlsx",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--evidence-ledger",
                    str(ledger),
                    "--expected-dataset-tasks",
                    "2",
                    "--expected-relations",
                    "2",
                    "--expected-candidates",
                    "1",
                ]
            )
            summary = json.loads(
                (output / "coverage_summary.json").read_text(encoding="utf-8")
            )
            matrix = json.loads(
                (output / "task_object_matrix.json").read_text(encoding="utf-8")
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, summary["validated_object_count"])
        self.assertEqual(119, summary["remaining_validation_gap"])
        passed = [item for item in matrix if item["validation_state"] == "passed"]
        self.assertEqual(1, len(passed))
        self.assertEqual("appliance_management", passed[0]["service_task_id"])
        self.assertEqual("remote_control", passed[0]["object_id"])

    def test_rebuild_rejects_changed_evidence_bundle_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scope.xlsx"
            output = root / "generated"
            evidence = root / "acceptance_summary.json"
            ledger = root / "evidence_ledger.json"
            write_registry_workbook(source)
            evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
            ledger.write_text(
                json.dumps(
                    {
                        "schema_version": "skill_coverage_evidence_ledger_v1",
                        "records": [
                            {
                                "relation_key": (
                                    "appliance_management::remote_control"
                                ),
                                "service_task_id": "appliance_management",
                                "object_id": "remote_control",
                                "validation_state": "passed",
                                "evidence_level": "L3",
                                "evaluation_mode": "hybrid_closed_loop",
                                "pure_autonomous_vla": False,
                                "bundle_path": str(evidence),
                                "bundle_sha256": "0" * 64,
                                "heldout_seeds": [101, 102, 103],
                                "successful_seeds": 3,
                                "evaluated_seeds": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "bundle SHA-256 mismatch"):
                main(
                    [
                        "--xlsx",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--evidence-ledger",
                        str(ledger),
                        "--expected-dataset-tasks",
                        "2",
                        "--expected-relations",
                        "2",
                        "--expected-candidates",
                        "1",
                    ]
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
