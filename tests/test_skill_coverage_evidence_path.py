"""Regression test for portable ledger and directly resolvable matrix evidence."""

from __future__ import annotations

import unittest

from tools.skill_coverage.evidence_merge import overlay_evidence_ledger


class SkillCoverageEvidencePathTests(unittest.TestCase):
    def test_matrix_uses_verified_absolute_bundle_path(self) -> None:
        matrix = [
            {
                "service_task_id": "organization_storage",
                "object_id": "toy",
                "applicability": "direct",
                "mapping_basis": "dataset_task:organizing",
                "validation_state": "untested",
                "evidence": [],
            }
        ]
        ledger = {
            "schema_version": "skill_coverage_evidence_ledger_v1",
            "records": [
                {
                    "relation_key": "organizing::toy",
                    "service_task_id": "organization_storage",
                    "object_id": "toy",
                    "validation_state": "passed",
                    "evidence_level": "L3",
                    "evaluation_mode": "hybrid_closed_loop",
                    "pure_autonomous_vla": False,
                    "bundle_path": (
                        "outputs/formal_skill_dagger_r3_hybrid_v3_eval/"
                        "acceptance_summary.json"
                    ),
                    "bundle_absolute_path": (
                        "C:\\OpenVLA-Simulator\\outputs\\"
                        "formal_skill_dagger_r3_hybrid_v3_eval\\"
                        "acceptance_summary.json"
                    ),
                    "bundle_sha256": "a" * 64,
                    "heldout_seeds": [101, 102, 103],
                    "successful_seeds": 3,
                    "evaluated_seeds": 3,
                }
            ],
        }

        updated = overlay_evidence_ledger(matrix, ledger)

        self.assertEqual(
            ledger["records"][0]["bundle_absolute_path"],
            updated[0]["evidence"][0]["path"],
        )


if __name__ == "__main__":
    unittest.main()
