"""Tests for audited acceptance-bundle ingestion into skill coverage."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.skill_coverage.evidence_merge import (
    coverage_with_preserved_metadata,
    merge_ledger_record,
    overlay_evidence_ledger,
    validate_hybrid_acceptance_bundle,
)


class SkillCoverageEvidenceMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = [
            {
                "relation_key": "organizing::toy",
                "task_id": "organizing",
                "object_id": "toy",
                "canonical_actions": [
                    "locate(toy)",
                    "grasp(toy)",
                    "move(storage)",
                    "place(toy)",
                ],
            }
        ]
        self.policy = {
            "primary_by_dataset_task": {"organizing": "organization_storage"}
        }
        self.bundle_path = self.root / "acceptance_summary.json"
        self._write_bundle()

    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _report(self, seed: int) -> dict[str, object]:
        return {
            "schema_version": "formal_skill_hybrid_model_evaluation_v1",
            "evidence_scope": "formal_skill_hybrid_model_evaluation",
            "relation_key": "organizing::toy",
            "seed": seed,
            "split": "held_out",
            "max_decision_steps": 300,
            "success": True,
            "success_predicates": {
                "simulator_success": True,
                "placed_in_storage": True,
                "gripper_released": True,
            },
            "ever_grasped": True,
            "ever_inside": True,
            "adapter_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "counts_toward_task2_coverage": True,
            "counts_toward_autonomous_vla_acceptance": False,
            "execution_mode_counts": {
                "openvla_guarded": 50,
                "state_aware_locate_recovery_safety_expert": 10,
                "calibrated_pick_timing": 40,
                "final_contact_servo": 100,
            },
        }

    def _bundle(self) -> dict[str, object]:
        results = []
        for seed in (101, 102, 103):
            report_path = self.root / "outputs" / f"seed_{seed}" / "report.json"
            self._write_json(report_path, self._report(seed))
            results.append(
                {
                    "seed": seed,
                    "success": True,
                    "success_predicates": {
                        "simulator_success": True,
                        "placed_in_storage": True,
                        "gripper_released": True,
                    },
                    "ever_grasped": True,
                    "ever_inside": True,
                    "report": str(report_path),
                }
            )
        return {
            "schema_version": "formal_skill_hybrid_acceptance_summary_v1",
            "relation_key": "organizing::toy",
            "split": "held_out",
            "protocol": {
                "seeds": [101, 102, 103],
                "max_decision_steps_per_seed": 300,
                "action_repeat": 2,
            },
            "model": {"adapter_sha256": "a" * 64},
            "data_isolation": {"manifest_sha256": "b" * 64},
            "evaluation_mode": {
                "kind": "hybrid_closed_loop",
                "state_aware_locate_recovery": True,
                "final_contact_servo": True,
                "pure_autonomous_vla": False,
            },
            "results": results,
            "aggregate": {
                "successful_seeds": 3,
                "evaluated_seeds": 3,
                "success_rate": 1.0,
                "all_native_success_predicates_true": True,
                "counts_toward_task2_coverage": True,
                "counts_toward_autonomous_vla_acceptance": False,
            },
        }

    def _write_bundle(self, mutate=None) -> dict[str, object]:
        bundle = self._bundle()
        if mutate is not None:
            mutate(bundle)
        self._write_json(self.bundle_path, bundle)
        return bundle

    def _validate(self) -> dict[str, object]:
        return validate_hybrid_acceptance_bundle(
            self.bundle_path,
            self.contract,
            self.policy,
            self.root,
        )

    def test_maps_organizing_toy_to_organization_storage(self) -> None:
        record = self._validate()

        self.assertEqual("organizing::toy", record["relation_key"])
        self.assertEqual("organizing", record["dataset_task_id"])
        self.assertEqual("organization_storage", record["service_task_id"])
        self.assertEqual("toy", record["object_id"])
        self.assertEqual("L3", record["evidence_level"])
        self.assertEqual("hybrid_closed_loop", record["evaluation_mode"])
        self.assertFalse(record["pure_autonomous_vla"])
        self.assertEqual(
            hashlib.sha256(self.bundle_path.read_bytes()).hexdigest(),
            record["bundle_sha256"],
        )

    def test_rejects_bundle_without_three_native_successes(self) -> None:
        self._write_bundle(
            lambda bundle: bundle["aggregate"].update(
                {"successful_seeds": 2, "success_rate": 2 / 3}
            )
        )

        with self.assertRaisesRegex(ValueError, "successful seeds"):
            self._validate()

    def test_rejects_autonomous_claim_for_hybrid_bundle(self) -> None:
        self._write_bundle(
            lambda bundle: bundle["evaluation_mode"].update(
                {"pure_autonomous_vla": True}
            )
        )

        with self.assertRaisesRegex(ValueError, "pure autonomous"):
            self._validate()

    def test_rejects_report_fingerprint_mismatch(self) -> None:
        bundle = self._write_bundle()
        report_path = Path(bundle["results"][0]["report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["adapter_sha256"] = "c" * 64
        self._write_json(report_path, report)

        with self.assertRaisesRegex(ValueError, "adapter fingerprint"):
            self._validate()

    def test_same_relation_replaces_instead_of_duplicates(self) -> None:
        record = self._validate()
        ledger = {
            "schema_version": "skill_coverage_evidence_ledger_v1",
            "records": [{**record, "bundle_sha256": "0" * 64}],
        }

        merged = merge_ledger_record(ledger, record)

        self.assertEqual(1, len(merged["records"]))
        self.assertEqual(record["bundle_sha256"], merged["records"][0]["bundle_sha256"])

    def test_overlay_marks_only_organization_storage_toy_passed(self) -> None:
        record = self._validate()
        ledger = {
            "schema_version": "skill_coverage_evidence_ledger_v1",
            "records": [record],
        }
        matrix = [
            {
                "service_task_id": "cleaning_service",
                "object_id": "toy",
                "applicability": "not_applicable",
                "mapping_basis": "default_not_applicable",
                "validation_state": "untested",
                "evidence": [],
            },
            {
                "service_task_id": "organization_storage",
                "object_id": "toy",
                "applicability": "direct",
                "mapping_basis": "dataset_task:organizing",
                "validation_state": "untested",
                "evidence": [],
            },
        ]

        updated = overlay_evidence_ledger(matrix, ledger)

        passed = [item for item in updated if item["validation_state"] == "passed"]
        self.assertEqual(1, len(passed))
        self.assertEqual("organization_storage", passed[0]["service_task_id"])
        self.assertEqual("toy", passed[0]["object_id"])
        self.assertEqual("L3", passed[0]["evidence"][0]["evidence_level"])
        self.assertFalse(passed[0]["evidence"][0]["pure_autonomous_vla"])

    def test_overlay_rejects_not_applicable_target(self) -> None:
        record = self._validate()
        record["service_task_id"] = "cleaning_service"
        ledger = {
            "schema_version": "skill_coverage_evidence_ledger_v1",
            "records": [record],
        }
        matrix = [
            {
                "service_task_id": "cleaning_service",
                "object_id": "toy",
                "applicability": "not_applicable",
                "mapping_basis": "default_not_applicable",
                "validation_state": "untested",
                "evidence": [],
            }
        ]

        with self.assertRaisesRegex(ValueError, "not applicable"):
            overlay_evidence_ledger(matrix, ledger)

    def test_coverage_summary_preserves_authoritative_metadata(self) -> None:
        record = self._validate()
        ledger = {
            "schema_version": "skill_coverage_evidence_ledger_v1",
            "records": [record],
        }
        matrix = overlay_evidence_ledger(
            [
                {
                    "service_task_id": "organization_storage",
                    "object_id": "toy",
                    "applicability": "direct",
                    "mapping_basis": "dataset_task:organizing",
                    "validation_state": "untested",
                    "evidence": [],
                }
            ],
            ledger,
        )
        existing = {
            "generator_version": "1.0.0",
            "source": {"path": "scope.xlsx", "sha256": "f" * 64},
            "dataset_task_count": 15,
            "source_relation_count": 138,
            "acceptance_target": 120,
        }

        summary = coverage_with_preserved_metadata(
            existing,
            [{"service_task_id": "organization_storage"}],
            [{"object_id": "toy"}],
            matrix,
        )

        self.assertEqual(existing["source"], summary["source"])
        self.assertEqual("1.0.0", summary["generator_version"])
        self.assertEqual(1, summary["validated_object_count"])
        self.assertEqual(119, summary["remaining_validation_gap"])


if __name__ == "__main__":
    unittest.main()
