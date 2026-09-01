"""Snapshot and corruption tests for the evidence-backed coverage registry."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from tools.validate_skill_coverage import validate_registry_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = PROJECT_ROOT / "data" / "skill_coverage" / "generated"
EXPECTED_SOURCE_SHA256 = (
    "23e39dd2803f3d90725169a6095f354b227feb4c1f54f92310679e84732e9ae0"
)
APPLICABLE = {"direct", "device_control", "composite_resource"}


def _load(name: str):
    return json.loads((REGISTRY_DIR / name).read_text(encoding="utf-8"))


class AuthoritativeSkillCoverageSnapshotTests(unittest.TestCase):
    def test_generated_registry_has_the_locked_authoritative_shape(self) -> None:
        dataset_tasks = _load("dataset_tasks.json")
        candidates = _load("object_candidates.json")
        relations = _load("source_relations.json")
        matrix = _load("task_object_matrix.json")
        summary = _load("coverage_summary.json")

        self.assertEqual(15, len(dataset_tasks))
        self.assertEqual(123, len(candidates))
        self.assertEqual(138, len(relations))
        self.assertEqual(15 * 123, len(matrix))
        self.assertEqual(EXPECTED_SOURCE_SHA256, summary["source"]["sha256"])

        pair_counts = Counter(
            (entry["service_task_id"], entry["object_id"]) for entry in matrix
        )
        self.assertTrue(pair_counts)
        self.assertTrue(all(count == 1 for count in pair_counts.values()))
        object_counts = Counter(entry["object_id"] for entry in matrix)
        self.assertEqual({15}, set(object_counts.values()))

        applicable_by_task: dict[str, set[str]] = defaultdict(set)
        for entry in matrix:
            if entry["applicability"] in APPLICABLE:
                applicable_by_task[entry["service_task_id"]].add(entry["object_id"])
        self.assertEqual(15, len(applicable_by_task))
        self.assertTrue(all(applicable_by_task.values()))

        fire_entry = next(
            item
            for item in matrix
            if item["service_task_id"] == "security_entry"
            and item["object_id"] == "fire_alarm"
        )
        self.assertEqual("not_applicable", fire_entry["applicability"])

        passed = [
            item for item in matrix if item["validation_state"] == "passed"
        ]
        self.assertEqual(1, len(passed))
        self.assertEqual("organization_storage", passed[0]["service_task_id"])
        self.assertEqual("toy", passed[0]["object_id"])
        self.assertEqual("L3", passed[0]["evidence"][0]["evidence_level"])
        self.assertFalse(passed[0]["evidence"][0]["pure_autonomous_vla"])
        self.assertTrue(Path(passed[0]["evidence"][0]["path"]).is_absolute())
        self.assertTrue(Path(passed[0]["evidence"][0]["path"]).is_file())

        self.assertEqual(123, summary["semantically_mapped_object_count"])
        self.assertEqual(1, summary["validated_object_count"])
        self.assertEqual(119, summary["remaining_validation_gap"])

    def test_validator_accepts_structure_but_does_not_claim_project_acceptance(self) -> None:
        result = validate_registry_dir(REGISTRY_DIR)

        self.assertEqual("PASS", result["structural_validation"])
        self.assertEqual("NOT_READY", result["project_acceptance"])
        self.assertEqual(1, result["validated_object_count"])
        self.assertEqual(1, result["validated_service_task_count"])
        self.assertEqual(120, result["acceptance_target"])

    def test_validator_rejects_changed_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "registry"
            shutil.copytree(REGISTRY_DIR, copy)
            summary_path = copy / "coverage_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["source"]["sha256"] = "0" * 64
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source workbook hash mismatch"):
                validate_registry_dir(copy)

    def test_validator_rejects_passed_not_applicable_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "registry"
            shutil.copytree(REGISTRY_DIR, copy)
            matrix_path = copy / "task_object_matrix.json"
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            entry = next(
                item for item in matrix if item["applicability"] == "not_applicable"
            )
            entry["validation_state"] = "passed"
            matrix_path.write_text(
                json.dumps(matrix, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "cannot be passed"):
                validate_registry_dir(copy, verify_source=False)


if __name__ == "__main__":
    unittest.main()

