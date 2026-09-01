"""End-to-end tests for the audited evidence-merge command."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.merge_skill_coverage_evidence import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_REGISTRY = PROJECT_ROOT / "data" / "skill_coverage" / "generated"
REAL_BUNDLE = (
    PROJECT_ROOT
    / "outputs"
    / "formal_skill_dagger_r3_hybrid_v3_eval"
    / "acceptance_summary.json"
)


class MergeSkillCoverageEvidenceCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root / "generated"
        shutil.copytree(REAL_REGISTRY, self.registry)
        self.ledger = self.root / "evidence_ledger.json"

    def _args(self, evidence: Path = REAL_BUNDLE) -> list[str]:
        return [
            "--evidence",
            str(evidence),
            "--registry-dir",
            str(self.registry),
            "--ledger",
            str(self.ledger),
            "--contract",
            str(REAL_REGISTRY / "task2_skill_contract.json"),
            "--policy",
            str(PROJECT_ROOT / "data" / "skill_coverage" / "applicability_policy.json"),
            "--service-tasks",
            str(PROJECT_ROOT / "data" / "skill_coverage" / "service_tasks.json"),
            "--project-root",
            str(PROJECT_ROOT),
        ]

    def _load(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_cli_writes_ledger_matrix_and_summary(self) -> None:
        exit_code = main(self._args())

        self.assertEqual(0, exit_code)
        ledger = self._load(self.ledger)
        matrix = self._load(self.registry / "task_object_matrix.json")
        summary = self._load(self.registry / "coverage_summary.json")
        passed = [item for item in matrix if item["validation_state"] == "passed"]
        self.assertEqual(1, len(ledger["records"]))
        self.assertEqual("organizing::toy", ledger["records"][0]["relation_key"])
        self.assertEqual(1, len(passed))
        self.assertEqual("organization_storage", passed[0]["service_task_id"])
        self.assertEqual("toy", passed[0]["object_id"])
        self.assertEqual(1, summary["validated_object_count"])
        self.assertEqual(119, summary["remaining_validation_gap"])

    def test_validation_failure_leaves_all_targets_byte_identical(self) -> None:
        bad_bundle = self.root / "bad_bundle.json"
        bundle = self._load(REAL_BUNDLE)
        bundle["model"]["adapter_sha256"] = "0" * 64
        bad_bundle.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        matrix_path = self.registry / "task_object_matrix.json"
        summary_path = self.registry / "coverage_summary.json"
        before_matrix = matrix_path.read_bytes()
        before_summary = summary_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "adapter fingerprint"):
            main(self._args(bad_bundle))

        self.assertFalse(self.ledger.exists())
        self.assertEqual(before_matrix, matrix_path.read_bytes())
        self.assertEqual(before_summary, summary_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
