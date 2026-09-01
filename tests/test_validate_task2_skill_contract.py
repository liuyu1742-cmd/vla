"""Import and validation tests for the frozen Task-2 contract snapshot."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_build_authoritative_skill_coverage import write_registry_workbook


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_benchmark(path: Path) -> None:
    records = [
        {
            "id": "acceptance_appliance_remote",
            "relation_key": "appliance_management::remote_control",
            "instruction": "打开遥控器",
            "acceptance_task": "appliance_management",
            "legacy_task_id": None,
            "object": "remote_control",
            "target": "none",
            "actions": [
                "locate(remote_control)",
                "turn_on(remote_control)",
                "inspect(remote_control)",
            ],
        },
        {
            "id": "acceptance_fetch_remote",
            "relation_key": "object_fetching::remote_control",
            "instruction": "把遥控器拿给我",
            "acceptance_task": "object_fetching",
            "legacy_task_id": None,
            "object": "remote_control",
            "target": "none",
            "actions": [
                "locate(remote_control)",
                "grasp(remote_control)",
                "move(person)",
                "release(remote_control)",
            ],
        },
    ]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def write_lock(path: Path, xlsx: Path, benchmark: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "task2_contract_lock_v1",
                "xlsx_path": str(xlsx),
                "xlsx_sheet": "任务物体数据表",
                "xlsx_sha256": sha256_file(xlsx),
                "benchmark_path": str(benchmark),
                "benchmark_sha256": sha256_file(benchmark),
                "expected_tasks": 2,
                "expected_unique_objects": 1,
                "expected_relations": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class Task2SnapshotTests(unittest.TestCase):
    def make_sources(self, root: Path) -> tuple[Path, Path, Path]:
        xlsx = root / "scope.xlsx"
        benchmark = root / "benchmark.json"
        lock = root / "lock.json"
        write_registry_workbook(xlsx)
        write_benchmark(benchmark)
        write_lock(lock, xlsx, benchmark)
        return xlsx, benchmark, lock

    def test_import_writes_dual_source_manifest_and_contract(self) -> None:
        from tools.import_task2_skill_contract import import_task2_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xlsx, benchmark, lock = self.make_sources(root)
            output = root / "generated"
            xlsx_before = sha256_file(xlsx)
            benchmark_before = sha256_file(benchmark)

            manifest = import_task2_contract(lock, output)

            contract = json.loads(
                (output / "task2_skill_contract.json").read_text(encoding="utf-8")
            )
            stored_manifest = json.loads(
                (output / "task2_contract_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(contract))
            self.assertEqual(2, manifest["task_count"])
            self.assertEqual(1, manifest["unique_object_count"])
            self.assertEqual(2, manifest["relation_count"])
            self.assertEqual(manifest, stored_manifest)
            self.assertEqual(xlsx_before, sha256_file(xlsx))
            self.assertEqual(benchmark_before, sha256_file(benchmark))
            self.assertEqual(
                ["remote_control"], manifest["cross_task_objects"]
            )
            self.assertIn("grasp", manifest["action_primitives"])

    def test_import_rejects_locked_hash_mismatch(self) -> None:
        from tools.import_task2_skill_contract import import_task2_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, lock = self.make_sources(root)
            record = json.loads(lock.read_text(encoding="utf-8"))
            record["benchmark_sha256"] = "0" * 64
            lock.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "benchmark hash mismatch"):
                import_task2_contract(lock, root / "generated")

    def test_validator_accepts_fresh_snapshot(self) -> None:
        from tools.import_task2_skill_contract import import_task2_contract
        from tools.validate_task2_skill_contract import validate_task2_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, lock = self.make_sources(root)
            output = root / "generated"
            import_task2_contract(lock, output)

            result = validate_task2_contract(output, lock)

            self.assertEqual("PASS", result["contract_validation"])
            self.assertEqual(2, result["task_count"])
            self.assertTrue(result["source_hashes_verified"])

    def test_validator_rejects_tampered_contract(self) -> None:
        from tools.import_task2_skill_contract import import_task2_contract
        from tools.validate_task2_skill_contract import validate_task2_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, lock = self.make_sources(root)
            output = root / "generated"
            import_task2_contract(lock, output)
            contract_path = output / "task2_skill_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract[0]["canonical_actions"] = ["locate(remote_control)"]
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "snapshot content mismatch"):
                validate_task2_contract(output, lock)


if __name__ == "__main__":
    unittest.main()
