"""Import a hash-locked Task-2 Excel/action contract into this workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tools.skill_coverage.task2_contract import (
    build_task2_contract,
    parse_action_call,
)
from tools.skill_coverage.xlsx_reader import read_sheet_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "data" / "skill_coverage" / "task2_contract_lock.json"
DEFAULT_OUTPUT = ROOT / "data" / "skill_coverage" / "generated"
GENERATOR_VERSION = "task2_contract_importer_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema_version") != "task2_contract_lock_v1":
        raise ValueError("unsupported Task-2 contract lock schema")
    for key in (
        "xlsx_path",
        "xlsx_sheet",
        "xlsx_sha256",
        "benchmark_path",
        "benchmark_sha256",
    ):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"lock field is required: {key}")
    for key in ("expected_tasks", "expected_unique_objects", "expected_relations"):
        if not isinstance(record.get(key), int) or record[key] < 1:
            raise ValueError(f"positive lock count is required: {key}")
    return record


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{label} hash mismatch: expected {expected.lower()}, found {actual.lower()}"
        )
    return actual


def build_locked_contract(
    lock_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read locked sources twice and return their exact joined contract."""
    lock_path = Path(lock_path).resolve()
    lock = load_lock(lock_path)
    xlsx = Path(lock["xlsx_path"]).resolve()
    benchmark = Path(lock["benchmark_path"]).resolve()
    xlsx_before = _require_hash(xlsx, lock["xlsx_sha256"], "xlsx")
    benchmark_before = _require_hash(
        benchmark, lock["benchmark_sha256"], "benchmark"
    )

    excel_rows = read_sheet_table(xlsx, lock["xlsx_sheet"])
    benchmark_records = json.loads(benchmark.read_text(encoding="utf-8"))
    if not isinstance(benchmark_records, list):
        raise ValueError("benchmark root must be a list")
    contract = build_task2_contract(excel_rows, benchmark_records)

    xlsx_after = sha256_file(xlsx)
    benchmark_after = sha256_file(benchmark)
    if xlsx_after != xlsx_before or benchmark_after != benchmark_before:
        raise RuntimeError("Task-2 source changed while it was being read")

    task_ids = {record["task_id"] for record in contract}
    object_ids = {record["object_id"] for record in contract}
    if len(task_ids) != lock["expected_tasks"]:
        raise ValueError(
            f"expected {lock['expected_tasks']} tasks, found {len(task_ids)}"
        )
    if len(object_ids) != lock["expected_unique_objects"]:
        raise ValueError(
            "expected "
            f"{lock['expected_unique_objects']} unique objects, found {len(object_ids)}"
        )
    if len(contract) != lock["expected_relations"]:
        raise ValueError(
            f"expected {lock['expected_relations']} relations, found {len(contract)}"
        )

    object_usage = Counter(record["object_id"] for record in contract)
    primitives = sorted(
        {
            parse_action_call(action)[0]
            for record in contract
            for action in record["canonical_actions"]
        }
    )
    metadata = {
        "schema_version": "task2_contract_manifest_v1",
        "generator_version": GENERATOR_VERSION,
        "lock_path": str(lock_path),
        "task_count": len(task_ids),
        "unique_object_count": len(object_ids),
        "relation_count": len(contract),
        "action_primitive_count": len(primitives),
        "action_primitives": primitives,
        "cross_task_objects": sorted(
            object_id for object_id, count in object_usage.items() if count > 1
        ),
        "sources": {
            "xlsx": {
                "path": str(xlsx),
                "sheet": lock["xlsx_sheet"],
                "sha256": xlsx_before,
                "size": xlsx.stat().st_size,
            },
            "benchmark": {
                "path": str(benchmark),
                "sha256": benchmark_before,
                "size": benchmark.stat().st_size,
            },
        },
    }
    return contract, metadata


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def import_task2_contract(lock_path: Path, output_dir: Path) -> dict[str, Any]:
    contract, metadata = build_locked_contract(lock_path)
    manifest = dict(metadata)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    output_dir = Path(output_dir).resolve()
    atomic_write_json(output_dir / "task2_skill_contract.json", contract)
    atomic_write_json(output_dir / "task2_contract_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = import_task2_contract(args.lock, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
