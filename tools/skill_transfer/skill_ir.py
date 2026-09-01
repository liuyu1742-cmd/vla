"""Create and validate Task-2-aligned ``household_skill_ir_v2`` files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import SCHEMA_VERSION, validate_relation_record, validate_skill_ir_record


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    ROOT / "data" / "skill_coverage" / "generated" / "task2_skill_contract.json"
)
DEFAULT_MANIFEST = (
    ROOT / "data" / "skill_coverage" / "generated" / "task2_contract_manifest.json"
)


def load_contract(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Task-2 contract root must be a list")
    contract: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("Task-2 contract records must be objects")
        record = validate_relation_record(raw_record)
        key = str(record["relation_key"])
        if key in contract:
            raise ValueError(f"duplicate contract relation: {key}")
        contract[key] = record
    return contract


def build_skill_ir(
    key: str,
    contract: Mapping[str, Mapping[str, object]],
    *,
    contract_version: str,
) -> dict[str, Any]:
    if key not in contract:
        raise ValueError(f"unknown relation: {key}")
    expected = validate_relation_record(contract[key])
    skill_ir = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": contract_version,
        "relation_key": expected["relation_key"],
        "task_id": expected["task_id"],
        "task_name": expected["task_name"],
        "object_id": expected["object_id"],
        "object_name": expected["object_name"],
        "instruction": expected["instruction"],
        "target": expected["target"],
        "canonical_actions": list(expected["canonical_actions"]),
        "source_video": None,
        "perception_evidence": None,
        "parser_evidence": None,
        "execution_request": None,
        "execution_evidence": None,
    }
    return validate_skill_ir(skill_ir, contract)


def validate_skill_ir(
    skill_ir: Mapping[str, object],
    contract: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    key = skill_ir.get("relation_key")
    if not isinstance(key, str) or key not in contract:
        raise ValueError(f"unknown relation: {key}")
    return validate_skill_ir_record(skill_ir, contract[key])


def write_skill_ir(
    path: Path,
    skill_ir: Mapping[str, object],
    contract: Mapping[str, Mapping[str, object]],
) -> None:
    validated = validate_skill_ir(skill_ir, contract)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def contract_version_from_manifest(path: Path) -> str:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        xlsx = manifest["sources"]["xlsx"]["sha256"]
        benchmark = manifest["sources"]["benchmark"]["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Task-2 manifest source hashes are missing") from error
    if not all(isinstance(value, str) and len(value) == 64 for value in (xlsx, benchmark)):
        raise ValueError("Task-2 manifest source hashes are invalid")
    return f"task2_snapshot_{xlsx[:12]}_{benchmark[:12]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-key", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_contract(args.contract)
    skill_ir = build_skill_ir(
        args.relation_key,
        contract,
        contract_version=contract_version_from_manifest(args.manifest),
    )
    write_skill_ir(args.output, skill_ir, contract)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
