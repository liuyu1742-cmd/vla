"""Rebuild and validate the shared Task-2 contract against its locked sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tools.import_task2_skill_contract import (
    DEFAULT_LOCK,
    DEFAULT_OUTPUT,
    build_locked_contract,
)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required Task-2 snapshot file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_task2_contract(
    registry_dir: Path,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    registry_dir = Path(registry_dir).resolve()
    contract = _load_json(registry_dir / "task2_skill_contract.json")
    manifest = _load_json(registry_dir / "task2_contract_manifest.json")
    expected_contract, expected_manifest = build_locked_contract(lock_path)
    if contract != expected_contract:
        raise ValueError("snapshot content mismatch with locked Task-2 sources")

    for key, expected_value in expected_manifest.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"snapshot manifest mismatch for {key}: "
                f"stored={manifest.get(key)!r}, expected={expected_value!r}"
            )
    generated_at = manifest.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise ValueError("snapshot manifest generated_at is missing")

    return {
        "contract_validation": "PASS",
        "task_count": expected_manifest["task_count"],
        "unique_object_count": expected_manifest["unique_object_count"],
        "relation_count": expected_manifest["relation_count"],
        "action_primitive_count": expected_manifest["action_primitive_count"],
        "cross_task_object_count": len(expected_manifest["cross_task_objects"]),
        "source_hashes_verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_task2_contract(args.registry_dir, args.lock)
    print(f"CONTRACT_VALIDATION: {result['contract_validation']}")
    print(f"TASKS: {result['task_count']}/{result['task_count']}")
    print(
        "OBJECTS: "
        f"{result['unique_object_count']}/{result['unique_object_count']}"
    )
    print(f"RELATIONS: {result['relation_count']}/{result['relation_count']}")
    print("SOURCE_HASHES: PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
