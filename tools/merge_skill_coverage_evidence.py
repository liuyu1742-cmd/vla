"""Merge one audited formal acceptance bundle into skill-coverage evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.skill_coverage.evidence_merge import (
    LEDGER_SCHEMA,
    coverage_with_preserved_metadata,
    merge_ledger_record,
    overlay_evidence_ledger,
    validate_hybrid_acceptance_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "skill_coverage"
DEFAULT_REGISTRY = DATA_ROOT / "generated"


def _load_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_group_write(values: Mapping[Path, Any]) -> None:
    """Replace a small JSON file group and roll back if a replacement fails."""
    prepared = {Path(path): _json_bytes(value) for path, value in values.items()}
    originals = {
        path: path.read_bytes() if path.is_file() else None for path in prepared
    }
    temporaries: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for index, (path, data) in enumerate(prepared.items()):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.merge-{index}.tmp")
            temporary.write_bytes(data)
            json.loads(temporary.read_text(encoding="utf-8"))
            temporaries[path] = temporary
        for path, temporary in temporaries.items():
            temporary.replace(path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            original = originals[path]
            if original is None:
                if path.exists():
                    path.unlink()
            else:
                rollback = path.with_name(f".{path.name}.rollback.tmp")
                rollback.write_bytes(original)
                rollback.replace(path)
        raise
    finally:
        for temporary in temporaries.values():
            if temporary.exists():
                temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DATA_ROOT / "evidence_ledger.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_REGISTRY / "task2_skill_contract.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DATA_ROOT / "applicability_policy.json",
    )
    parser.add_argument(
        "--service-tasks",
        type=Path,
        default=DATA_ROOT / "service_tasks.json",
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    registry_dir = args.registry_dir.resolve()
    ledger_path = args.ledger.resolve()
    matrix_path = registry_dir / "task_object_matrix.json"
    summary_path = registry_dir / "coverage_summary.json"
    candidates_path = registry_dir / "object_candidates.json"

    contract = _load_json(args.contract.resolve())
    policy = _load_json(args.policy.resolve())
    service_tasks = _load_json(args.service_tasks.resolve())
    candidates = _load_json(candidates_path)
    matrix = _load_json(matrix_path)
    summary = _load_json(summary_path)
    ledger = (
        _load_json(ledger_path)
        if ledger_path.is_file()
        else {"schema_version": LEDGER_SCHEMA, "records": []}
    )

    record = validate_hybrid_acceptance_bundle(
        args.evidence.resolve(),
        contract,
        policy,
        project_root,
    )
    next_ledger = merge_ledger_record(ledger, record)
    next_matrix = overlay_evidence_ledger(matrix, next_ledger)
    next_summary = coverage_with_preserved_metadata(
        summary,
        service_tasks,
        candidates,
        next_matrix,
    )

    passed = [
        item for item in next_matrix if item.get("validation_state") == "passed"
    ]
    passed_tasks = {item["service_task_id"] for item in passed}
    if next_summary["validated_object_count"] < 1 or not passed_tasks:
        raise ValueError("evidence overlay produced no validated coverage")

    _atomic_group_write(
        {
            ledger_path: next_ledger,
            matrix_path: next_matrix,
            summary_path: next_summary,
        }
    )
    result = {
        "status": "PASS",
        "relation_key": record["relation_key"],
        "matrix_key": f"{record['service_task_id']}::{record['object_id']}",
        "evidence_level": record["evidence_level"],
        "evaluation_mode": record["evaluation_mode"],
        "pure_autonomous_vla": record["pure_autonomous_vla"],
        "validated_object_count": next_summary["validated_object_count"],
        "validated_service_task_count": len(passed_tasks),
        "remaining_validation_gap": next_summary["remaining_validation_gap"],
        "ledger": str(ledger_path),
        "registry_dir": str(registry_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
