"""Validate generated authoritative skill-coverage registries and evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from tools.skill_coverage.contracts import (
    Applicability,
    ValidationState,
    validate_matrix_entry,
    validate_object_candidate,
)
from tools.skill_coverage.matrix_builder import summarize_coverage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "skill_coverage"
DEFAULT_REGISTRY_DIR = DATA_ROOT / "generated"
REQUIRED_FILES = (
    "dataset_tasks.json",
    "object_candidates.json",
    "source_relations.json",
    "task_object_matrix.json",
    "coverage_summary.json",
)
EXPECTED_SERVICE_TASKS = 15
EXPECTED_DATASET_TASKS = 15
EXPECTED_SOURCE_RELATIONS = 138
EXPECTED_CANDIDATES = 123


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required registry file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_unique_ids(records: list[dict[str, Any]], key: str) -> set[str]:
    values = [item.get(key) for item in records]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"every {key} must be non-empty text")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {key}: {duplicates}")
    return set(values)


def _validate_evidence(
    entry: dict[str, Any],
    project_root: Path,
) -> None:
    evidence = entry.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("matrix evidence must be a list")
    if entry["validation_state"] == ValidationState.PASSED.value and not evidence:
        raise ValueError("passed pair requires evidence")

    for item in evidence:
        if isinstance(item, str):
            evidence_path = item
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            evidence_path = item["path"]
        else:
            raise ValueError("evidence entries must be paths or records with a path")
        path = Path(evidence_path)
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.is_file():
            raise ValueError(f"evidence file is missing: {resolved}")


def _validate_source(summary: dict[str, Any]) -> None:
    source = summary.get("source")
    if not isinstance(source, dict):
        raise ValueError("coverage summary source metadata is missing")
    path_text = source.get("path")
    expected_hash = source.get("sha256")
    if not isinstance(path_text, str) or not isinstance(expected_hash, str):
        raise ValueError("coverage summary source path/hash is invalid")
    path = Path(path_text)
    if not path.is_file():
        raise ValueError(f"source workbook is missing: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            "source workbook hash mismatch: "
            f"expected {expected_hash.lower()}, found {actual_hash.lower()}"
        )


def validate_registry_dir(
    registry_dir: Path,
    *,
    verify_source: bool = True,
) -> dict[str, Any]:
    """Validate registry shape, cross-references, summary, and evidence links."""
    registry_dir = Path(registry_dir).resolve()
    loaded = {name: _load_json(registry_dir / name) for name in REQUIRED_FILES}
    dataset_tasks = loaded["dataset_tasks.json"]
    candidates = loaded["object_candidates.json"]
    relations = loaded["source_relations.json"]
    matrix = loaded["task_object_matrix.json"]
    summary = loaded["coverage_summary.json"]
    service_tasks = _load_json(DATA_ROOT / "service_tasks.json")

    for label, value in (
        ("dataset_tasks", dataset_tasks),
        ("object_candidates", candidates),
        ("source_relations", relations),
        ("task_object_matrix", matrix),
        ("service_tasks", service_tasks),
    ):
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a list")
    if not isinstance(summary, dict):
        raise ValueError("coverage_summary must be an object")

    if len(service_tasks) != EXPECTED_SERVICE_TASKS:
        raise ValueError(f"expected 15 service tasks, found {len(service_tasks)}")
    if len(dataset_tasks) != EXPECTED_DATASET_TASKS:
        raise ValueError(f"expected 15 dataset tasks, found {len(dataset_tasks)}")
    if len(relations) != EXPECTED_SOURCE_RELATIONS:
        raise ValueError(f"expected 138 source relations, found {len(relations)}")
    if len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError(f"expected 123 candidate objects, found {len(candidates)}")

    service_task_ids = _require_unique_ids(service_tasks, "service_task_id")
    dataset_task_ids = _require_unique_ids(dataset_tasks, "dataset_task_id")
    object_ids = _require_unique_ids(candidates, "object_id")

    for candidate in candidates:
        validate_object_candidate(candidate)
        unknown_groups = set(candidate["dataset_task_groups"]) - dataset_task_ids
        if unknown_groups:
            raise ValueError(f"candidate references unknown dataset tasks: {unknown_groups}")

    relation_pairs: set[tuple[str, str]] = set()
    for relation in relations:
        dataset_task_id = relation.get("dataset_task_id")
        object_id = relation.get("object_id")
        if dataset_task_id not in dataset_task_ids:
            raise ValueError(f"relation references unknown dataset task: {dataset_task_id}")
        if object_id not in object_ids:
            raise ValueError(f"relation references unknown object: {object_id}")
        pair = (dataset_task_id, object_id)
        if pair in relation_pairs:
            raise ValueError(f"duplicate source relation: {pair}")
        relation_pairs.add(pair)

    expected_pairs = {
        (service_task_id, object_id)
        for service_task_id in service_task_ids
        for object_id in object_ids
    }
    actual_pairs: set[tuple[str, str]] = set()
    applicable_by_task: dict[str, set[str]] = defaultdict(set)
    passed_by_task: dict[str, set[str]] = defaultdict(set)
    project_root = registry_dir.parents[2] if len(registry_dir.parents) > 2 else PROJECT_ROOT
    applicable_values = {
        Applicability.DIRECT.value,
        Applicability.DEVICE_CONTROL.value,
        Applicability.COMPOSITE_RESOURCE.value,
    }

    for entry in matrix:
        validate_matrix_entry(entry)
        service_task_id = entry["service_task_id"]
        object_id = entry["object_id"]
        if service_task_id not in service_task_ids:
            raise ValueError(f"matrix references unknown service task: {service_task_id}")
        if object_id not in object_ids:
            raise ValueError(f"matrix references unknown object: {object_id}")
        pair = (service_task_id, object_id)
        if pair in actual_pairs:
            raise ValueError(f"duplicate matrix pair: {pair}")
        actual_pairs.add(pair)
        _validate_evidence(entry, project_root)
        if entry["applicability"] in applicable_values:
            applicable_by_task[service_task_id].add(object_id)
            if entry["validation_state"] == ValidationState.PASSED.value:
                passed_by_task[service_task_id].add(object_id)

    if actual_pairs != expected_pairs:
        missing = len(expected_pairs - actual_pairs)
        extra = len(actual_pairs - expected_pairs)
        raise ValueError(f"matrix is not complete: missing={missing}, extra={extra}")
    tasks_without_applicable_objects = sorted(service_task_ids - set(applicable_by_task))
    if tasks_without_applicable_objects:
        raise ValueError(
            "service tasks without applicable objects: "
            f"{tasks_without_applicable_objects}"
        )

    recomputed = summarize_coverage(
        service_tasks,
        candidates,
        matrix,
        acceptance_target=int(summary.get("acceptance_target", 120)),
    )
    recomputed.update(
        {
            "dataset_task_count": len(dataset_tasks),
            "source_relation_count": len(relations),
        }
    )
    for key, value in recomputed.items():
        if summary.get(key) != value:
            raise ValueError(
                f"coverage summary mismatch for {key}: "
                f"stored={summary.get(key)!r}, recomputed={value!r}"
            )

    if verify_source:
        _validate_source(summary)

    acceptance_target = recomputed["acceptance_target"]
    validated_objects = recomputed["validated_object_count"]
    validated_tasks = len(passed_by_task)
    project_ready = (
        validated_objects >= acceptance_target
        and validated_tasks == EXPECTED_SERVICE_TASKS
    )
    return {
        "structural_validation": "PASS",
        "project_acceptance": "READY" if project_ready else "NOT_READY",
        "validated_object_count": validated_objects,
        "acceptance_target": acceptance_target,
        "remaining_validation_gap": recomputed["remaining_validation_gap"],
        "validated_service_task_count": validated_tasks,
        "service_task_count": len(service_tasks),
        "candidate_object_count": len(candidates),
        "matrix_entry_count": len(matrix),
        "source_hash_verified": verify_source,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=DEFAULT_REGISTRY_DIR,
    )
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="Skip the read-only source workbook hash check.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_registry_dir(
        args.registry_dir,
        verify_source=not args.skip_source_hash,
    )
    print(f"STRUCTURAL_VALIDATION: {result['structural_validation']}")
    print(
        "PROJECT_ACCEPTANCE: "
        f"{result['project_acceptance']} "
        f"validated_objects={result['validated_object_count']} "
        f"target={result['acceptance_target']} "
        f"validated_tasks={result['validated_service_task_count']}/"
        f"{result['service_task_count']}"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
