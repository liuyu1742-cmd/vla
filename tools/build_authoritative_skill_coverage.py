"""Build deterministic task/object coverage registries from a read-only XLSX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from tools.skill_coverage.matrix_builder import (
    build_task_object_matrix,
    summarize_coverage,
)
from tools.skill_coverage.registry_builder import build_source_registries
from tools.skill_coverage.xlsx_reader import read_sheet_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "skill_coverage"
GENERATOR_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--sheet", default="任务物体数据表")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "generated",
    )
    parser.add_argument("--expected-dataset-tasks", type=int, default=15)
    parser.add_argument("--expected-relations", type=int, default=138)
    parser.add_argument("--expected-candidates", type=int, default=123)
    return parser


def _require_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise ValueError(f"expected {expected} {label}, found {actual}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.xlsx.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir == source.parent:
        raise ValueError("output directory must not be the source workbook directory")

    source_hash_before = sha256_file(source)
    rows = read_sheet_table(source, args.sheet)
    registries = build_source_registries(rows)
    _require_count(
        len(registries["dataset_tasks"]),
        args.expected_dataset_tasks,
        "dataset tasks",
    )
    _require_count(
        len(registries["source_relations"]),
        args.expected_relations,
        "source relations",
    )
    _require_count(
        len(registries["object_candidates"]),
        args.expected_candidates,
        "candidate objects",
    )

    service_tasks = load_json(DATA_ROOT / "service_tasks.json")
    policy = load_json(DATA_ROOT / "applicability_policy.json")
    matrix = build_task_object_matrix(
        service_tasks,
        registries["object_candidates"],
        policy,
    )
    source_hash_after = sha256_file(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source workbook changed while it was being read")

    summary = summarize_coverage(
        service_tasks,
        registries["object_candidates"],
        matrix,
        acceptance_target=120,
    )
    summary.update(
        {
            "generator_version": GENERATOR_VERSION,
            "dataset_task_count": len(registries["dataset_tasks"]),
            "source_relation_count": len(registries["source_relations"]),
            "source": {
                "path": str(source),
                "sheet": args.sheet,
                "sha256": source_hash_before,
            },
        }
    )

    outputs = {
        "dataset_tasks.json": registries["dataset_tasks"],
        "object_candidates.json": registries["object_candidates"],
        "source_relations.json": registries["source_relations"],
        "task_object_matrix.json": matrix,
        "coverage_summary.json": summary,
    }
    for filename, value in outputs.items():
        atomic_write_json(output_dir / filename, value)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
