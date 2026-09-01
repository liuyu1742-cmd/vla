"""Evidence-preserving wrapper for the legacy authoritative registry builder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from tools.merge_skill_coverage_evidence import _atomic_group_write
from tools.skill_coverage.evidence_merge import (
    LEDGER_SCHEMA,
    coverage_with_preserved_metadata,
    overlay_evidence_ledger,
)


_source = Path(__file__).resolve().parents[1] / "build_authoritative_skill_coverage.py"
_spec = importlib.util.spec_from_file_location(
    "tools._build_authoritative_skill_coverage_legacy",
    _source,
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load authoritative builder implementation: {_source}")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

PROJECT_ROOT = _legacy.PROJECT_ROOT
DATA_ROOT = _legacy.DATA_ROOT
GENERATOR_VERSION = _legacy.GENERATOR_VERSION
atomic_write_json = _legacy.atomic_write_json
load_json = _legacy.load_json
sha256_file = _legacy.sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = _legacy.build_parser()
    parser.add_argument(
        "--evidence-ledger",
        type=Path,
        default=None,
        help=(
            "Replay an audited skill-coverage evidence ledger. The project ledger "
            "is replayed automatically only for the default output directory."
        ),
    )
    return parser


def _resolve_ledger(explicit: Path | None, output_dir: Path) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    default_output = (DATA_ROOT / "generated").resolve()
    default_ledger = (DATA_ROOT / "evidence_ledger.json").resolve()
    if output_dir.resolve() == default_output and default_ledger.is_file():
        return default_ledger
    return None


def _validate_ledger_artifacts(ledger: dict[str, Any], project_root: Path) -> None:
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise ValueError("unsupported evidence ledger schema")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise ValueError("evidence ledger records must be a list")
    for record in records:
        path = Path(str(record.get("bundle_path", "")))
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != record.get("bundle_sha256"):
            raise ValueError(
                f"bundle SHA-256 mismatch for {record.get('relation_key')}: "
                f"stored={record.get('bundle_sha256')} actual={actual}"
            )
        reports = record.get("reports", [])
        if not isinstance(reports, list):
            raise ValueError("evidence ledger reports must be a list")
        for report in reports:
            report_path = Path(str(report.get("path", "")))
            report_resolved = (
                report_path if report_path.is_absolute() else project_root / report_path
            )
            if not report_resolved.is_file():
                raise FileNotFoundError(report_resolved)
            report_hash = hashlib.sha256(report_resolved.read_bytes()).hexdigest()
            if report_hash != report.get("sha256"):
                raise ValueError(
                    f"report SHA-256 mismatch: {report_resolved}"
                )


def _legacy_arguments(args: argparse.Namespace, staging: Path) -> list[str]:
    return [
        "--xlsx",
        str(args.xlsx),
        "--sheet",
        str(args.sheet),
        "--output-dir",
        str(staging),
        "--expected-dataset-tasks",
        str(args.expected_dataset_tasks),
        "--expected-relations",
        str(args.expected_relations),
        "--expected-candidates",
        str(args.expected_candidates),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    ledger_path = _resolve_ledger(args.evidence_ledger, output_dir)
    ledger = None
    if ledger_path is not None:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        _validate_ledger_artifacts(ledger, PROJECT_ROOT)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".coverage-build-",
        dir=output_dir.parent,
    ) as directory:
        staging = Path(directory)
        exit_code = _legacy.main(_legacy_arguments(args, staging))
        outputs = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in staging.glob("*.json")
        }
        if ledger is not None:
            matrix = overlay_evidence_ledger(
                outputs["task_object_matrix.json"],
                ledger,
            )
            summary = coverage_with_preserved_metadata(
                outputs["coverage_summary.json"],
                load_json(DATA_ROOT / "service_tasks.json"),
                outputs["object_candidates.json"],
                matrix,
            )
            outputs["task_object_matrix.json"] = matrix
            outputs["coverage_summary.json"] = summary

        required = {
            "dataset_tasks.json",
            "object_candidates.json",
            "source_relations.json",
            "task_object_matrix.json",
            "coverage_summary.json",
        }
        if set(outputs) != required:
            raise ValueError(
                f"staged registry file set mismatch: {sorted(outputs)}"
            )
        _atomic_group_write(
            {output_dir / name: value for name, value in outputs.items()}
        )

    print(
        json.dumps(
            outputs["coverage_summary.json"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return exit_code


__all__ = [
    "DATA_ROOT",
    "GENERATOR_VERSION",
    "PROJECT_ROOT",
    "atomic_write_json",
    "build_parser",
    "load_json",
    "main",
    "sha256_file",
]
