"""Active evidence merge API with directly resolvable matrix evidence links."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence


_source = Path(__file__).resolve().parents[1] / "evidence_merge.py"
_spec = importlib.util.spec_from_file_location(
    "tools.skill_coverage._evidence_merge_legacy",
    _source,
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load evidence merge implementation: {_source}")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

BUNDLE_SCHEMA = _legacy.BUNDLE_SCHEMA
LEDGER_SCHEMA = _legacy.LEDGER_SCHEMA
coverage_with_preserved_metadata = _legacy.coverage_with_preserved_metadata
merge_ledger_record = _legacy.merge_ledger_record


def validate_hybrid_acceptance_bundle(
    bundle_path: Path,
    contract: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Validate the bundle and retain both portable and directly usable paths."""
    record = _legacy.validate_hybrid_acceptance_bundle(
        bundle_path,
        contract,
        policy,
        project_root,
    )
    record["bundle_absolute_path"] = str(Path(bundle_path).resolve())
    return record


def overlay_evidence_ledger(
    matrix: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Overlay ledger state while exposing absolute evidence paths to validators."""
    updated = _legacy.overlay_evidence_ledger(matrix, ledger)
    records = {
        record["relation_key"]: record for record in ledger.get("records", [])
    }
    for entry in updated:
        for evidence in entry.get("evidence", []):
            relation_key = evidence.get("relation_key")
            record = records.get(relation_key)
            if record is not None:
                evidence["path"] = record.get(
                    "bundle_absolute_path",
                    record["bundle_path"],
                )
    return updated


__all__ = [
    "BUNDLE_SCHEMA",
    "LEDGER_SCHEMA",
    "coverage_with_preserved_metadata",
    "merge_ledger_record",
    "overlay_evidence_ledger",
    "validate_hybrid_acceptance_bundle",
]
