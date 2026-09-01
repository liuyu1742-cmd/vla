"""Fixed-scope and anti-shortcut contracts for full VLA82 simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .annotations import DEFAULT_REGISTRY_PATH, OperationSpec, sha256_file
from tools.vla82_acceptance.demo_evidence import load_registry


FORBIDDEN_EXECUTION_KINDS = {
    "abstract_gantry",
    "direct_state_write",
    "scripted_full_trajectory",
}


def _authoritative_scope() -> tuple[list[str], set[str]]:
    registry = load_registry(DEFAULT_REGISTRY_PATH)
    selections = registry["selected_objects"]
    return (
        [str(selection["selection_id"]) for selection in selections],
        {str(selection["source_table"]) for selection in selections},
    )


def validate_fixed_scope(specs: Sequence[OperationSpec]) -> list[str]:
    """Return stable error codes if a compiled spec set is not the fixed 8x60 scope."""
    errors: list[str] = []
    expected_ids, expected_tables = _authoritative_scope()
    actual_ids = [spec.selection_id for spec in specs]
    if len(specs) != 60:
        errors.append(f"selection_count:{len(specs)}")
    if len(set(actual_ids)) != len(actual_ids):
        errors.append("selection_id_duplicate")
    if actual_ids != expected_ids:
        errors.append("selection_order_mismatch")
    actual_tables = {spec.source_table for spec in specs}
    if len(actual_tables) != 8:
        errors.append(f"source_table_count:{len(actual_tables)}")
    if actual_tables != expected_tables:
        errors.append("source_tables_mismatch")
    for spec in specs:
        if not spec.phases:
            errors.append(f"phases_empty:{spec.selection_id}")
        if not spec.predicate_names:
            errors.append(f"predicates_empty:{spec.selection_id}")
        source = Path(spec.source_path)
        if not source.is_file():
            errors.append(f"source_missing:{spec.selection_id}")
        elif not spec.source_sha256 or sha256_file(source) != spec.source_sha256:
            errors.append(f"source_hash_mismatch:{spec.selection_id}")
    return errors


def validate_no_shortcuts(report: Mapping[str, object], bundle: Path) -> list[str]:
    """Require complete-robot, learned execution evidence before accepting a rollout."""
    errors: list[str] = []
    if report.get("complete_robot_model") is not True:
        errors.append("complete_robot_missing")
    if report.get("execution_kind") in FORBIDDEN_EXECUTION_KINDS:
        errors.append("forbidden_execution_kind")
    if report.get("terminal_state_directly_written") is not False:
        errors.append("terminal_state_write_not_disproved")
    if not report.get("adapter_checkpoint_sha256"):
        errors.append("adapter_fingerprint_missing")
    if bundle.exists() and not (bundle / "bundle_evidence.json").is_file():
        errors.append("bundle_evidence_missing")
    return errors
