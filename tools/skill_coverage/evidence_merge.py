"""Validate and deterministically overlay formal evidence on coverage registries."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.skill_coverage.contracts import Applicability, validate_matrix_entry
from tools.skill_coverage.matrix_builder import summarize_coverage


LEDGER_SCHEMA = "skill_coverage_evidence_ledger_v1"
BUNDLE_SCHEMA = "formal_skill_hybrid_acceptance_summary_v1"
REPORT_SCHEMA = "formal_skill_hybrid_model_evaluation_v1"
REPORT_SCOPE = "formal_skill_hybrid_model_evaluation"
NATIVE_PREDICATES = (
    "simulator_success",
    "placed_in_storage",
    "gripper_released",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
APPLICABLE = {
    Applicability.DIRECT.value,
    Applicability.DEVICE_CONTROL.value,
    Applicability.COMPOSITE_RESOURCE.value,
}


def _load_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _project_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path(project_root) / path


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _contract_index(
    contract: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(contract, Mapping):
        records = [
            dict(value, relation_key=key)
            if "relation_key" not in value
            else value
            for key, value in contract.items()
        ]
    else:
        records = list(contract)
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in records:
        key = item.get("relation_key")
        if not isinstance(key, str) or not key:
            raise ValueError("Task-2 contract relation_key is required")
        if key in indexed:
            raise ValueError(f"duplicate Task-2 relation: {key}")
        indexed[key] = item
    return indexed


def _all_native_predicates_true(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        value.get(name) is True for name in NATIVE_PREDICATES
    )


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if not SHA256.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _validate_report(
    report_path: Path,
    *,
    relation_key: str,
    seed: int,
    adapter_sha256: str,
    manifest_sha256: str,
    project_root: Path,
) -> dict[str, str]:
    resolved = _project_path(str(report_path), project_root)
    report = _load_json(resolved)
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("referenced report schema mismatch")
    if report.get("evidence_scope") != REPORT_SCOPE:
        raise ValueError("referenced report evidence scope mismatch")
    if report.get("relation_key") != relation_key:
        raise ValueError("referenced report relation mismatch")
    if int(report.get("seed", -1)) != seed:
        raise ValueError("referenced report seed mismatch")
    if report.get("split") != "held_out":
        raise ValueError("referenced report must use held_out split")
    if int(report.get("max_decision_steps", 0)) != 300:
        raise ValueError("referenced report must use the 300-decision protocol")
    if report.get("success") is not True:
        raise ValueError("referenced report is not successful")
    if not _all_native_predicates_true(report.get("success_predicates")):
        raise ValueError("referenced report native success predicates are incomplete")
    if report.get("ever_grasped") is not True or report.get("ever_inside") is not True:
        raise ValueError("referenced report lacks grasp or insertion evidence")
    if report.get("counts_toward_task2_coverage") is not True:
        raise ValueError("referenced report does not count toward Task-2 coverage")
    if report.get("counts_toward_autonomous_vla_acceptance") is not False:
        raise ValueError("hybrid report must not count as autonomous VLA acceptance")
    if report.get("adapter_sha256") != adapter_sha256:
        raise ValueError("referenced report adapter fingerprint mismatch")
    if report.get("manifest_sha256") != manifest_sha256:
        raise ValueError("referenced report manifest fingerprint mismatch")
    modes = report.get("execution_mode_counts")
    if not isinstance(modes, Mapping) or int(modes.get("final_contact_servo", 0)) < 1:
        raise ValueError("hybrid report must disclose final-contact servo use")
    recovery = sum(
        int(count)
        for mode, count in modes.items()
        if str(mode).startswith("state_aware_locate_recovery_")
    )
    if recovery < 1:
        raise ValueError("hybrid report must disclose state-aware locate recovery")
    return {
        "path": _portable_path(resolved, project_root),
        "sha256": _sha256(resolved),
    }


def validate_hybrid_acceptance_bundle(
    bundle_path: Path,
    contract: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Return one normalized L3 ledger record for a complete hybrid bundle."""
    project_root = Path(project_root).resolve()
    bundle_path = Path(bundle_path).resolve()
    bundle = _load_json(bundle_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unsupported acceptance bundle schema")
    relation_key = bundle.get("relation_key")
    indexed = _contract_index(contract)
    if not isinstance(relation_key, str) or relation_key not in indexed:
        raise ValueError(f"relation {relation_key!r} is not in Task-2 contract")
    if bundle.get("split") != "held_out":
        raise ValueError("acceptance bundle must use held_out split")

    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("acceptance bundle protocol is required")
    seeds = [int(seed) for seed in protocol.get("seeds", [])]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("acceptance bundle requires at least three unique held-out seeds")
    if int(protocol.get("max_decision_steps_per_seed", 0)) != 300:
        raise ValueError("acceptance bundle must use 300 decisions per seed")

    aggregate = bundle.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("acceptance aggregate is required")
    evaluated = int(aggregate.get("evaluated_seeds", 0))
    successful = int(aggregate.get("successful_seeds", 0))
    if successful != evaluated or evaluated != len(seeds) or successful < 3:
        raise ValueError("successful seeds must equal all evaluated held-out seeds")
    if float(aggregate.get("success_rate", 0.0)) != 1.0:
        raise ValueError("acceptance success rate must be 1.0")
    if aggregate.get("all_native_success_predicates_true") is not True:
        raise ValueError("all native success predicates must be true")
    if aggregate.get("counts_toward_task2_coverage") is not True:
        raise ValueError("acceptance bundle does not count toward Task-2 coverage")
    if aggregate.get("counts_toward_autonomous_vla_acceptance") is not False:
        raise ValueError("hybrid acceptance must not count as autonomous VLA")

    evaluation_mode = bundle.get("evaluation_mode")
    if not isinstance(evaluation_mode, Mapping):
        raise ValueError("acceptance evaluation mode is required")
    if evaluation_mode.get("kind") != "hybrid_closed_loop":
        raise ValueError("acceptance bundle must be hybrid_closed_loop")
    if evaluation_mode.get("pure_autonomous_vla") is not False:
        raise ValueError("hybrid bundle cannot make a pure autonomous VLA claim")
    if evaluation_mode.get("state_aware_locate_recovery") is not True:
        raise ValueError("hybrid bundle must disclose state-aware locate recovery")
    if evaluation_mode.get("final_contact_servo") is not True:
        raise ValueError("hybrid bundle must disclose final-contact servo")

    model = bundle.get("model")
    data_isolation = bundle.get("data_isolation")
    if not isinstance(model, Mapping) or not isinstance(data_isolation, Mapping):
        raise ValueError("model and data-isolation fingerprints are required")
    adapter_sha256 = _require_sha256(model.get("adapter_sha256"), "adapter_sha256")
    manifest_sha256 = _require_sha256(
        data_isolation.get("manifest_sha256"), "manifest_sha256"
    )

    results = bundle.get("results")
    if not isinstance(results, list) or len(results) != len(seeds):
        raise ValueError("acceptance results must match protocol seeds")
    results_by_seed: dict[int, Mapping[str, Any]] = {}
    report_records: list[dict[str, str]] = []
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("acceptance result must be an object")
        seed = int(result.get("seed", -1))
        if seed in results_by_seed:
            raise ValueError(f"duplicate acceptance result seed: {seed}")
        results_by_seed[seed] = result
        if result.get("success") is not True:
            raise ValueError(f"acceptance result seed {seed} is not successful")
        if not _all_native_predicates_true(result.get("success_predicates")):
            raise ValueError(f"acceptance result seed {seed} predicates are incomplete")
        if result.get("ever_grasped") is not True or result.get("ever_inside") is not True:
            raise ValueError(f"acceptance result seed {seed} lacks manipulation evidence")
        report_text = result.get("report")
        if not isinstance(report_text, str) or not report_text:
            raise ValueError(f"acceptance result seed {seed} report path is required")
        report_records.append(
            {
                "seed": seed,
                **_validate_report(
                    Path(report_text),
                    relation_key=relation_key,
                    seed=seed,
                    adapter_sha256=adapter_sha256,
                    manifest_sha256=manifest_sha256,
                    project_root=project_root,
                ),
            }
        )
    if set(results_by_seed) != set(seeds):
        raise ValueError("acceptance result seeds differ from protocol seeds")

    dataset_task_id, separator, object_id = relation_key.partition("::")
    if not separator or not dataset_task_id or not object_id:
        raise ValueError("relation_key must be dataset_task_id::object_id")
    contract_record = indexed[relation_key]
    if contract_record.get("task_id") not in (None, dataset_task_id):
        raise ValueError("Task-2 contract task_id differs from relation_key")
    if contract_record.get("object_id") not in (None, object_id):
        raise ValueError("Task-2 contract object_id differs from relation_key")
    primary = policy.get("primary_by_dataset_task")
    if not isinstance(primary, Mapping) or dataset_task_id not in primary:
        raise ValueError(f"dataset task {dataset_task_id!r} has no service-task mapping")
    service_task_id = str(primary[dataset_task_id])

    return {
        "relation_key": relation_key,
        "dataset_task_id": dataset_task_id,
        "service_task_id": service_task_id,
        "object_id": object_id,
        "validation_state": "passed",
        "evidence_level": "L3",
        "evaluation_mode": "hybrid_closed_loop",
        "pure_autonomous_vla": False,
        "bundle_path": _portable_path(bundle_path, project_root),
        "bundle_sha256": _sha256(bundle_path),
        "heldout_seeds": sorted(seeds),
        "successful_seeds": successful,
        "evaluated_seeds": evaluated,
        "adapter_sha256": adapter_sha256,
        "manifest_sha256": manifest_sha256,
        "reports": sorted(report_records, key=lambda item: int(item["seed"])),
    }


def merge_ledger_record(
    ledger: Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace one relation record and return a deterministically sorted ledger."""
    if ledger is None:
        ledger = {"schema_version": LEDGER_SCHEMA, "records": []}
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise ValueError("unsupported evidence ledger schema")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise ValueError("evidence ledger records must be a list")
    by_relation: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("evidence ledger record must be an object")
        key = item.get("relation_key")
        if not isinstance(key, str) or not key:
            raise ValueError("evidence ledger relation_key is required")
        if key in by_relation:
            raise ValueError(f"duplicate evidence ledger relation: {key}")
        by_relation[key] = dict(item)
    key = record.get("relation_key")
    if not isinstance(key, str) or not key:
        raise ValueError("new evidence record relation_key is required")
    by_relation[key] = dict(record)
    return {
        "schema_version": LEDGER_SCHEMA,
        "records": [by_relation[name] for name in sorted(by_relation)],
    }


def overlay_evidence_ledger(
    matrix: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Overlay passed ledger records onto a copy of the baseline matrix."""
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise ValueError("unsupported evidence ledger schema")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise ValueError("evidence ledger records must be a list")
    updated = [copy.deepcopy(dict(item)) for item in matrix]
    index = {
        (item["service_task_id"], item["object_id"]): item for item in updated
    }
    if len(index) != len(updated):
        raise ValueError("coverage matrix contains duplicate task/object pairs")
    for record in records:
        key = (record.get("service_task_id"), record.get("object_id"))
        if key not in index:
            raise ValueError(f"evidence target matrix row is missing: {key}")
        entry = index[key]
        if entry.get("applicability") not in APPLICABLE:
            raise ValueError(f"evidence target is not applicable: {key}")
        if record.get("validation_state") != "passed":
            raise ValueError("first-version evidence ledger only accepts passed records")
        entry["validation_state"] = "passed"
        entry["evidence"] = [
            {
                "path": record["bundle_path"],
                "sha256": record["bundle_sha256"],
                "relation_key": record["relation_key"],
                "evidence_level": record["evidence_level"],
                "evaluation_mode": record["evaluation_mode"],
                "pure_autonomous_vla": record["pure_autonomous_vla"],
                "heldout_seeds": list(record["heldout_seeds"]),
                "successful_seeds": int(record["successful_seeds"]),
                "evaluated_seeds": int(record["evaluated_seeds"]),
            }
        ]
        validate_matrix_entry(entry)
    return updated


def coverage_with_preserved_metadata(
    existing_summary: Mapping[str, Any],
    service_tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute coverage counters while preserving authoritative source metadata."""
    acceptance_target = int(existing_summary.get("acceptance_target", 120))
    summary = dict(existing_summary)
    summary.update(
        summarize_coverage(
            service_tasks,
            candidates,
            matrix,
            acceptance_target=acceptance_target,
        )
    )
    return summary


__all__ = [
    "BUNDLE_SCHEMA",
    "LEDGER_SCHEMA",
    "coverage_with_preserved_metadata",
    "merge_ledger_record",
    "overlay_evidence_ledger",
    "validate_hybrid_acceptance_bundle",
]
