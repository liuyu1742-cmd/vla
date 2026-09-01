"""Classify execution reports and validate formal Task-2 skill evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SMOKE_SCOPE = "infrastructure_smoke"
FORMAL_SCOPE = "formal_skill_execution"


class FormalEvidenceError(ValueError):
    """Raised when a report cannot count as formal Task-2 execution evidence."""


def infrastructure_smoke_metadata() -> dict[str, object]:
    """Return immutable-by-convention metadata for legacy water-cup checks."""
    return {
        "evidence_scope": SMOKE_SCOPE,
        "relation_key": None,
        "counts_toward_task2_coverage": False,
    }


def as_infrastructure_smoke(report: Mapping[str, object]) -> dict[str, Any]:
    """Copy an arbitrary run report and force infrastructure-only semantics."""
    classified = dict(report)
    classified.update(infrastructure_smoke_metadata())
    return classified


def validate_formal_skill_evidence(
    report: Mapping[str, object],
    contract: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Return a formal report only if relation, actions and predicates all agree."""
    scope = report.get("evidence_scope")
    if scope != FORMAL_SCOPE:
        raise FormalEvidenceError(
            f"evidence_scope {scope!r} is not formal; {SMOKE_SCOPE!r} never counts"
        )
    key = report.get("relation_key")
    if not isinstance(key, str) or key not in contract:
        raise FormalEvidenceError(f"relation {key!r} is not in Task-2 contract")
    if report.get("counts_toward_task2_coverage") is not True:
        raise FormalEvidenceError("formal evidence must explicitly count toward coverage")
    if report.get("success") is not True:
        raise FormalEvidenceError("formal evidence requires relation-level success")

    expected_actions = contract[key].get("canonical_actions")
    actual_actions = report.get("canonical_actions")
    if not isinstance(expected_actions, list) or actual_actions != expected_actions:
        raise FormalEvidenceError("formal evidence canonical actions differ from contract")
    if report.get("canonical_action_progress") != len(expected_actions):
        raise FormalEvidenceError("formal evidence did not complete all canonical actions")

    predicates = report.get("success_predicates")
    if not isinstance(predicates, Mapping) or not predicates:
        raise FormalEvidenceError("formal evidence requires success predicates")
    if not all(value is True for value in predicates.values()):
        raise FormalEvidenceError("all formal success predicates must be true")
    return dict(report)


__all__ = [
    "FORMAL_SCOPE",
    "SMOKE_SCOPE",
    "FormalEvidenceError",
    "as_infrastructure_smoke",
    "infrastructure_smoke_metadata",
    "validate_formal_skill_evidence",
]
