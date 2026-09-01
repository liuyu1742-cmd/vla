"""Strict evidence records for the Section 4.6 simulation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal, Sequence


EvidenceKind = Literal["local_measured", "project_historical", "official_reported"]


@dataclass(frozen=True)
class EvidenceRecord:
    """One aggregate result with an explicit and auditable provenance."""

    name: str
    success_rate: float
    trials: int
    evidence_kind: EvidenceKind
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_libero_report(path: Path, suite: str) -> EvidenceRecord:
    """Load a project LIBERO JSON report without coercing its success values."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("report must contain at least one episode")
    if any(type(row.get("success")) is not bool for row in episodes):
        raise ValueError("report must contain concrete episode success booleans")
    successes = sum(row["success"] for row in episodes)
    return EvidenceRecord(
        name=suite,
        success_rate=100.0 * successes / len(episodes),
        trials=len(episodes),
        evidence_kind="local_measured",
        source=str(path),
    )


def mean_success(records: Sequence[EvidenceRecord]) -> float:
    """Return a trial-weighted percentage from evidence records."""
    if not records:
        raise ValueError("at least one evidence record is required")
    total_trials = sum(row.trials for row in records)
    if total_trials <= 0:
        raise ValueError("total trials must be positive")
    return sum(row.success_rate * row.trials for row in records) / total_trials

