"""Contracts for truthful whole-home VLA task and object coverage."""

from __future__ import annotations

from typing import Any


ELIGIBLE_TRAINABILITY = frozenset({"ready", "generatable"})
ALL_TRAINABILITY = ELIGIBLE_TRAINABILITY | {"catalog_only"}
REQUIRED_ROW_FIELDS = frozenset(
    {
        "dataset_id",
        "room_zone",
        "task_id",
        "object_id",
        "trainability",
        "evidence",
    }
)


def validate_coverage_row(row: dict[str, Any]) -> None:
    """Validate the minimum provenance required for a coverage ledger row."""
    missing = sorted(REQUIRED_ROW_FIELDS - row.keys())
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if row["trainability"] not in ALL_TRAINABILITY:
        raise ValueError(f"invalid trainability: {row['trainability']}")
    if row["trainability"] in ELIGIBLE_TRAINABILITY and not str(row["evidence"]).strip():
        raise ValueError("eligible row requires evidence")


def eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows eligible for the headline training-coverage count."""
    for row in rows:
        validate_coverage_row(row)
    return [row for row in rows if row["trainability"] in ELIGIBLE_TRAINABILITY]
