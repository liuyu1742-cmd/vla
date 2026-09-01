"""Audit locally available robot-demonstration sources for household VLA work.

The report deliberately separates *locally paired* demonstrations from metadata
catalogues and raw archives.  A row is eligible only when a real instruction,
RGB observation, and action record are all locally available (or have a
documented local conversion path).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HOME_EXCLUSIONS = (
    "supermarket",
    "outdoor",
    "garden",
    "patio",
    "garage",
    "car",
    "vehicle",
)
ELIGIBLE_CONVERSIONS = {"direct", "conversion_required"}


def classify_record(
    dataset: str,
    instruction: str,
    rgb_status: bool,
    action_status: bool,
    conversion_status: str,
    environment: str,
    **optional: Any,
) -> dict[str, Any]:
    """Return one normalized, JSON-serializable audit record."""
    record: dict[str, Any] = {
        "dataset": dataset,
        "instruction": instruction,
        "rgb_status": bool(rgb_status),
        "action_status": bool(action_status),
        "conversion_status": conversion_status,
        "environment": environment,
        "eligible": bool(
            instruction
            and rgb_status
            and action_status
            and environment == "home"
            and conversion_status in ELIGIBLE_CONVERSIONS
        ),
    }
    record.update({key: value for key, value in optional.items() if value is not None})
    return record


def is_non_home_instruction(instruction: str) -> bool:
    lowered = instruction.casefold()
    return any(term in lowered for term in HOME_EXCLUSIONS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_behavior(root: Path) -> list[dict[str, Any]]:
    subset = root / "datasets" / "behavior_1k" / "2025_challenge_household_subset"
    rows = read_jsonl(subset / "manifest.jsonl")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        key = (int(row.get("task_index", -1)), instruction)
        grouped.setdefault(key, []).append(row)

    records: list[dict[str, Any]] = []
    for (task_index, instruction), episodes in sorted(grouped.items()):
        paired = all(
            (subset / item["parquet_path"]).is_file()
            and (subset / item["head_rgb_path"]).is_file()
            for item in episodes
        )
        environment = "non_home" if is_non_home_instruction(instruction) else "home"
        records.append(
            classify_record(
                "BEHAVIOR-1K 2025 Challenge",
                instruction,
                paired,
                paired,
                "conversion_required",
                environment,
                task_index=task_index,
                episode_count=len(episodes),
                reason=(
                    "Paired local head-RGB MP4 and action/state Parquet; convert the "
                    "R1Pro action representation before OpenVLA training."
                    if paired
                    else "One or more local RGB/action pairs are missing."
                ),
            )
        )
    return records


def _read_parquet(path: Path):
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("pandas with Parquet support is required for RoboCasa audit") from error
    return pd.read_parquet(path)


def audit_robocasa(root: Path) -> list[dict[str, Any]]:
    metadata = root / "datasets" / "robocasa365_metadata" / "meta"
    episode_files = sorted((metadata / "episodes").glob("**/*.parquet"))
    counts: Counter[str] = Counter()
    for path in episode_files:
        frame = _read_parquet(path)
        if "tasks" not in frame.columns:
            continue
        for value in frame["tasks"]:
            values = value.tolist() if hasattr(value, "tolist") else value
            if isinstance(values, str):
                values = [values]
            for instruction in values or []:
                if str(instruction).strip():
                    counts[str(instruction).strip()] += 1

    has_schema = (metadata / "info.json").is_file() and (metadata / "tasks.parquet").is_file()
    return [
        classify_record(
            "RoboCasa-365",
            instruction,
            False,
            False,
            "metadata_only",
            "home",
            episode_count=count,
            reason=(
                "Local metadata exposes task text and RGB/action schema, but the RGB and "
                "action payload shards are not present locally."
                if has_schema
                else "RoboCasa metadata is not present locally."
            ),
        )
        for instruction, count in sorted(counts.items())
        if not is_non_home_instruction(instruction)
    ]


def audit_bridge(root: Path) -> list[dict[str, Any]:
    pass
