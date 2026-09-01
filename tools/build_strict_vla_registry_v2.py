"""Strict audit helpers for legacy Chapter 5 archive records."""

from __future__ import annotations

from typing import Any


def validate_record(record: dict[str, Any]) -> list[str]:
    """Network-interface rows are control endpoints, not VLA training samples."""
    if record.get("archive_kind") == "network_interface":
        return []
    missing: list[str] = []
    if not record.get("instruction"):
        missing.append("missing_instruction")
    if not record.get("rgb_files"):
        missing.append("missing_rgb")
    if not record.get("action_file"):
        missing.append("missing_actions")
    if record.get("source_episode") is None:
        missing.append("missing_source_episode")
    return missing
