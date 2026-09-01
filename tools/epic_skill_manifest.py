"""Validate compact human-video skill manifests for robot-task transfer."""

from __future__ import annotations

from typing import Any


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate clip identity and the semantic fields of every skill phase."""
    if not manifest.get("clip_id"):
        raise ValueError("clip_id is required")
    phases = manifest.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases are required")
    for phase in phases:
        for key in ("verb", "object", "target"):
            if not phase.get(key):
                raise ValueError(f"phase {key} is required")
    return manifest
