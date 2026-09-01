"""Build and validate a household-object registry without treating metadata as training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_vla_entry(entry: dict[str, Any]) -> list[str]:
    """Return missing required VLA evidence fields for a single object record."""
    missing: list[str] = []
    if not isinstance(entry.get("instruction"), str) or not entry["instruction"].strip():
        missing.append("missing_instruction")
    if not entry.get("rgb_files"):
        missing.append("missing_rgb")
    if not isinstance(entry.get("action_file"), str) or not entry["action_file"].strip():
        missing.append("missing_actions")
    if entry.get("source_episode") is None:
        missing.append("missing_source_episode")
    return missing


def validate_entry(entry: dict[str, Any]) -> list[str]:
    """Network-control rows are retained as interfaces, not mislabelled as VLA samples."""
    if entry.get("source_type") == "network_control":
        return []
    return validate_vla_entry(entry)


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = manifest["objects"]
    audit = []
    for record in objects:
        source_type = record.get("source_type") or record.get("archive_type") or "vla"
        check = dict(record)
        check["source_type"] = source_type
        audit.append(
            {
                "task_id": record.get("task_id"),
                "object": record.get("object"),
                "source_type": source_type,
                "missing": validate_entry(check),
            }
        )
    return {
        "manifest": str(manifest_path),
        "object_count": len(objects),
        "network_control_count": sum(x["source_type"] == "network_control" for x in audit),
        "vla_verified_count": sum(not x["missing"] and x["source_type"] != "network_control" for x in audit),
        "invalid_or_incomplete": [x for x in audit if x["missing"]],
        "records": audit,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit_manifest(args.manifest), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
