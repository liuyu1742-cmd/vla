"""Archive Chapter 5 objects while keeping network metadata out of VLA samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.robocasa_episode_archive_v2 import archive_episode, normalized_object_id


def task_dir(section: str, task_name: str) -> str:
    return f"chapter5_{section.replace('.', '_')}_{normalized_object_id(task_name)}"


def write_non_training_stub(destination: Path, row: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "archive_status.json").write_text(json.dumps({
        "status": row["archive_kind"], "not_a_vla_training_sample": True,
        "reason": "network control requires an app/interface state trace" if row["archive_kind"] == "network_interface" else "no matched instruction/RGB/action source episode",
        "object_name": row["object_name"], "section": row["section"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def archive(manifest: dict, source_root: Path, dataset_root: Path, report_path: Path) -> dict:
    by_section = {row["section"]: row for row in manifest["tasks"]}
    report = {"tasks": len(by_section), "objects": len(manifest["objects"]), "verified": [], "non_training": [], "errors": []}
    for section, task in by_section.items():
        (dataset_root / task_dir(section, task["task_name"])).mkdir(parents=True, exist_ok=True)
    for row in manifest["objects"]:
        destination = dataset_root / task_dir(row["section"], row["task_name"]) / normalized_object_id(row["object_name"])
        try:
            if row["archive_kind"] != "vla_training":
                write_non_training_stub(destination, row)
                report["non_training"].append(str(destination))
                continue
            evidence = row["source_evidence"]
            status = archive_episode(source_root, destination, int(evidence["episode_index"]), evidence["instruction"], row["object_name"])
            (destination / "operation.json").write_text(json.dumps({
                "section": row["section"], "task_name": row["task_name"], "object_name": row["object_name"],
                "object_type": row["object_type"], "operation": row["operation"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            report["verified"].append({"path": str(destination), **status})
        except Exception as exc:  # preserve evidence and continue with unrelated objects
            report["errors"].append({"object": row["object_name"], "section": row["section"], "error": repr(exc)})
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = archive(json.loads(args.manifest.read_text(encoding="utf-8")), args.source_root, args.dataset_root, args.report)
    print(json.dumps({"verified": len(report["verified"]), "non_training": len(report["non_training"]), "errors": len(report["errors"])}, ensure_ascii=False))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
