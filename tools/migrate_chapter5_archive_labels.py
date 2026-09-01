"""Safely rename Chapter 5 archive task directories and add section 5.2.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.archive_chapter5_objects import task_dir
from tools.robocasa_episode_archive_v2 import normalized_object_id

RENAMES = {
    "5.2.8": "\u7269\u54c1\u9012\u9001",
    "5.2.9": "\u6574\u7406\u6536\u7eb3",
    "5.2.10": "\u50a8\u7269\u67dc\u4e0e\u62bd\u5c49\u5f00\u5173",
    "5.2.11": "\u5bb6\u7535\u6309\u94ae\u4e0e\u65cb\u94ae\u64cd\u4f5c",
    "5.2.12": "\u9910\u5177\u4e0e\u5bb9\u5668\u6446\u653e",
}
HOME = "\u5bb6\u7535\u7efc\u5408\u7ba1\u7406"
HOME_OBJECTS = ["\u7167\u660e", "\u7a7a\u8c03", "\u7535\u89c6", "\u51b0\u7bb1", "\u5fae\u6ce2\u7089", "\u70e4\u9762\u5305\u673a", "\u7535\u70ed\u6c34\u58f6", "\u6405\u62cc\u673a"]


def section_prefix(section: str) -> str:
    return "chapter5_" + section.replace(".", "_") + "_"


def move_renamed_task_dirs(dataset_root: Path, dry_run: bool) -> list[dict[str, str]]:
    planned = []
    for section, new_name in RENAMES.items():
        matches = list(dataset_root.glob(section_prefix(section) + "*"))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one directory for {section}, found {matches}")
        source = matches[0]
        target = dataset_root / task_dir(section, new_name)
        if target.exists() and target != source:
            raise FileExistsError(target)
        planned.append({"section": section, "source": str(source), "target": str(target)})
        if not dry_run and target != source:
            source.rename(target)
    return planned


def add_home_task(manifest: dict, dataset_root: Path, dry_run: bool) -> None:
    if not any(row["section"] == "5.2.1" for row in manifest["tasks"]):
        manifest["tasks"].insert(0, {"section": "5.2.1", "task_name": HOME, "route": "\u7f51\u7edc\u63a7\u5236", "difficulty": "\u2605\u2605\u2606\u2606\u2606"})
        for name in HOME_OBJECTS:
            manifest["objects"].append({"section": "5.2.1", "task_name": HOME, "object_name": name, "archive_kind": "network_interface", "source_evidence": None, "conversion_status": "not_vla_training", "acceptance_boundary": "\u547d\u4ee4\u4e0b\u53d1\u3001\u72b6\u6001\u56de\u8bfb\u4e0e\u65e5\u5fd7\u9a8c\u6536"})
    for row in manifest["tasks"]:
        if row["section"] in RENAMES:
            row["task_name"] = RENAMES[row["section"]]
    for row in manifest["objects"]:
        if row["section"] in RENAMES:
            row["task_name"] = RENAMES[row["section"]]
    task_path = dataset_root / task_dir("5.2.1", HOME)
    if not dry_run:
        task_path.mkdir(parents=True, exist_ok=True)
        for name in HOME_OBJECTS:
            obj = task_path / normalized_object_id(name)
            obj.mkdir(exist_ok=False)
            (obj / "archive_status.json").write_text(json.dumps({"status": "network_interface", "not_a_vla_training_sample": True, "object_name": name, "section": "5.2.1", "reason": "requires app/interface state trace"}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    moves = move_renamed_task_dirs(args.dataset_root, args.dry_run)
    add_home_task(manifest, args.dataset_root, args.dry_run)
    if not args.dry_run:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dry_run": args.dry_run, "moves": moves, "tasks": len(manifest["tasks"]), "objects": len(manifest["objects"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
