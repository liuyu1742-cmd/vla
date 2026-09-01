"""Audit Chapter 5 object folders against the current Chapter 5 document."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from docx import Document


SKILL_TO_TASK = {
    "室内卫生清洁": "5.2.5",
    "烹饪与加热辅助": "5.2.6",
    "设备维护与工具使用": "5.2.7",
    "固定路线物品递送": "5.2.8",
    "物品递送": "5.2.8",
    "单件整理收纳": "5.2.9",
    "整理收纳": "5.2.9",
    "储物设施开合": "5.2.10",
    "室内安装与布置": "5.2.11",
    "固定路线餐具与容器转运": "5.2.12",
    "餐具与容器摆放": "5.2.12",
    "衣物鞋类与玄关归位": "5.2.13",
    "工作学习区服务": "5.2.14",
    "卫浴用品与个人卫生服务": "5.2.15",
}


def compare_object_sets(
    dataset_objects: dict[str, set[str]], document_objects: dict[str, set[str]]
) -> dict[str, dict[str, list[str]]]:
    """Return sorted mismatches between two task-to-object mappings."""
    document_only: dict[str, list[str]] = {}
    dataset_only: dict[str, list[str]] = {}
    for task in sorted(set(dataset_objects) | set(document_objects)):
        only_document = sorted(document_objects.get(task, set()) - dataset_objects.get(task, set()))
        only_dataset = sorted(dataset_objects.get(task, set()) - document_objects.get(task, set()))
        if only_document:
            document_only[task] = only_document
        if only_dataset:
            dataset_only[task] = only_dataset
    return {"document_only": document_only, "dataset_only": dataset_only}


def task_code_from_directory(name: str) -> str | None:
    match = re.match(r"chapter5_(5_2_\d+)_", name)
    return match.group(1).replace("_", ".") if match else None


def load_dataset_records(dataset_root: Path) -> tuple[dict[str, set[str]], list[dict]]:
    by_task: dict[str, set[str]] = defaultdict(set)
    records: list[dict] = []
    for task_dir in sorted(dataset_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("chapter5_5_2_"):
            continue
        task_code = task_code_from_directory(task_dir.name)
        if task_code not in SKILL_TO_TASK.values():
            continue
        for object_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            instruction_path = object_dir / "instruction.json"
            action_path = object_dir / "actions.parquet"
            videos = sorted(object_dir.rglob("*.mp4"))
            instruction = ""
            if instruction_path.exists():
                try:
                    instruction = json.loads(instruction_path.read_text(encoding="utf-8")).get("instruction", "")
                except json.JSONDecodeError:
                    instruction = ""
            evidence_ok = bool(instruction and action_path.exists() and videos)
            by_task[task_code].add(object_dir.name)
            records.append(
                {
                    "task_code": task_code,
                    "task_dir": task_dir.name,
                    "object": object_dir.name,
                    "instruction": instruction,
                    "instruction_file": instruction_path.exists(),
                    "rgb_video_count": len(videos),
                    "actions_parquet": action_path.exists(),
                    "triple_complete": evidence_ok,
                }
            )
    return dict(by_task), records


def load_document_objects(doc_path: Path) -> tuple[dict[str, set[str]], list[dict]]:
    document = Document(doc_path)
    table = next((candidate for candidate in document.tables if len(candidate.rows) > 1 and len(candidate.rows[0].cells) == 5), None)
    if table is None:
        raise ValueError("could not find the five-column VLA object audit table")
    by_task: dict[str, set[str]] = defaultdict(set)
    rows: list[dict] = []
    for row in table.rows[1:]:
        cells = [cell.text.strip() for cell in row.cells]
        if len(cells) < 3:
            continue
        skill, obj, operation = cells[:3]
        source_instruction = cells[3] if len(cells) > 3 else ""
        verification_status = cells[4] if len(cells) > 4 else ""
        task_code = SKILL_TO_TASK.get(skill)
        if task_code is None:
            continue
        by_task[task_code].add(obj)
        rows.append(
            {
                "task_code": task_code,
                "skill_name": skill,
                "object": obj,
                "document_operation": operation,
                "document_source_instruction": source_instruction,
                "document_verification_status": verification_status,
            }
        )
    return dict(by_task), rows


def build_report(dataset_root: Path, doc_path: Path) -> dict:
    dataset_objects, dataset_records = load_dataset_records(dataset_root)
    document_objects, document_rows = load_document_objects(doc_path)
    doc_lookup = {(row["task_code"], row["object"]): row for row in document_rows}
    joined = []
    for record in dataset_records:
        document_row = doc_lookup.get((record["task_code"], record["object"]))
        joined.append(
            {
                **record,
                "documented": document_row is not None,
                "document_operation": document_row["document_operation"] if document_row else "",
                "document_source_instruction": document_row["document_source_instruction"] if document_row else "",
                "document_instruction_matches_source": bool(
                    document_row and document_row["document_source_instruction"] == record["instruction"]
                ),
            }
        )
    mismatch = compare_object_sets(dataset_objects, document_objects)
    return {
        "summary": {
            "dataset_object_count": len(dataset_records),
            "document_object_count": sum(len(values) for values in document_objects.values()),
            "triple_complete_count": sum(record["triple_complete"] for record in dataset_records),
            "document_only_count": sum(len(values) for values in mismatch["document_only"].values()),
            "dataset_only_count": sum(len(values) for values in mismatch["dataset_only"].values()),
            "source_instruction_match_count": sum(
                row["document_instruction_matches_source"] for row in joined
            ),
        },
        "mismatch": mismatch,
        "records": joined,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.dataset_root, args.doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


