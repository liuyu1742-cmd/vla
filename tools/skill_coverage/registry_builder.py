"""Normalize workbook rows into lossless source and candidate registries."""

from __future__ import annotations

from typing import Any

from .contracts import validate_object_candidate


REQUIRED_COLUMNS = (
    "task_id",
    "task_name",
    "object",
    "物体中文名称",
    "images",
    "videos",
    "annotations",
    "metadata",
    "object_dir",
)


def as_int(value: str, field: str, source_row: int) -> int:
    try:
        return int(float(value or "0"))
    except ValueError as error:
        raise ValueError(
            f"row {source_row}: invalid {field}={value!r}"
        ) from error


def build_source_registries(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Preserve relations while deduplicating candidates by stable object ID."""
    dataset_tasks: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for source_row, row in enumerate(rows, start=2):
        missing = [key for key in REQUIRED_COLUMNS if key not in row]
        if missing:
            raise ValueError(f"row {source_row}: missing columns {missing}")

        task_id = row["task_id"].strip()
        task_name = row["task_name"].strip()
        object_id = row["object"].strip()
        display_name = row["物体中文名称"].strip()
        if not all((task_id, task_name, object_id, display_name)):
            raise ValueError(f"row {source_row}: blank identifier or name")

        pair = (task_id, object_id)
        if pair in seen_pairs:
            raise ValueError(f"row {source_row}: duplicate relation {pair}")
        seen_pairs.add(pair)

        counts = {
            key: as_int(row[key], key, source_row)
            for key in ("images", "videos", "annotations", "metadata")
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError(f"row {source_row}: negative media count")

        relations.append(
            {
                "source_row": source_row,
                "dataset_task_id": task_id,
                "dataset_task_name": task_name,
                "object_id": object_id,
                "display_name": display_name,
                **counts,
                "object_dir": row["object_dir"].strip(),
            }
        )

        task = dataset_tasks.setdefault(
            task_id,
            {
                "dataset_task_id": task_id,
                "display_name": task_name,
                "source_rows": [],
            },
        )
        if task["display_name"] != task_name:
            raise ValueError(
                f"row {source_row}: inconsistent task name for {task_id}"
            )
        task["source_rows"].append(source_row)

        candidate = candidates.setdefault(
            object_id,
            {
                "object_id": object_id,
                "display_name": display_name,
                "dataset_task_groups": [],
                "source_rows": [],
            },
        )
        if candidate["display_name"] != display_name:
            raise ValueError(
                f"row {source_row}: inconsistent object name for {object_id}"
            )
        candidate["dataset_task_groups"].append(task_id)
        candidate["source_rows"].append(source_row)

    for task in dataset_tasks.values():
        task["source_rows"].sort()
        task["relation_count"] = len(task["source_rows"])
    for candidate in candidates.values():
        candidate["dataset_task_groups"] = sorted(
            set(candidate["dataset_task_groups"])
        )
        candidate["source_rows"].sort()
        validate_object_candidate(candidate)

    return {
        "dataset_tasks": sorted(
            dataset_tasks.values(), key=lambda item: item["dataset_task_id"]
        ),
        "object_candidates": sorted(
            candidates.values(), key=lambda item: item["object_id"]
        ),
        "source_relations": sorted(
            relations, key=lambda item: item["source_row"]
        ),
    }
