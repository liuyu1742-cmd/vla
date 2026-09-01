from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class EvidenceLevel(str, Enum):
    SAMPLED_EPISODE = "sampled_episode"
    LOCAL_TASK_CODE = "local_task_code"
    OFFICIAL_TASK_TABLE = "official_task_table"
    OFFICIAL_REGISTRY = "official_registry"
    OFFICIAL_AGGREGATE = "official_aggregate"
    INFERRED_FROM_NAME = "inferred_from_name"


REQUIRED_TASK_FIELDS = {
    "dataset_id",
    "task_id",
    "task_name",
    "source_path_or_url",
    "source_kind",
    "evidence_level",
}


def validate_task_record(record: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_TASK_FIELDS - set(record))
    if missing:
        raise ValueError(f"task record missing source/provenance fields: {missing}")
    if record["evidence_level"] not in {level.value for level in EvidenceLevel}:
        raise ValueError(f"unknown evidence level: {record['evidence_level']}")
    return record


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _static_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "dict":
            result: dict[str, Any] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    return None
                result[keyword.arg] = _static_value(keyword.value)
            return result
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        bits: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            bits.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            bits.append(current.id)
        return ".".join(reversed(bits))
    return None


def _dict_call_values(call: ast.Call) -> dict[str, Any]:
    if not (isinstance(call.func, ast.Name) and call.func.id == "dict"):
        return {}
    return {
        keyword.arg: _static_value(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }


def _lang_assignments(class_node: ast.ClassDef) -> list[str]:
    instructions: list[str] = []
    for node in ast.walk(class_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = _static_value(node.value)
        if not isinstance(value, str) or not value.strip():
            continue
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            key = _static_value(target.slice)
            if key in {"lang", "language", "language_instruction"}:
                instructions.append(" ".join(value.split()))
                break
    return sorted(set(instructions))


def _object_groups(class_node: ast.ClassDef) -> list[str]:
    groups: set[str] = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Call):
            continue
        values = _dict_call_values(node)
        if "name" not in values or "obj_groups" not in values:
            continue
        obj_groups = values["obj_groups"]
        if isinstance(obj_groups, str):
            groups.add(obj_groups)
        elif isinstance(obj_groups, (list, tuple)):
            groups.update(str(item) for item in obj_groups if isinstance(item, str))
    return sorted(groups)


def _fixtures(class_node: ast.ClassDef) -> list[str]:
    fixtures: set[str] = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_fixture_ref"
        ):
            continue
        if node.args:
            alias = _static_value(node.args[0])
            if isinstance(alias, str):
                fixtures.add(alias)
    return sorted(fixtures)


def _native_family(relative_path: Path) -> str:
    parts = relative_path.parts
    if "composite" in parts:
        index = parts.index("composite")
        if index + 1 < len(parts):
            return parts[index + 1]
    if "single_stage" in parts:
        return "single_stage"
    if "atomic" in parts:
        index = parts.index("atomic")
        if index + 1 < len(parts) - 1:
            return parts[index + 1]
        return "atomic"
    return relative_path.parent.name or "unclassified"


def extract_robocasa_tasks(kitchen_root: Path) -> list[dict[str, Any]]:
    kitchen_root = Path(kitchen_root)
    records: list[dict[str, Any]] = []
    for source_path in sorted(kitchen_root.rglob("*.py")):
        if source_path.name in {"__init__.py", "kitchen.py"}:
            continue
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        relative = source_path.relative_to(kitchen_root)
        for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            if class_node.name.startswith("_"):
                continue
            method_names = {
                node.name
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if not (
                method_names
                & {"_get_obj_cfgs", "_check_success", "get_ep_meta", "__init__"}
            ):
                continue
            docstring = (ast.get_docstring(class_node) or "").strip()
            first_line = next(
                (line.strip() for line in docstring.splitlines() if line.strip()),
                class_node.name,
            )
            task_name = first_line.split(":", 1)[0].rstrip(".")
            native_family = _native_family(relative)
            record = {
                "dataset_id": "robocasa",
                "task_id": f"{native_family}::{_snake_case(class_node.name)}",
                "task_name": task_name,
                "task_family_native": native_family,
                "description": " ".join(docstring.split()),
                "instructions": _lang_assignments(class_node),
                "manipulated_objects": _object_groups(class_node),
                "target_objects": [],
                "fixtures": _fixtures(class_node),
                "source_path_or_url": str(source_path.resolve()),
                "source_kind": "local_task_code",
                "evidence_level": EvidenceLevel.LOCAL_TASK_CODE.value,
            }
            records.append(validate_task_record(record))
    records.sort(key=lambda row: (row["task_family_native"], row["task_id"]))
    return records


def extract_robocasa_object_registry(registry_path: Path) -> list[dict[str, Any]]:
    registry_path = Path(registry_path)
    tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    raw_categories: dict[str, Any] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "OBJ_CATEGORIES"
            for target in node.targets
        ):
            raw_categories = _static_value(node.value)
            break
    if not isinstance(raw_categories, dict):
        raise ValueError(f"OBJ_CATEGORIES not found or not static in {registry_path}")

    records: list[dict[str, Any]] = []
    for name, properties in sorted(raw_categories.items()):
        if not isinstance(properties, dict):
            properties = {}
        cleaned: dict[str, Any] = {}
        for key, value in properties.items():
            if key in {"aigen", "objaverse"}:
                continue
            if isinstance(value, tuple):
                value = list(value)
            cleaned[str(key)] = value
        types = cleaned.get("types")
        if isinstance(types, str):
            cleaned["types"] = [types]
        records.append(
            {
                "canonical_name": str(name),
                "aliases": [str(name).replace("_", " ")],
                "datasets": ["robocasa"],
                "roles": ["registered_object_category"],
                "properties": cleaned,
                "source_path_or_url": str(registry_path.resolve()),
                "source_kind": "official_registry_local_copy",
                "evidence_level": EvidenceLevel.OFFICIAL_REGISTRY.value,
            }
        )
    return records


def select_task_families(
    tasks: Iterable[dict[str, Any]], limit: int = 15
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        family = str(task.get("task_family_native", "")).strip()
        if family and family not in {"atomic", "single_stage", "unclassified"}:
            grouped[family].append(str(task["task_id"]))
    ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    return [
        {
            "rank": index,
            "task_family": family,
            "task_count": len(task_ids),
            "task_ids": sorted(task_ids),
        }
        for index, (family, task_ids) in enumerate(ranked[:limit], start=1)
    ]


def rank_objects(
    registry: Iterable[dict[str, Any]],
    tasks: Iterable[dict[str, Any]],
    limit: int = 120,
) -> list[dict[str, Any]]:
    task_sets: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        for name in task.get("manipulated_objects", []):
            task_sets[str(name)].add(str(task["task_id"]))

    rows: list[dict[str, Any]] = []
    for item in registry:
        name = str(item["canonical_name"])
        properties = item.get("properties", {})
        object_types = properties.get("types", [])
        if isinstance(object_types, str):
            object_types = [object_types]
        eligible_groups = {name, *[str(value) for value in object_types]}
        eligible_groups -= {"all", "food", "receptacle", "container"}
        exact_tasks = set(task_sets.get(name, set()))
        group_tasks: set[str] = set()
        for group in eligible_groups - {name}:
            group_tasks.update(task_sets.get(group, set()))
        group_tasks -= exact_tasks
        eligible_tasks = exact_tasks | group_tasks
        rows.append(
            {
                **item,
                "task_count": len(eligible_tasks),
                "exact_task_count": len(exact_tasks),
                "group_eligible_task_count": len(group_tasks),
                "task_ids": sorted(eligible_tasks),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["exact_task_count"],
            -row["group_eligible_task_count"],
            row["canonical_name"],
        )
    )
    for index, row in enumerate(rows[:limit], start=1):
        row["rank"] = index
    return rows[:limit]


def _split_rows(info: dict[str, Any]) -> list[dict[str, Any]]:
    splits = info.get("splits", [])
    if isinstance(splits, list):
        return [row for row in splits if isinstance(row, dict)]
    if isinstance(splits, dict):
        return [
            {"name": name, **value}
            for name, value in splits.items()
            if isinstance(value, dict)
        ]
    return []


def _feature_key_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(_feature_key_paths(child, path))
    elif isinstance(value, list):
        for child in value[:1]:
            paths.extend(_feature_key_paths(child, prefix))
    return paths


def extract_bridge_summary(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    features_path = dataset_dir / "features.json"
    features = (
        json.loads(features_path.read_text(encoding="utf-8"))
        if features_path.exists()
        else {}
    )
    split_summary: dict[str, dict[str, int]] = {}
    total = 0
    for row in _split_rows(info):
        lengths = row.get("shardLengths", [])
        episodes = sum(int(value) for value in lengths)
        total += episodes
        split_summary[str(row.get("name", "unknown"))] = {
            "episodes": episodes,
            "shards": len(lengths),
            "bytes": int(row.get("numBytes", 0)),
        }
    feature_paths = _feature_key_paths(features)
    language_fields = [
        path for path in feature_paths if "language" in path.lower()
    ]
    return {
        "dataset_id": str(info.get("name", "bridge_dataset")),
        "version": str(info.get("version", "unknown")),
        "local_path": str(dataset_dir.resolve()),
        "total_episodes": total,
        "splits": split_summary,
        "language_fields": sorted(set(language_fields)),
        "feature_schema_present": bool(features),
        "source_kind": "local_tfds_schema",
        "evidence_level": EvidenceLevel.OFFICIAL_REGISTRY.value,
    }


def task_family_stats(tasks: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("task_family_native", "")) for row in tasks)
    counts.pop("", None)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

