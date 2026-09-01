from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .core import (
    extract_bridge_summary,
    extract_robocasa_object_registry,
    extract_robocasa_tasks,
    rank_objects,
    select_task_families,
    task_family_stats,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_task_csv(path: Path, tasks: list[dict[str, Any]]) -> None:
    fields = [
        "dataset_id",
        "task_id",
        "task_name",
        "task_family_native",
        "instructions",
        "manipulated_objects",
        "fixtures",
        "evidence_level",
        "source_path_or_url",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    **{field: task.get(field, "") for field in fields},
                    "instructions": " | ".join(task.get("instructions", [])),
                    "manipulated_objects": " | ".join(
                        task.get("manipulated_objects", [])
                    ),
                    "fixtures": " | ".join(task.get("fixtures", [])),
                }
            )


def _write_object_csv(path: Path, objects: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "canonical_name",
        "task_count",
        "exact_task_count",
        "group_eligible_task_count",
        "task_ids",
        "roles",
        "properties",
        "evidence_level",
        "source_path_or_url",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in objects:
            writer.writerow(
                {
                    **{field: item.get(field, "") for field in fields},
                    "task_ids": " | ".join(item.get("task_ids", [])),
                    "roles": " | ".join(item.get("roles", [])),
                    "properties": json.dumps(
                        item.get("properties", {}), ensure_ascii=False, sort_keys=True
                    ),
                }
            )


def _build_report(
    bridge: dict[str, Any],
    tasks: list[dict[str, Any]],
    all_objects: list[dict[str, Any]],
    families: list[dict[str, Any]],
    top_objects: list[dict[str, Any]],
    source_catalog: list[dict[str, Any]],
) -> str:
    lines = [
        "# VLA 数据集元数据普查",
        "",
        "日期：2026-07-24",
        "",
        "## 结论",
        "",
        f"- 本地 BridgeData V2：{bridge['total_episodes']} 条 episode，"
        f"{sum(x['shards'] for x in bridge['splits'].values())} 个 TFRecord 分片。",
        f"- 本地 RoboCasa 1.0.1 源码：提取 {len(tasks)} 个任务类、"
        f"{len(all_objects)} 个正式对象类别。",
        f"- 数据驱动任务候选：取原生复合任务目录中任务数最多的 {len(families)} 类。",
        f"- 对象候选：按任务实际引用频次优先、正式对象注册表补齐，当前输出 {len(top_objects)} 个。",
        "- 分类不使用任务二、Excel 或参考文档第五章中的类别名称。",
        "",
        "## 15 类任务候选（数据集原生类别）",
        "",
        "| 排名 | 原生任务类别 | 任务数 | 示例任务 |",
        "|---:|---|---:|---|",
    ]
    for family in families:
        examples = "、".join(family["task_ids"][:5])
        lines.append(
            f"| {family['rank']} | `{family['task_family']}` | "
            f"{family['task_count']} | {examples} |"
        )
    lines.extend(
        [
            "",
            "## 120 物体候选",
            "",
            "| 排名 | 对象类别 | 精确任务引用 | 对象组可选 | 合计覆盖 | 证据 |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in top_objects:
        lines.append(
            f"| {item['rank']} | `{item['canonical_name']}` | "
            f"{item['exact_task_count']} | {item['group_eligible_task_count']} | "
            f"{item['task_count']} | `{item['evidence_level']}` |"
        )
    lines.extend(
        [
            "",
            "## BridgeData V2 本地结构",
            "",
            f"- 路径：`{bridge['local_path']}`",
            f"- 语言字段：{', '.join(f'`{x}`' for x in bridge['language_fields']) or '待样本级解析'}",
        ]
    )
    for split, values in bridge["splits"].items():
        lines.append(
            f"- `{split}`：{values['episodes']} episodes，"
            f"{values['shards']} shards，{values['bytes'] / 1e9:.2f} GB"
        )
    lines.extend(
        [
            "",
            "## 其他官方数据源登记",
            "",
            "| 数据集 | 官方任务/轨迹规模 | 本轮使用方式 | 任务级元数据状态 |",
            "|---|---|---|---|",
        ]
    )
    for source in source_catalog:
        lines.append(
            f"| {source['dataset_id']} | {source['official_scale']} | "
            f"{source['survey_role']} | {source['task_metadata_status']} |"
        )
    lines.extend(
        [
            "",
            "## 限制与下一步",
            "",
            "- RoboCasa 对象注册项是官方可生成对象类别；精确引用与对象组可选均来自源码静态配置，不等同于完整轨迹帧频。",
            "- BridgeData 已由隔离的轻量 TFRecord 读取器抽样真实语言指令，未安装完整 TensorFlow；详见 `bridge_language_sample_summary.json`。",
            "- 其他数据集先拉任务表/语言元数据，只有确认能补足当前覆盖缺口后才下载轨迹子集。",
            "- 训练验收必须继续按 held-out seed 和仿真成功谓词执行，不能用元数据覆盖数代替动作成功率。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey VLA dataset metadata")
    parser.add_argument(
        "--bridge-dir",
        type=Path,
        default=ROOT / "datasets" / "oxe" / "bridge_orig" / "1.0.0",
    )
    parser.add_argument(
        "--robocasa-root",
        type=Path,
        default=ROOT / "third_party" / "robocasa" / "robocasa",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=ROOT / "data" / "vla_metadata" / "official_sources.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "vla_metadata_survey",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs" / "VLA_METADATA_SURVEY_2026-07-24.md",
    )
    args = parser.parse_args()

    kitchen_root = args.robocasa_root / "environments" / "kitchen"
    registry_path = args.robocasa_root / "models" / "objects" / "kitchen_objects.py"
    tasks = extract_robocasa_tasks(kitchen_root)
    registered_objects = extract_robocasa_object_registry(registry_path)
    bridge = extract_bridge_summary(args.bridge_dir)
    families = select_task_families(tasks, limit=15)
    top_objects = rank_objects(registered_objects, tasks, limit=120)
    sources = json.loads(args.sources.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "unified_task_catalog.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks),
        encoding="utf-8",
    )
    _write_json(args.output_dir / "object_catalog.json", registered_objects)
    _write_json(args.output_dir / "task_family_candidates.json", families)
    _write_json(args.output_dir / "object_candidates_top120.json", top_objects)
    _write_json(
        args.output_dir / "dataset_summary.json",
        {
            "bridge": bridge,
            "robocasa": {
                "task_count": len(tasks),
                "object_registry_count": len(registered_objects),
                "native_family_counts": task_family_stats(tasks),
            },
            "external_sources": sources,
            "independence": {
                "uses_robotproject": False,
                "uses_excel_taxonomy": False,
                "uses_reference_doc_task_object_names": False,
            },
        },
    )
    _write_task_csv(args.output_dir / "task_catalog.csv", tasks)
    _write_object_csv(args.output_dir / "object_catalog.csv", top_objects)
    args.report.write_text(
        _build_report(
            bridge, tasks, registered_objects, families, top_objects, sources
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "registered_objects": len(registered_objects),
                "selected_task_families": len(families),
                "selected_objects": len(top_objects),
                "output_dir": str(args.output_dir.resolve()),
                "report": str(args.report.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

