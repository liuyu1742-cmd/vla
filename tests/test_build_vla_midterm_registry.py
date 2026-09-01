from __future__ import annotations

import json
from pathlib import Path

from tools.build_vla_midterm_registry import build_registry, write_registry


def test_registry_uses_only_vla_sources_and_has_exact_contract(tmp_path: Path) -> None:
    registry = build_registry()

    assert registry["source"]["benchmark"] == "RoboCasa365 pretrain atomic"
    assert registry["source"]["task_metadata"].startswith(
        "datasets/robocasa365_pretrain_atomic/"
    )
    assert "RobotProject" not in json.dumps(registry, ensure_ascii=False)
    assert registry["summary"]["task_count"] == 8
    assert registry["summary"]["object_count"] == 60
    assert registry["summary"]["unique_object_count"] == 60
    assert registry["summary"]["all_task_classes_found"] is True
    assert registry["summary"]["all_object_groups_registered"] is True
    assert registry["summary"]["all_objects_attested_by_task_language"] is True

    task_classes = {row["task_class"] for row in registry["relations"]}
    object_groups = {row["object_group"] for row in registry["relations"]}
    assert len(task_classes) == 8
    assert len(object_groups) == 60
    assert all(row["example_instruction"] for row in registry["relations"])

    outputs = write_registry(registry, tmp_path)
    assert outputs["json"].is_file()
    assert outputs["csv"].is_file()
    assert outputs["runs_csv"].is_file()

    runs = outputs["runs_csv"].read_text(encoding="utf-8-sig").splitlines()
    assert len(runs) == 1 + 60 * 10
