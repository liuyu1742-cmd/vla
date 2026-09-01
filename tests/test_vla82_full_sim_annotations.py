from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from tools.vla82_full_sim.annotations import (
    OperationSpec,
    compile_all,
    compile_operation_spec,
)
from tools.vla82_full_sim.contracts import validate_fixed_scope


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs" / "midterm_testing_vla82" / "vla82_midterm_registry.json"


def _selection(data_path: Path) -> dict[str, str]:
    return {
        "selection_id": "VLA82-002",
        "task": "室内卫生清洁",
        "object": "海绵",
        "operation_label": "pick and place the sponge",
        "real_video_annotation": "抓取—擦拭—放回",
        "data_path": str(data_path),
    }


def test_operation_json_overrides_ledger_fields(tmp_path: Path) -> None:
    data = tmp_path / "object"
    data.mkdir()
    (data / "operation.json").write_text(
        json.dumps({"instruction": "抓取—擦拭—放回"}, ensure_ascii=False), encoding="utf-8"
    )

    spec = compile_operation_spec(_selection(data))

    assert spec.source_kind == "operation_json"
    assert spec.operation_text == "抓取—擦拭—放回"
    assert spec.phases == ("grasp", "wipe", "return")


def test_instruction_json_is_used_only_when_operation_json_lacks_instruction(tmp_path: Path) -> None:
    data = tmp_path / "object"
    data.mkdir()
    (data / "operation.json").write_text(json.dumps({"operation": ""}), encoding="utf-8")
    (data / "instruction.json").write_text(
        json.dumps({"payload": {"instruction": "Pick the sponge and place it."}}),
        encoding="utf-8",
    )

    spec = compile_operation_spec(_selection(data))

    assert spec.source_kind == "instruction_json"
    assert spec.operation_text == "Pick the sponge and place it."
    assert spec.phases == ("grasp", "place")


def test_operation_json_metadata_does_not_mask_a_usable_instruction(tmp_path: Path) -> None:
    data = tmp_path / "object"
    data.mkdir()
    (data / "operation.json").write_text(json.dumps({"object": "海绵"}), encoding="utf-8")
    (data / "instruction.json").write_text(
        json.dumps({"instruction": "Pick the sponge and place it."}), encoding="utf-8"
    )

    spec = compile_operation_spec(_selection(data))

    assert spec.source_kind == "instruction_json"
    assert spec.operation_text == "Pick the sponge and place it."


def test_english_pick_and_place_yield_grasp_and_place(tmp_path: Path) -> None:
    data = tmp_path / "object"
    data.mkdir()
    (data / "operation.json").write_text(
        json.dumps({"operation": "Pick the sponge, then place it in the bin."}), encoding="utf-8"
    )

    assert compile_operation_spec(_selection(data)).phases == ("grasp", "place")


def test_chinese_open_close_and_source_movement_annotations_compile_to_concrete_phases(tmp_path: Path) -> None:
    data = tmp_path / "object"
    data.mkdir()
    (data / "operation.json").write_text(json.dumps({"operation": "开门/关门"}, ensure_ascii=False), encoding="utf-8")
    assert compile_operation_spec(_selection(data)).phases == ("open", "close")

    (data / "operation.json").write_text(json.dumps({"operation": "抽屉→台面放置"}, ensure_ascii=False), encoding="utf-8")
    assert compile_operation_spec(_selection(data)).phases == ("place",)


def test_source_annotation_repeated_action_is_not_collapsed(tmp_path: Path) -> None:
    data = tmp_path / "object"
    data.mkdir()
    (data / "operation.json").write_text(json.dumps({"operation": "擦拭—擦拭"}, ensure_ascii=False), encoding="utf-8")
    assert compile_operation_spec(_selection(data)).phases == ("wipe", "wipe")


def test_all_authoritative_annotations_have_concrete_non_composite_phase_sequences() -> None:
    specs = compile_all(REGISTRY)
    assert all("composite" not in spec.phases for spec in specs)
    assert next(spec for spec in specs if spec.selection_id == "VLA82-008").phases == ("open", "close")
    assert next(spec for spec in specs if spec.selection_id == "VLA82-032").phases == ("open", "insert")


def test_compile_all_preserves_the_registry_order_and_sources() -> None:
    specs = compile_all(REGISTRY)

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert [spec.selection_id for spec in specs] == [
        item["selection_id"] for item in registry["selected_objects"]
    ]
    assert all(spec.source_sha256 for spec in specs)
    assert all(Path(spec.source_path).is_file() for spec in specs)


def test_fixed_scope_requires_exactly_the_registry_sixty() -> None:
    specs = compile_all(REGISTRY)

    assert validate_fixed_scope(specs) == []
    assert "selection_count:59" in validate_fixed_scope(specs[:-1])


def test_fixed_scope_rejects_duplicate_or_reordered_ids() -> None:
    specs = compile_all(REGISTRY)
    duplicate = [replace(specs[0], selection_id=specs[1].selection_id), *specs[1:]]

    duplicate_errors = validate_fixed_scope(duplicate)
    assert "selection_id_duplicate" in duplicate_errors
    assert "selection_order_mismatch" in duplicate_errors
    assert "selection_order_mismatch" in validate_fixed_scope([specs[1], specs[0], *specs[2:]])


def test_fixed_scope_rejects_empty_phases_and_missing_sources() -> None:
    specs = compile_all(REGISTRY)

    assert "phases_empty:VLA82-001" in validate_fixed_scope([replace(specs[0], phases=()), *specs[1:]])
    missing = replace(specs[0], source_path=str(ROOT / "does-not-exist.json"))
    assert "source_missing:VLA82-001" in validate_fixed_scope([missing, *specs[1:]])
