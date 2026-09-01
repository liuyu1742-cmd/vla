from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tools.vla82_full_sim.annotations import OperationSpec
from tools.vla82_full_sim.assets import (
    DEFAULT_CATALOG_PATH,
    AssetResolutionError,
    AssetSpec,
    load_asset_catalog,
    resolve_asset,
    _xml_for_asset,
)


def fixture_operation_spec(selection_id: str = "VLA82-050", object_name: str = "订书机") -> OperationSpec:
    return OperationSpec(
        selection_id=selection_id,
        task="工作学习区服务",
        object_name=object_name,
        operation_text="摆放至书桌",
        source_kind="operation_json",
        source_path="C:/source/operation.json",
        phases=("place",),
        manipulated_objects=(object_name,),
        predicate_names=("place_completed",),
        source_sha256="a" * 64,
    )


def test_functional_proxy_can_never_be_final_exact_asset() -> None:
    mapping = {
        "selection_id": "VLA82-050",
        "mapping_mode": "functional_proxy",
        "object_group": "can",
    }

    asset = resolve_asset(fixture_operation_spec(), mapping)

    # Resolver returns an unvalidated source-backed candidate. It cannot claim
    # exactness until a source frame and collision model are actually checked.
    assert asset.exact_class is False
    assert asset.kind != "functional_proxy"
    assert asset.semantic_class == "订书机"
    assert asset.asset_path_or_group != "can"
    assert asset.collision_validated is False
    assert asset.visible_validated is False


def test_catalog_must_have_exactly_one_valid_same_class_asset_per_fixed_id(tmp_path: Path) -> None:
    catalog_path = tmp_path / "assets.json"
    catalog_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "selection_id": "VLA82-001",
                        "semantic_class": "杯子",
                        "kind": "custom_same_class",
                        "asset_path_or_group": "assets/custom/cup.xml",
                        "exact_class": True,
                        "collision_validated": True,
                        "visible_validated": True,
                        "affordances": ["grasp", "place"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetResolutionError, match="exactly 60"):
        load_asset_catalog(catalog_path)


def test_resolve_rejects_catalog_semantic_class_mismatch() -> None:
    mapping = {
        "selection_id": "VLA82-050",
        "mapping_mode": "native_object_task",
        "object_group": "stapler",
        "asset": {
            "selection_id": "VLA82-050",
            "semantic_class": "罐头",
            "kind": "robocasa_native",
            "asset_path_or_group": "stapler",
            "exact_class": True,
            "collision_validated": True,
            "visible_validated": True,
            "affordances": ["grasp"],
        },
    }

    with pytest.raises(AssetResolutionError, match="semantic class"):
        resolve_asset(fixture_operation_spec(), mapping)


def test_project_catalog_covers_the_fixed_60_exact_ids() -> None:
    catalog = load_asset_catalog(DEFAULT_CATALOG_PATH)

    assert len(catalog) == 60
    assert set(catalog) == {f"VLA82-{index:03d}" for index in range(1, 61)}
    assert all(item["exact_class"] is False for item in catalog.values())
    assert all(item["collision_validated"] is False for item in catalog.values())
    assert all(item["visible_validated"] is False for item in catalog.values())


def test_open_receptacle_xml_has_physical_collision_walls() -> None:
    asset = resolve_asset(
        fixture_operation_spec("VLA82-001", "垃圾桶"),
        {"selection_id": "VLA82-001", "mapping_mode": "functional_proxy"},
    )

    xml = _xml_for_asset(asset, "source_texture.png")

    assert 'name="collision_bottom"' in xml
    assert 'name="collision_front"' in xml


def test_custom_asset_xml_uses_selection_material_friction() -> None:
    asset = AssetSpec(
        selection_id="VLA82-006",
        semantic_class="rolling pin",
        kind="custom_same_class",
        asset_path_or_group="assets/vla82_source_textured/VLA82-006/model.xml",
        exact_class=True,
        collision_validated=True,
        visible_validated=True,
        affordances=("grasp", "place"),
        dimensions=(0.014, 0.014, 0.095),
        geometry="capsule",
    )

    xml = _xml_for_asset(asset, "source_texture.png")

    assert 'friction="0.82 0.008 0.0003"' in xml
    assert 'friction="0.95 0.3 0.1"' not in xml
