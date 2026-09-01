from __future__ import annotations

from pathlib import Path

from tools.vla82_acceptance.assets import (
    DigitalTwinSpec,
    build_exact_asset_resolutions,
    resolve_asset,
    validate_mjcf,
    write_digital_twin,
)


ROOT = Path(__file__).resolve().parents[1]


def test_functional_proxy_is_not_exact_without_digital_twin():
    resolution = resolve_asset(
        {
            "selection_id": "VLA82-001",
            "object": "垃圾桶",
            "mapping_mode": "functional_proxy",
            "object_group": "can",
            "proxy_disclosed": True,
        }
    )

    assert resolution.exact_class is False
    assert resolution.kind == "unresolved_proxy"


def test_native_same_class_mapping_is_exact():
    resolution = resolve_asset(
        {
            "selection_id": "VLA82-002",
            "object": "海绵",
            "mapping_mode": "native_object_task",
            "object_group": "sponge",
            "proxy_disclosed": False,
        }
    )

    assert resolution.exact_class is True
    assert resolution.kind == "robocasa_native"


def test_digital_twin_uses_target_identity_and_loads_in_mujoco(tmp_path: Path):
    spec = DigitalTwinSpec(
        selection_id="VLA82-001",
        target_class="垃圾桶",
        shape="cylinder",
        size=(0.12, 0.12, 0.18),
        affordances=("container", "drop_target"),
    )

    path = write_digital_twin(spec, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "VLA82-001" in text
    assert "垃圾桶" in text
    assert validate_mjcf(path) == []


def test_valid_digital_twin_replaces_proxy_without_hiding_provenance(tmp_path: Path):
    spec = DigitalTwinSpec(
        selection_id="VLA82-001",
        target_class="垃圾桶",
        shape="cylinder",
        size=(0.12, 0.12, 0.18),
        affordances=("container", "drop_target"),
    )
    path = write_digital_twin(spec, tmp_path)

    resolution = resolve_asset(
        {
            "selection_id": "VLA82-001",
            "object": "垃圾桶",
            "mapping_mode": "functional_proxy",
            "object_group": "can",
            "proxy_disclosed": True,
        },
        digital_twin=path,
    )

    assert resolution.exact_class is True
    assert resolution.kind == "digital_twin"
    assert resolution.original_mapping_mode == "functional_proxy"


def test_all_sixty_mappings_resolve_to_native_or_loadable_digital_twin(
    tmp_path: Path,
):
    resolutions = build_exact_asset_resolutions(
        mapping_plan=ROOT / "outputs/midterm_testing_vla82/simulator_mapping_plan.json",
        registry=ROOT / "outputs/midterm_testing_vla82/vla82_midterm_registry.json",
        output_root=tmp_path,
    )

    assert len(resolutions) == 60
    assert all(item.exact_class for item in resolutions)
    assert sum(item.kind == "digital_twin" for item in resolutions) == 23
