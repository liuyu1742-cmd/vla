"""Resolve exact RoboCasa assets and create loadable same-class digital twins."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


Shape = Literal["box", "cylinder", "sphere"]


@dataclass(frozen=True)
class DigitalTwinSpec:
    selection_id: str
    target_class: str
    shape: Shape
    size: tuple[float, float, float]
    affordances: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssetResolution:
    selection_id: str
    target_class: str
    kind: str
    exact_class: bool
    original_mapping_mode: str
    source: str | None
    original_object_group: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_name(value: str) -> str:
    return "".join(character if character.isascii() and character.isalnum() else "_" for character in value)


def _geom_attributes(spec: DigitalTwinSpec) -> str:
    x, y, z = (float(value) for value in spec.size)
    if min(x, y, z) <= 0:
        raise ValueError("digital twin dimensions must be positive")
    if spec.shape == "box":
        size = f"{x / 2:.6f} {y / 2:.6f} {z / 2:.6f}"
    elif spec.shape == "cylinder":
        size = f"{max(x, y) / 2:.6f} {z / 2:.6f}"
    elif spec.shape == "sphere":
        size = f"{max(x, y, z) / 2:.6f}"
    else:
        raise ValueError(f"unsupported digital twin shape: {spec.shape}")
    return f'type="{spec.shape}" size="{size}"'


def write_digital_twin(spec: DigitalTwinSpec, output_root: Path) -> Path:
    """Write a standalone MJCF that MuJoCo must load before it is accepted."""
    directory = output_root / spec.selection_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.xml"
    body_name = _safe_name(spec.selection_id)
    affordances = ",".join(spec.affordances)
    xml = (
        f'<!-- selection_id: {spec.selection_id}; target_class: {spec.target_class}; affordances: {affordances} -->\n'
        f'<mujoco model="{body_name}">\n'
        '  <compiler angle="radian"/>\n'
        '  <option gravity="0 0 -9.81"/>\n'
        '  <worldbody>\n'
        f'    <body name="{body_name}" pos="0 0 0.3">\n'
        '      <freejoint/>\n'
        f'      <geom name="{body_name}_geom" {_geom_attributes(spec)} mass="0.5" rgba="0.25 0.55 0.85 1"/>\n'
        '    </body>\n'
        '  </worldbody>\n'
        '</mujoco>\n'
    )
    temporary = path.with_suffix(".xml.tmp")
    temporary.write_text(xml, encoding="utf-8")
    temporary.replace(path)
    errors = validate_mjcf(path)
    if errors:
        raise ValueError("digital twin failed MuJoCo validation: " + ", ".join(errors))
    return path


def validate_mjcf(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return ["mjcf_missing"]
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(path))
        if model.ngeom < 1:
            errors.append("mjcf_has_no_geometry")
        if model.nbody < 2:
            errors.append("mjcf_has_no_object_body")
    except Exception as error:  # MuJoCo exposes several parser exception types.
        errors.append(f"mjcf_load_error:{type(error).__name__}:{error}")
    return errors


def resolve_asset(
    mapping: dict[str, Any], *, digital_twin: Path | None = None
) -> AssetResolution:
    selection_id = str(mapping["selection_id"])
    target_class = str(mapping["object"])
    mode = str(mapping.get("mapping_mode", ""))
    object_group = mapping.get("object_group")
    if digital_twin is not None:
        errors = validate_mjcf(digital_twin)
        text = digital_twin.read_text(encoding="utf-8") if digital_twin.is_file() else ""
        exact = not errors and target_class in text
        return AssetResolution(
            selection_id=selection_id,
            target_class=target_class,
            kind="digital_twin" if exact else "invalid_digital_twin",
            exact_class=exact,
            original_mapping_mode=mode,
            source=str(digital_twin.resolve()),
            original_object_group=str(object_group) if object_group is not None else None,
        )
    native = mode in {"native_object_task", "native_furniture_task"} and mapping.get(
        "proxy_disclosed"
    ) is not True
    return AssetResolution(
        selection_id=selection_id,
        target_class=target_class,
        kind="robocasa_native" if native else "unresolved_proxy",
        exact_class=native,
        original_mapping_mode=mode,
        source=str(object_group) if native and object_group is not None else None,
        original_object_group=str(object_group) if object_group is not None else None,
    )


def default_digital_twin_spec(mapping: dict[str, Any]) -> DigitalTwinSpec:
    """Choose conservative geometry from the recorded object affordance."""
    group = str(mapping.get("object_group") or "").lower()
    text = " ".join(
        str(mapping.get(key, "")).lower()
        for key in ("object", "operation_label", "manipulated_object_text")
    )
    if any(token in group or token in text for token in ("bottle", "can", "cup", "soap")):
        shape: Shape = "cylinder"
        size = (0.07, 0.07, 0.16)
    elif any(token in group or token in text for token in ("ball", "fruit", "apple")):
        shape = "sphere"
        size = (0.08, 0.08, 0.08)
    elif "垃圾桶" in text or "bin" in text:
        shape = "cylinder"
        size = (0.24, 0.24, 0.36)
    else:
        shape = "box"
        size = (0.10, 0.07, 0.05)
    task_class = str(mapping.get("task_class", ""))
    affordances = ["visual_target"]
    if "PickPlace" in task_class:
        affordances.extend(("graspable", "placeable"))
    if "Sink" in task_class or "trash" in text or "垃圾桶" in text:
        affordances.append("container_or_drop_target")
    return DigitalTwinSpec(
        selection_id=str(mapping["selection_id"]),
        target_class=str(mapping["object"]),
        shape=shape,
        size=size,
        affordances=tuple(affordances),
    )


def build_exact_asset_resolutions(
    *, mapping_plan: Path, registry: Path, output_root: Path
) -> list[AssetResolution]:
    """Resolve all fixed selections, replacing disclosed proxies with loadable twins."""
    plan = json.loads(mapping_plan.read_text(encoding="utf-8"))
    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    selected = {
        str(item["selection_id"]): item
        for item in registry_payload.get("selected_objects", [])
    }
    resolutions: list[AssetResolution] = []
    for original in plan.get("mappings", []):
        mapping = dict(original)
        source = selected.get(str(mapping.get("selection_id")))
        if source is None:
            raise ValueError(f"mapping is not in fixed sixty-object registry: {mapping}")
        mapping["object"] = source["object"]
        mapping["operation_label"] = source["operation_label"]
        resolution = resolve_asset(mapping)
        if not resolution.exact_class:
            spec = default_digital_twin_spec(mapping)
            twin = write_digital_twin(spec, output_root)
            resolution = resolve_asset(mapping, digital_twin=twin)
        resolutions.append(resolution)
    if len(resolutions) != 60:
        raise ValueError(f"expected 60 asset resolutions, got {len(resolutions)}")
    return resolutions
