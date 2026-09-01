"""Project-owned RoboCasa assets for formal Task-2 relation environments."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOY_CATEGORY = "toy"
LOCAL_REGISTRY = "task2_local"
TOY_ASSET_ROOT = ROOT / "assets" / "robocasa" / "task2" / "toy"
TOY_MODEL_XML = TOY_ASSET_ROOT / "toy_block_0" / "model.xml"


def _floats(value: str | None, *, field: str) -> list[float]:
    if value is None:
        raise ValueError(f"toy MJCF {field} is missing")
    try:
        return [float(item) for item in value.split()]
    except ValueError as error:
        raise ValueError(f"toy MJCF {field} is not numeric") from error


def inspect_toy_asset(path: Path = TOY_MODEL_XML) -> dict[str, object]:
    """Validate the lightweight asset without importing MuJoCo or RoboCasa."""
    path = Path(path).resolve()
    if TOY_ASSET_ROOT.resolve() not in path.parents:
        raise ValueError("toy MJCF must remain under the project Task-2 asset root")
    root = ET.parse(path).getroot()
    object_body = root.find("./worldbody/body/body[@name='object']")
    if object_body is None:
        raise ValueError("toy MJCF object body is missing")
    visual = object_body.find("./geom[@name='toy_visual']")
    collision = object_body.find("./geom[@name='toy_collision']")
    bbox = object_body.find("./geom[@name='reg_bbox']")
    if bbox is None:
        raise ValueError("toy MJCF reg_bbox is missing")
    half_size = _floats(bbox.get("size"), field="reg_bbox size")
    if len(half_size) != 3 or any(value <= 0 or value > 0.08 for value in half_size):
        raise ValueError("toy MJCF reg_bbox must have three stable positive half-sizes")
    if visual is None or visual.get("class") != "visual":
        raise ValueError("toy MJCF visual geom is missing")
    if collision is None or collision.get("class") != "collision":
        raise ValueError("toy MJCF collision geom is missing")
    mass = _floats(collision.get("mass"), field="collision mass")
    friction = _floats(collision.get("friction"), field="collision friction")
    if len(mass) != 1 or mass[0] <= 0:
        raise ValueError("toy MJCF collision mass must be positive")
    if len(friction) != 3 or any(value <= 0 for value in friction):
        raise ValueError("toy MJCF collision friction must contain three values")
    return {
        "model": root.get("model"),
        "path": str(path),
        "half_size": half_size,
        "full_size": [2 * value for value in half_size],
        "has_visual_geom": True,
        "has_collision_geom": True,
        "mass": mass[0],
        "friction": friction,
    }


def _is_our_category(category: object) -> bool:
    paths = getattr(category, "mjcf_paths", None)
    if not isinstance(paths, list):
        return False
    expected = str(TOY_MODEL_XML.resolve()).casefold()
    return any(str(Path(path).resolve()).casefold() == expected for path in paths)


def register_local_toy_assets(
    categories: MutableMapping[str, Any] | None = None,
    groups: MutableMapping[str, list[str]] | None = None,
    *,
    factory: Callable[..., Any] | None = None,
) -> object:
    """Idempotently register the project toy without replacing upstream data."""
    inspect_toy_asset()
    if categories is None or groups is None or factory is None:
        from robocasa.models.objects.kitchen_object_utils import ObjCat
        from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS

        categories = OBJ_CATEGORIES if categories is None else categories
        groups = OBJ_GROUPS if groups is None else groups
        factory = ObjCat if factory is None else factory

    existing = categories.get(TOY_CATEGORY)
    if existing is not None:
        if not isinstance(existing, MutableMapping):
            raise ValueError("refuse to overwrite foreign RoboCasa toy category")
        registered = existing.get(LOCAL_REGISTRY)
        if registered is not None and _is_our_category(registered):
            if groups.get(TOY_CATEGORY) != [TOY_CATEGORY]:
                raise ValueError("refuse to overwrite foreign RoboCasa toy group")
            return registered
        raise ValueError("refuse to overwrite foreign RoboCasa toy category")

    if TOY_CATEGORY in groups and groups[TOY_CATEGORY] != [TOY_CATEGORY]:
        raise ValueError("refuse to overwrite foreign RoboCasa toy group")
    category = factory(
        name=TOY_CATEGORY,
        types=("toy", "household_object"),
        model_folders=[str(TOY_ASSET_ROOT.resolve())],
        graspable=True,
        scale=1.0,
        density=100,
        friction=(0.9, 0.25, 0.08),
        reg_type=LOCAL_REGISTRY,
    )
    if not _is_our_category(category):
        raise ValueError("local toy registry did not discover the project MJCF")
    categories[TOY_CATEGORY] = {LOCAL_REGISTRY: category}
    groups[TOY_CATEGORY] = [TOY_CATEGORY]
    return category


__all__ = [
    "LOCAL_REGISTRY",
    "TOY_ASSET_ROOT",
    "TOY_CATEGORY",
    "TOY_MODEL_XML",
    "inspect_toy_asset",
    "register_local_toy_assets",
]
