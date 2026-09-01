"""Active project-owned RoboCasa asset registry with grasp-calibrated toy geometry."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any


_SOURCE = Path(__file__).resolve().parents[1] / "robocasa_assets.py"
_MODULE_NAME = "tools.skill_transfer._robocasa_assets_base"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load RoboCasa asset implementation from {_SOURCE}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, _BASE)
_SPEC.loader.exec_module(_BASE)

ROOT = _BASE.ROOT
TOY_CATEGORY = _BASE.TOY_CATEGORY
LOCAL_REGISTRY = _BASE.LOCAL_REGISTRY
TOY_ASSET_ROOT = _BASE.TOY_ASSET_ROOT
TOY_MODEL_XML = _BASE.TOY_MODEL_XML
ACTIVE_TOY_MODEL_XML = TOY_ASSET_ROOT / "toy_block_calibrated_0" / "model.xml"


def inspect_toy_asset(path: Path | None = None) -> dict[str, object]:
    """Inspect the active graspable asset by default; explicit legacy paths remain valid."""
    return _BASE.inspect_toy_asset(ACTIVE_TOY_MODEL_XML if path is None else Path(path))


def _is_project_category(category: object) -> bool:
    paths = getattr(category, "mjcf_paths", None)
    if not isinstance(paths, list):
        return False
    accepted = {
        str(TOY_MODEL_XML.resolve()).casefold(),
        str(ACTIVE_TOY_MODEL_XML.resolve()).casefold(),
    }
    return any(str(Path(path).resolve()).casefold() in accepted for path in paths)


def register_local_toy_assets(
    categories: MutableMapping[str, Any] | None = None,
    groups: MutableMapping[str, list[str]] | None = None,
    *,
    factory: Callable[..., Any] | None = None,
) -> object:
    """Register only the calibrated block while retaining legacy inspection support."""
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
        if registered is not None and _is_project_category(registered):
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
        exclude=[TOY_MODEL_XML.parent.name],
        graspable=True,
        scale=1.0,
        density=100,
        friction=(0.9, 0.25, 0.08),
        reg_type=LOCAL_REGISTRY,
    )
    if not _is_project_category(category):
        raise ValueError("local toy registry did not discover the calibrated project MJCF")
    categories[TOY_CATEGORY] = {LOCAL_REGISTRY: category}
    groups[TOY_CATEGORY] = [TOY_CATEGORY]
    return category


__all__ = [
    "ACTIVE_TOY_MODEL_XML",
    "LOCAL_REGISTRY",
    "TOY_ASSET_ROOT",
    "TOY_CATEGORY",
    "TOY_MODEL_XML",
    "inspect_toy_asset",
    "register_local_toy_assets",
]
