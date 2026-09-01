"""Exact-class VLA82 asset resolution and source-preserving custom MJCF assets.

The historical mapping plan contains convenience ``functional_proxy`` entries.
Those are deliberately never returned from this module: unsupported semantic
classes are represented by their own source-textured MJCF asset instead.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tools.vla82_full_sim.annotations import OperationSpec
from tools.vla82_full_sim.friction import format_friction, profile_for


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "configs" / "vla82_full_simulation" / "object_assets.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "configs" / "vla82_full_simulation" / "authoritative_asset_manifest.json"
CUSTOM_ASSET_ROOT = PROJECT_ROOT / "assets" / "vla82_source_textured"
VALID_KINDS = frozenset({"robocasa_native", "custom_same_class", "fixture_part"})
CURATED_APPEARANCE_SELECTIONS = frozenset({"VLA82-055"})


class AssetResolutionError(ValueError):
    """Raised when a mapping would conceal an inexact or proxy asset."""


@dataclass(frozen=True)
class AssetSpec:
    selection_id: str
    semantic_class: str
    kind: Literal["robocasa_native", "custom_same_class", "fixture_part"]
    asset_path_or_group: str
    exact_class: bool
    collision_validated: bool
    visible_validated: bool
    affordances: tuple[str, ...]
    source_path: str = ""
    dimensions: tuple[float, float, float] = (0.04, 0.04, 0.04)
    geometry: str = "box"
    evidence_path: str = ""
    asset_key: str = ""


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _safe_asset_name(selection_id: str) -> str:
    return "vla82_" + re.sub(r"[^a-zA-Z0-9_]", "_", selection_id).lower()


def _legacy_geometry_and_size(name: str) -> tuple[str, tuple[float, float, float]]:
    """Give every class a physical collision volume, not a proxy primitive label."""
    lowered = name.lower()
    if "垃圾桶" in name or "trash" in lowered or "waste" in lowered:
        return "open_receptacle", (0.115, 0.115, 0.150)
    if any(token in name for token in ("瓶", "杯", "壶", "罐", "液", "剂")) or any(
        token in lowered for token in ("bottle", "cup", "mug", "can", "teapot")
    ):
        return "cylinder", (0.032, 0.032, 0.085)
    if any(token in name for token in ("笔", "刷", "叉", "勺", "擀", "牙", "刀")) or any(
        token in lowered for token in ("pen", "brush", "fork", "spoon", "whisk", "rolling")
    ):
        return "capsule", (0.012, 0.012, 0.110)
    if any(token in name for token in ("盘", "碗", "托盘", "盖")) or any(
        token in lowered for token in ("plate", "bowl", "tray", "lid")
    ):
        return "cylinder", (0.090, 0.090, 0.018)
    if any(token in name for token in ("书", "夹", "键盘", "电脑", "箱", "盒", "纸")) or any(
        token in lowered for token in ("book", "folder", "keyboard", "laptop", "box")
    ):
        return "box", (0.110, 0.075, 0.016)
    return "box", (0.045, 0.035, 0.045)


# Explicit per-selection collision/appearance profiles.  A profile may share a
# geometry family, but it is never inferred from a loose object-name heuristic.
_PROFILE_ROWS = (
    ("open_receptacle", .115, .115, .150), ("sponge", .055, .035, .018), ("tool", .014, .014, .095),
    ("bottle", .025, .025, .085), ("tool", .012, .012, .105), ("tool", .020, .020, .130),
    ("fixture", .120, .120, .050), ("fixture", .180, .160, .120), ("tray", .110, .080, .012),
    ("tray", .120, .085, .015), ("dish", .075, .075, .030), ("bottle", .050, .050, .095),
    ("bottle", .025, .025, .055), ("cup", .040, .040, .055), ("pan", .105, .105, .035),
    ("book", .110, .075, .016), ("bottle", .032, .032, .100), ("box", .035, .025, .080),
    ("can", .032, .032, .055), ("cup", .045, .045, .055), ("box", .085, .060, .028),
    ("fixture", .130, .120, .100), ("box", .100, .075, .050), ("box", .140, .100, .075),
    ("box", .150, .110, .090), ("fixture", .160, .080, .015), ("fixture", .130, .090, .050),
    ("fixture", .180, .020, .220), ("fixture", .150, .020, .120), ("fixture", .120, .080, .050),
    ("fixture", .160, .020, .180), ("fixture", .160, .100, .020), ("fixture", .180, .020, .160),
    ("fixture", .150, .020, .120), ("fixture", .120, .020, .100), ("cup", .040, .040, .055),
    ("tool", .012, .012, .095), ("tool", .012, .012, .100), ("tool", .012, .012, .120),
    ("bottle", .050, .050, .115), ("tool", .012, .012, .110), ("tool", .016, .016, .115),
    ("dish", .100, .100, .040), ("dish", .100, .100, .015), ("box", .080, .030, .030),
    ("folder", .120, .085, .008), ("box", .100, .060, .035), ("book", .120, .085, .015),
    ("laptop", .150, .105, .014), ("stapler", .070, .025, .025), ("tool", .006, .006, .095),
    ("keyboard", .130, .055, .012), ("mouse", .035, .025, .020), ("fixture", .120, .070, .015),
    ("box", .090, .055, .025), ("soap", .040, .028, .015), ("bottle", .035, .025, .090),
    ("tool", .010, .010, .090), ("tube", .018, .018, .085), ("bottle", .035, .035, .100),
)
PROFILE_BY_SELECTION = {
    f"VLA82-{index:03d}": (row[0], (row[1], row[2], row[3])) for index, row in enumerate(_PROFILE_ROWS, start=1)
}
PROFILE_BY_SELECTION["VLA82-023"] = ("box", (.050, .040, .030))
PROFILE_BY_SELECTION["VLA82-048"] = ("book", (.050, .025, .008))


def _default_geometry_and_size(name: str, selection_id: str = "") -> tuple[str, tuple[float, float, float]]:
    """Return the recorded asset profile; no generic proxy fallback is allowed."""
    profile = PROFILE_BY_SELECTION.get(selection_id)
    if profile is None:
        # Auxiliary objects are only created from explicit source annotations.
        known = {
            "can of soda": ("can", (.032, .032, .055)), "sandal": ("sandal", (.105, .040, .025)),
            "\u6c7d\u6c34\u7f50": ("can", (.032, .032, .055)),
            "\u51c9\u978b": ("sandal", (.105, .040, .025)),
            "\u4e66\u7c4d": ("book", (.050, .035, .008)),
            "\u6587\u4ef6\u5939": ("folder", (.055, .040, .005)),
            "\u7b14\u76d2": ("box", (.100, .060, .035)),
            "\u94c5\u7b14": ("tool", (.006, .006, .095)), "\u7b14": ("tool", (.006, .006, .100)),
            "\u8ba2\u4e66\u673a": ("stapler", (.070, .025, .025)),
            "\u7b14\u8bb0\u672c\u7535\u8111": ("laptop", (.150, .105, .014)),
            "\u952e\u76d8": ("keyboard", (.065, .030, .006)), "\u9f20\u6807": ("mouse", (.018, .013, .010)),
            "\u7259\u5237": ("tool", (.010, .010, .090)), "\u7259\u818f": ("tube", (.018, .018, .085)),
            "\u7682\u6db2\u5668": ("bottle", (.035, .035, .100)),
            "book": ("book", (.050, .035, .008)), "paperback book": ("book", (.050, .035, .008)),
            "folder": ("folder", (.055, .040, .005)), "pencil case": ("box", (.050, .030, .018)),
            "pencil": ("tool", (.006, .006, .095)), "pen": ("tool", (.006, .006, .100)),
            "stapler": ("stapler", (.070, .025, .025)), "laptop": ("laptop", (.150, .105, .014)),
            "keyboard": ("keyboard", (.065, .030, .006)), "mouse": ("mouse", (.018, .013, .010)),
            "toothbrush": ("tool", (.010, .010, .090)), "toothpaste": ("tube", (.018, .018, .085)),
            "soap dispenser": ("bottle", (.035, .035, .100)),
        }
        profile = known.get(name.lower())
    if profile is None:
        raise AssetResolutionError(f"no object-specific profile for {selection_id or name}")
    return profile


def _asset_from_mapping(spec: OperationSpec, payload: Mapping[str, Any]) -> AssetSpec:
    semantic_class = _clean_text(payload.get("semantic_class", spec.object_name))
    if semantic_class != spec.object_name:
        raise AssetResolutionError(
            f"semantic class mismatch for {spec.selection_id}: {semantic_class!r} != {spec.object_name!r}"
        )
    kind = _clean_text(payload.get("kind", "custom_same_class"))
    if kind not in VALID_KINDS:
        raise AssetResolutionError(f"unsupported or proxy asset kind for {spec.selection_id}: {kind}")
    asset_path_or_group = _clean_text(payload.get("asset_path_or_group"))
    if not asset_path_or_group:
        raise AssetResolutionError(f"asset path is missing for {spec.selection_id}")
    dimensions_raw = payload.get("dimensions")
    geometry, defaults = _default_geometry_and_size(spec.object_name, spec.selection_id)
    dimensions = tuple(float(value) for value in (dimensions_raw or defaults))
    if len(dimensions) != 3 or any(value <= 0 for value in dimensions):
        raise AssetResolutionError(f"invalid dimensions for {spec.selection_id}")
    return AssetSpec(
        selection_id=spec.selection_id,
        semantic_class=semantic_class,
        kind=kind,  # type: ignore[arg-type]
        asset_path_or_group=asset_path_or_group,
        exact_class=bool(payload.get("exact_class", False)),
        collision_validated=bool(payload.get("collision_validated", False)),
        visible_validated=bool(payload.get("visible_validated", False)),
        affordances=tuple(str(value) for value in payload.get("affordances", spec.phases)),
        source_path=_clean_text(payload.get("source_path", spec.source_path)),
        dimensions=dimensions,  # type: ignore[arg-type]
        geometry=_clean_text(payload.get("geometry", geometry)),
        asset_key=_clean_text(payload.get("asset_key", spec.selection_id)),
    )


def resolve_asset(spec: OperationSpec, mapping: Mapping[str, Any]) -> AssetSpec:
    """Resolve an exact class asset, refusing the legacy functional proxy path."""
    embedded = mapping.get("asset")
    if isinstance(embedded, Mapping):
        return _asset_from_mapping(spec, embedded)
    geometry, dimensions = _default_geometry_and_size(spec.object_name, spec.selection_id)
    asset_path = CUSTOM_ASSET_ROOT / spec.selection_id / "model.xml"
    return AssetSpec(
        selection_id=spec.selection_id,
        semantic_class=spec.object_name,
        kind="custom_same_class",
        asset_path_or_group=str(asset_path),
        exact_class=False,
        collision_validated=False,
        visible_validated=False,
        affordances=tuple(dict.fromkeys((*spec.phases, "grasp"))),
        source_path=spec.source_path,
        dimensions=dimensions,
        geometry=geometry,
        asset_key=spec.selection_id,
    )


def load_asset_catalog(path: Path = DEFAULT_CATALOG_PATH) -> dict[str, Mapping[str, Any]]:
    """Load the fixed 60-item asset catalog and reject substitutions eagerly."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AssetResolutionError(f"cannot read asset catalog: {path}") from error
    records = payload.get("assets", []) if isinstance(payload, Mapping) else []
    if not isinstance(records, list) or len(records) != 60:
        raise AssetResolutionError("asset catalog must contain exactly 60 records")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise AssetResolutionError("asset catalog record is not an object")
        selection_id = _clean_text(record.get("selection_id"))
        if not re.fullmatch(r"VLA82-\d{3}", selection_id) or selection_id in result:
            raise AssetResolutionError(f"duplicate or invalid selection ID: {selection_id!r}")
        result[selection_id] = record
    expected = {f"VLA82-{index:03d}" for index in range(1, 61)}
    if set(result) != expected:
        raise AssetResolutionError("asset catalog IDs differ from fixed VLA82 scope")
    return result


def _workbook_annotations() -> dict[str, dict[str, str]]:
    """Read the authoritative ledger with stdlib only (openpyxl is optional)."""
    workbook_dir = PROJECT_ROOT / "outputs" / "midterm_testing_vla82"
    files = sorted(workbook_dir.glob("*.xlsx"), key=lambda item: item.stat().st_size, reverse=True)
    if not files:
        return {}
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    with zipfile.ZipFile(files[0]) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.itertext()) for node in shared_root]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.attrib["Id"]: rel.attrib["Target"].lstrip("/") for rel in rels}
        sheets = list(workbook.find(f"{ns}sheets") or [])
        if len(sheets) < 3:
            return {}
        target = targets.get(sheets[2].attrib.get(f"{rel_ns}id", ""), "")
        if not target:
            return {}
        sheet = ET.fromstring(archive.read("xl/" + target.lstrip("xl/")))
        output: dict[str, dict[str, str]] = {}
        for row in sheet.findall(f".//{ns}row"):
            values: dict[str, str] = {}
            for cell in row.findall(f"{ns}c"):
                column = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
                value = cell.find(f"{ns}v")
                if column is None or value is None:
                    continue
                raw = value.text or ""
                values[column.group(0)] = shared[int(raw)] if cell.attrib.get("t") == "s" else raw
            selection_id = values.get("A", "").strip()
            if re.fullmatch(r"VLA82-\d{3}", selection_id):
                output[selection_id] = {
                    "ledger_row": row.attrib.get("r", ""),
                    "object": values.get("H", ""),
                    "operation_label": values.get("I", ""),
                    "real_video_annotation": values.get("J", ""),
                    "data_path": values.get("K", ""),
                }
        return output


def source_manipulated_objects(spec: OperationSpec) -> tuple[str, ...]:
    """Extract counted objects from operation/instruction JSON for the manifest."""
    texts = [spec.operation_text]
    folder = Path(spec.source_path).parent
    for filename in ("operation.json", "instruction.json", "source_manifest.json"):
        candidate = folder / filename
        if candidate.is_file():
            texts.append(candidate.read_text(encoding="utf-8"))
    aliases = (
        ("can of soda", "\u6c7d\u6c34\u7f50"), ("sandal", "\u51c9\u978b"),
        ("paperback book", "\u4e66\u7c4d"), ("book", "\u4e66\u7c4d"),
        ("folder", "\u6587\u4ef6\u5939"), ("pencil case", "\u7b14\u76d2"),
        ("pencil", "\u94c5\u7b14"), ("pen", "\u7b14"), ("stapler", "\u8ba2\u4e66\u673a"),
        ("laptop", "\u7b14\u8bb0\u672c\u7535\u8111"), ("keyboard", "\u952e\u76d8"),
        ("mouse", "\u9f20\u6807"), ("toothbrush", "\u7259\u5237"),
        ("toothpaste", "\u7259\u818f"), ("soap dispenser", "\u7682\u6db2\u5668"),
    )
    counts = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "both": 2}
    objects = list(spec.manipulated_objects or (spec.object_name,))
    text = "\n".join(texts).lower()
    for phrase, canonical in aliases:
        match = re.search(rf"(?:(one|two|three|four|five|six|both)\s+)?{re.escape(phrase)}s?\b", text)
        if match:
            wanted = counts.get(match.group(1) or "one", 1)
            objects.extend([canonical] * max(0, wanted - objects.count(canonical)))
    return tuple(objects)


def load_authoritative_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Mapping[str, Any]]:
    """Load the immutable source-ledger manifest used by runtime scene assembly."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AssetResolutionError(f"cannot read authoritative manifest: {path}") from error
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    result = {str(item.get("selection_id")): item for item in records if isinstance(item, Mapping)}
    expected = {f"VLA82-{index:03d}" for index in range(1, 61)}
    if set(result) != expected:
        raise AssetResolutionError("authoritative manifest IDs differ from fixed VLA82 scope")
    return result


def write_authoritative_asset_manifest(
    specs: Sequence[OperationSpec], path: Path = DEFAULT_MANIFEST_PATH
) -> Path:
    """Compile source row/video/frame, profile, and fixture target into one manifest."""
    if len(specs) != 60:
        raise AssetResolutionError("authoritative manifest requires 60 selection records")
    annotations = _workbook_annotations()
    mappings = json.loads((PROJECT_ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json").read_text(encoding="utf-8"))
    classes = {str(item["selection_id"]): str(item.get("task_class", "")) for item in mappings.get("mappings", [])}
    records = []
    for spec in specs:
        geometry, dimensions = _default_geometry_and_size(spec.object_name, spec.selection_id)
        video = _source_video(spec.source_path)
        task_class = classes.get(spec.selection_id, "")
        records.append(
            {
                "selection_id": spec.selection_id,
                "source_table": spec.source_table,
                "source_path": spec.source_path,
                "source_sha256": spec.source_sha256,
                "ledger": annotations.get(spec.selection_id, {}),
                "source_video": str(video.resolve()) if video else "",
                "source_frame": str((CUSTOM_ASSET_ROOT / spec.selection_id / "source_texture.png").resolve()),
                "object_profile": {"geometry": geometry, "dimensions": list(dimensions)},
                "manipulated_objects": list(source_manipulated_objects(spec)),
                "fixture_target": {
                    "task_class": task_class,
                    "part_role": "fixture" if geometry == "fixture" else "object",
                    "required_identifier": task_class if geometry == "fixture" else "obj",
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"schema_version": "vla82_authoritative_manifest_v1", "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def write_asset_catalog(specs: Sequence[OperationSpec], path: Path = DEFAULT_CATALOG_PATH) -> Path:
    """Materialize the versioned 60-item catalog from compiled source specs."""
    if len(specs) != 60 or len({item.selection_id for item in specs}) != 60:
        raise AssetResolutionError("cannot build catalog outside the fixed 60-item scope")
    write_authoritative_asset_manifest(specs)
    mapping_path = PROJECT_ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
    mappings = {
        str(item["selection_id"]): item
        for item in json.loads(mapping_path.read_text(encoding="utf-8")).get("mappings", [])
    }
    records: list[dict[str, Any]] = []
    for spec in specs:
        geometry, dimensions = _default_geometry_and_size(spec.object_name, spec.selection_id)
        task_class = str(mappings.get(spec.selection_id, {}).get("task_class", ""))
        # Furniture / appliance part labels are validated as native fixtures
        # even where the historical mapping used a PickPlace task shell.
        fixture_part = spec.selection_id in {f"VLA82-{index:03d}" for index in range(26, 36)} or (
            bool(task_class) and not task_class.startswith("PickPlace")
        )
        records.append(
            {
                "selection_id": spec.selection_id,
                "semantic_class": spec.object_name,
                "kind": "fixture_part" if fixture_part else "custom_same_class",
                "asset_path_or_group": task_class if fixture_part else str((CUSTOM_ASSET_ROOT / spec.selection_id / "model.xml").resolve()),
                # These gates are deliberately false in source control.  They
                # turn true only when this selection's source frame, MJCF, and
                # collision bodies have been verified at runtime.
                "exact_class": False,
                "collision_validated": False,
                "visible_validated": False,
                "affordances": list(dict.fromkeys((*spec.phases, "grasp"))),
                "source_path": spec.source_path,
                "dimensions": list(dimensions),
                "geometry": geometry,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": "vla82_exact_assets_v1", "assets": records}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _source_video(source_path: str) -> Path | None:
    folder = Path(source_path).parent
    if not folder.is_dir():
        return None
    videos = sorted(
        candidate
        for candidate in folder.rglob("*")
        if candidate.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    )
    return videos[0] if videos else None


def _source_texture(video: Path | None, output: Path) -> bool:
    if video is None:
        return False
    try:
        import imageio.v3 as iio

        frame = iio.imread(video, index=0)
        if getattr(frame, "ndim", 0) != 3:
            return False
        iio.imwrite(output, frame)
        return output.is_file() and output.stat().st_size > 0
    except Exception:
        return False


def curated_appearance_texture_path(asset: AssetSpec) -> Path | None:
    """Return an approved recognizable package texture without replacing source evidence."""
    if (
        asset.selection_id not in CURATED_APPEARANCE_SELECTIONS
        or asset.semantic_class != "卫生巾盒"
    ):
        return None
    candidate = Path(asset.asset_path_or_group).parent / "appearance_texture.png"
    return candidate if candidate.is_file() else None


def _vla82_037_oriented_half_extents(asset: AssetSpec) -> tuple[float, float, float]:
    """Lay long utensils flat without changing their dimensions."""
    sx, sy, sz = (float(value) for value in asset.dimensions)
    if asset.selection_id not in {"VLA82-037", "VLA82-039", "VLA82-041"}:
        return sx, sy, sz
    return sz, sy, sx


def validate_vla82_037_drawer_asset_xml(
    asset: AssetSpec, model_text: str,
) -> tuple[float, float, float]:
    """Reject vertical or out-of-bounds long-utensil geometry before drawer reset."""
    if asset.selection_id not in {"VLA82-037", "VLA82-039", "VLA82-041"}:
        return tuple(float(value) for value in asset.dimensions)
    try:
        root = ET.fromstring(model_text)
        bbox_geom = next(
            geom for geom in root.iter("geom") if geom.get("name") == "reg_bbox"
        )
        bbox = np.fromstring(bbox_geom.attrib["size"], sep=" ", dtype=float)
    except (ET.ParseError, KeyError, StopIteration, ValueError) as error:
        raise AssetResolutionError(f"{asset.selection_id} drawer preflight cannot read reg_bbox") from error
    if bbox.shape != (3,) or int(np.argmax(bbox)) == 2 or 2.0 * float(bbox[2]) > 0.05:
        raise AssetResolutionError(f"{asset.selection_id} requires horizontal drawer orientation")
    if not np.allclose(np.sort(bbox), np.sort(np.asarray(asset.dimensions, dtype=float))):
        raise AssetResolutionError("VLA82-037 oriented dimensions differ from authoritative dimensions")
    for geom in root.iter("geom"):
        if geom.get("name") not in {"visual", "collision"}:
            continue
        size = np.fromstring(geom.attrib.get("size", ""), sep=" ", dtype=float)
        pos = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ", dtype=float)
        if size.shape != (3,) or pos.shape != (3,) or np.any(np.abs(pos) + size > bbox + 1e-9):
            raise AssetResolutionError("VLA82-037 reg_bbox does not cover physical geometry")
    return tuple(float(value) for value in bbox)


def _xml_for_asset(asset: AssetSpec, texture_name: str) -> str:
    sx, sy, sz = asset.dimensions
    friction_profile = profile_for(asset.selection_id, asset.semantic_class)
    friction = format_friction(friction_profile)
    # 110 x 70 x 36 mm bounding volume at 40 kg/m³ is ~11.1 g: a dry
    # household sponge.  The softer contact curve remains ordinary MuJoCo
    # collision physics (no weld, adhesion, or runtime state modification).
    density, solref, solimp = (65, "0.02 1", "0.88 0.95 0.003") if asset.geometry == "sponge" else (300, "0.02 1", "0.9 0.95 0.001")
    if asset.selection_id == "VLA82-004":
        # Translucent cleaning spray bottle with a clearly readable trigger
        # silhouette.  The original proven box remains the only collision.
        bbox = "0.045000 0.030000 0.112000"
        visual_and_collision = '''    <geom name="spray_bottle_body" class="visual" type="cylinder" pos="0 0 -0.014" size="0.024 0.068" rgba="0.34 0.78 0.96 0.72"/>
    <geom name="spray_bottle_shoulder" class="visual" type="ellipsoid" pos="0 0 0.049" size="0.024 0.024 0.018" rgba="0.42 0.84 0.98 0.78"/>
    <geom name="spray_neck" class="visual" type="cylinder" pos="0 0 0.070" size="0.010 0.014" rgba="0.94 0.96 0.98 1"/>
    <geom name="spray_trigger_head" class="visual" type="box" pos="0 0 0.087" size="0.021 0.014 0.010" rgba="0.96 0.97 0.99 1"/>
    <geom name="spray_trigger" class="visual" type="box" pos="-0.013 -0.001 0.075" size="0.004 0.011 0.019" euler="0 -0.35 0" rgba="0.94 0.95 0.97 1"/>
    <geom name="spray_nozzle" class="visual" type="box" pos="0.029 0 0.091" size="0.018 0.009 0.006" rgba="0.18 0.23 0.30 1"/>
    <geom name="product_label" class="visual" type="box" pos="0 -0.0243 -0.012" size="0.015 0.001 0.028" rgba="0.96 0.98 1.00 1"/>
    <geom name="collision" class="collision" type="box" size="0.025000 0.025000 0.085000"/>'''
    elif asset.selection_id == "VLA82-005":
        bbox = "0.033000 0.033000 0.112000"
        wires = "\n".join(
            f'    <geom name="whisk_wire_{index}" class="visual" type="capsule" '
            f'pos="{(-.012 + index * .0048):.4f} 0 0.052" size="0.0014 0.043" '
            f'euler="0 {(-.22 + index * .088):.4f} 0" rgba="0.76 0.79 0.82 1"/>'
            for index in range(6)
        )
        visual_and_collision = f'''    <geom name="whisk_handle" class="visual" type="cylinder" pos="0 0 -0.052" size="0.010 0.050" rgba="0.10 0.13 0.18 1"/>
    <geom name="whisk_handle_cap" class="visual" type="ellipsoid" pos="0 0 -0.103" size="0.010 0.010 0.008" rgba="0.08 0.10 0.14 1"/>
    <geom name="whisk_metal_neck" class="visual" type="cylinder" pos="0 0 0.001" size="0.005 0.018" rgba="0.72 0.75 0.79 1"/>
{wires}
    <geom name="whisk_wire_crown" class="visual" type="ellipsoid" pos="0 0 0.094" size="0.018 0.018 0.011" rgba="0.70 0.73 0.77 0.55"/>
    <geom name="collision" class="collision" type="box" size="0.012000 0.012000 0.105000"/>'''
    elif asset.selection_id == "VLA82-014":
        bbox = "0.071000 0.045000 0.060000"
        visual_and_collision = '''    <geom name="measuring_cup_body" class="visual" type="cylinder" pos="0 0 -0.002" size="0.039 0.053" rgba="0.72 0.91 1.00 0.52"/>
    <geom name="measuring_cup_inner" class="visual" type="cylinder" pos="0 0 0.051" size="0.034 0.003" rgba="0.15 0.28 0.35 1"/>
    <geom name="measuring_cup_rim" class="visual" type="cylinder" pos="0 0 0.055" size="0.042 0.003" rgba="0.80 0.95 1.00 0.95"/>
    <geom name="measuring_handle" class="visual" type="capsule" pos="0.054 0 -0.004" size="0.006 0.031" euler="0 0.55 0" rgba="0.55 0.83 0.94 0.92"/>
    <geom name="measure_mark_25" class="visual" type="box" pos="0 -0.0393 -0.026" size="0.014 0.001 0.0015" rgba="0.88 0.08 0.08 1"/>
    <geom name="measure_mark_50" class="visual" type="box" pos="0 -0.0393 -0.005" size="0.019 0.001 0.0015" rgba="0.88 0.08 0.08 1"/>
    <geom name="measure_mark_75" class="visual" type="box" pos="0 -0.0393 0.016" size="0.014 0.001 0.0015" rgba="0.88 0.08 0.08 1"/>
    <geom name="collision" class="collision" type="cylinder" size="0.040000 0.055000"/>'''
    elif asset.selection_id == "VLA82-017":
        bbox = "0.036000 0.036000 0.119000"
        rings = "\n".join(
            f'    <geom name="grip_ring_{index}" class="visual" type="cylinder" pos="0 0 {(-.040 + index * .020):.4f}" size="0.0335 0.0022" rgba="0.25 0.66 0.91 0.88"/>'
            for index in range(3)
        )
        visual_and_collision = f'''    <geom name="bottle_body" class="visual" type="cylinder" pos="0 0 -0.012" size="0.031 0.082" rgba="0.22 0.66 0.94 0.62"/>
{rings}
    <geom name="bottle_shoulder" class="visual" type="ellipsoid" pos="0 0 0.070" size="0.031 0.031 0.018" rgba="0.27 0.71 0.96 0.68"/>
    <geom name="bottle_neck" class="visual" type="cylinder" pos="0 0 0.090" size="0.014 0.014" rgba="0.34 0.76 0.97 0.76"/>
    <geom name="bottle_cap" class="visual" type="cylinder" pos="0 0 0.107" size="0.016 0.010" rgba="0.08 0.38 0.78 1"/>
    <geom name="collision" class="collision" type="box" size="0.032000 0.032000 0.100000"/>'''
    elif asset.selection_id == "VLA82-019":
        bbox = "0.035000 0.035000 0.059000"
        visual_and_collision = '''    <geom name="can_body" class="visual" type="cylinder" size="0.0315 0.054" rgba="0.72 0.74 0.76 1"/>
    <geom name="can_label" class="visual" type="cylinder" pos="0 0 -0.003" size="0.0325 0.038" rgba="0.82 0.18 0.10 1"/>
    <geom name="can_top_rim" class="visual" type="cylinder" pos="0 0 0.055" size="0.034 0.003" rgba="0.86 0.88 0.90 1"/>
    <geom name="can_bottom_rim" class="visual" type="cylinder" pos="0 0 -0.055" size="0.034 0.003" rgba="0.82 0.84 0.86 1"/>
    <geom name="pull_tab" class="visual" type="ellipsoid" pos="0.007 0 0.0585" size="0.010 0.005 0.0012" rgba="0.36 0.38 0.40 1"/>
    <geom name="collision" class="collision" type="cylinder" size="0.032000 0.055000"/>'''
    elif asset.selection_id == "VLA82-036":
        bbox = "0.070000 0.045000 0.059000"
        visual_and_collision = '''    <geom name="mug_body" class="visual" type="cylinder" pos="0 0 -0.002" size="0.039 0.053" rgba="0.94 0.94 0.91 1"/>
    <geom name="mug_rim" class="visual" type="cylinder" pos="0 0 0.054" size="0.042 0.003" rgba="0.99 0.99 0.98 1"/>
    <geom name="mug_inner" class="visual" type="cylinder" pos="0 0 0.056" size="0.034 0.0015" rgba="0.18 0.10 0.06 1"/>
    <geom name="mug_handle_0" class="visual" type="capsule" pos="0.051 0 0.019" size="0.006 0.018" euler="0 0.62 0" rgba="0.94 0.94 0.91 1"/>
    <geom name="mug_handle_1" class="visual" type="capsule" pos="0.064 0 -0.004" size="0.006 0.018" rgba="0.94 0.94 0.91 1"/>
    <geom name="mug_handle_2" class="visual" type="capsule" pos="0.051 0 -0.027" size="0.006 0.018" euler="0 -0.62 0" rgba="0.94 0.94 0.91 1"/>
    <geom name="collision" class="collision" type="cylinder" size="0.040000 0.055000"/>'''
    elif asset.selection_id == "VLA82-038":
        bbox = "0.048000 0.024000 0.105000"
        visual_and_collision = '''    <geom name="pizza_handle" class="visual" type="capsule" pos="0 0 -0.050" size="0.010 0.050" rgba="0.10 0.12 0.16 1"/>
    <geom name="pizza_handle_grip" class="visual" type="cylinder" pos="0 0 -0.056" size="0.012 0.036" rgba="0.18 0.20 0.24 1"/>
    <geom name="pizza_fork" class="visual" type="box" pos="0 0 0.014" size="0.006 0.005 0.026" rgba="0.66 0.69 0.72 1"/>
    <geom name="pizza_blade" class="visual" type="cylinder" pos="0 0 0.066" size="0.040 0.0025" quat="0.7071068 0.7071068 0 0" rgba="0.78 0.81 0.84 1"/>
    <geom name="blade_axle" class="visual" type="cylinder" pos="0 0 0.066" size="0.007 0.008" quat="0.7071068 0.7071068 0 0" rgba="0.24 0.26 0.30 1"/>
    <geom name="collision" class="collision" type="box" size="0.012000 0.012000 0.100000"/>'''
    elif asset.selection_id == "VLA82-040":
        # A household pitcher needs a body narrower than the Panda aperture.
        # Keep the authoritative 100 x 100 x 230 mm bounding volume, while the
        # physical liquid vessel is a 70 mm cylinder and the compact U-handle
        # is visual-only so it cannot snag the fingers during transport.
        bbox = "0.050000 0.050000 0.115000"
        visual_and_collision = '''    <geom name="body_visual" class="visual" type="cylinder" size="0.035 0.105" material="source_material"/>
    <geom name="rim_visual" class="visual" type="cylinder" pos="0 0 0.108" size="0.039 0.006" rgba="0.82 0.84 0.88 1"/>
    <geom name="spout_visual" class="visual" type="box" pos="-0.040 0 0.095" size="0.015 0.018 0.010" euler="0 -0.28 0" rgba="0.76 0.80 0.84 1"/>
    <geom name="handle_outer_visual" class="visual" type="capsule" pos="0.039 0 -0.055" size="0.005 0.006" euler="0 1.5707963 0" rgba="0.22 0.25 0.30 1"/>
    <geom name="handle_upper_visual" class="visual" type="capsule" pos="0.045 0 0.0025" size="0.005 0.0575" rgba="0.22 0.25 0.30 1"/>
    <geom name="handle_lower_visual" class="visual" type="capsule" pos="0.039 0 0.060" size="0.005 0.006" euler="0 1.5707963 0" rgba="0.22 0.25 0.30 1"/>
    <geom name="body_collision" class="collision" type="cylinder" size="0.035 0.105"/>'''
    elif asset.selection_id == "VLA82-042":
        bbox = "0.042000 0.034000 0.124000"
        bristles = "\n".join(
            f'    <geom name="bristle_group_{index}" class="visual" type="box" '
            f'pos="{(-.024 + index * .016):.4f} 0.021 0.105" size="0.006 0.018 0.012" '
            f'rgba="{("0.93 0.96 0.98 1" if index % 2 == 0 else "0.20 0.70 0.92 1")}"/>'
            for index in range(4)
        )
        visual_and_collision = f'''    <geom name="handle_visual" class="visual" type="capsule" pos="0 0 -0.034" size="0.012 0.078" rgba="0.08 0.48 0.75 1"/>
    <geom name="brush_grip" class="visual" type="box" pos="0 -0.011 -0.038" size="0.006 0.002 0.042" rgba="0.88 0.95 0.98 1"/>
    <geom name="brush_neck" class="visual" type="cylinder" pos="0 0 0.057" size="0.010 0.025" rgba="0.10 0.53 0.80 1"/>
    <geom name="brush_head_visual" class="visual" type="ellipsoid" pos="0 0 0.087" size="0.037 0.023 0.020" rgba="0.10 0.56 0.82 1"/>
    <geom name="brush_head" class="visual" type="ellipsoid" pos="0 -0.018 0.091" size="0.025 0.006 0.012" rgba="0.18 0.66 0.88 1"/>
    <geom name="bristles_visual" class="visual" type="box" pos="0 0.018 0.104" size="0.034 0.004 0.011" rgba="0.88 0.94 0.97 1"/>
{bristles}
    <geom name="collision" class="collision" type="box" size="0.016000 0.016000 0.115000"/>'''
    elif asset.selection_id == "VLA82-045":
        bbox = "0.082000 0.042000 0.050000"
        visual_and_collision = '''    <geom name="foil_box" class="visual" type="box" pos="0 0 -0.006" size="0.080 0.030 0.024" rgba="0.16 0.38 0.73 1"/>
    <geom name="foil_roll" class="visual" type="cylinder" pos="0 0 0.017" size="0.020 0.070" quat="0.7071068 0 0.7071068 0" rgba="0.83 0.86 0.89 1"/>
    <geom name="foil_core" class="visual" type="cylinder" pos="0 0 0.017" size="0.007 0.072" quat="0.7071068 0 0.7071068 0" rgba="0.50 0.32 0.18 1"/>
    <geom name="foil_sheet" class="visual" type="box" pos="0 -0.026 0.023" size="0.075 0.001 0.018" euler="-0.22 0 0" rgba="0.88 0.91 0.94 0.92"/>
    <geom name="serrated_cutter" class="visual" type="box" pos="0 -0.031 0.003" size="0.081 0.0015 0.002" rgba="0.72 0.75 0.78 1"/>
    <geom name="collision" class="collision" type="box" size="0.080000 0.030000 0.030000"/>'''
    elif asset.semantic_class == "卫浴搁板":
        # A recognizable wall-style bathroom rack. Every visible structural
        # part has a matching collision geom so neither the object nor the
        # gripper can pass through the rack. The capture contract separately
        # binds collision_bottom as the semantic support surface.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="bathroom_shelf_bottom" class="visual" mass="0" type="box" pos="0 0 0" size="{sx:.6f} {sy:.6f} {sz:.6f}" rgba="0.86 0.91 0.94 1"/>
    <geom name="bathroom_shelf_back" class="visual" mass="0" type="box" pos="0 {sy - .006:.6f} 0.030" size="{sx:.6f} 0.006 0.030" rgba="0.72 0.80 0.84 1"/>
    <geom name="bathroom_shelf_left_side" class="visual" mass="0" type="box" pos="{-sx + .006:.6f} 0 0.026" size="0.006 {sy:.6f} 0.026" rgba="0.76 0.84 0.88 1"/>
    <geom name="bathroom_shelf_right_side" class="visual" mass="0" type="box" pos="{sx - .006:.6f} 0 0.026" size="0.006 {sy:.6f} 0.026" rgba="0.76 0.84 0.88 1"/>
    <geom name="bathroom_shelf_front_lip" class="visual" mass="0" type="box" pos="0 {-sy + .004:.6f} 0.010" size="{sx:.6f} 0.004 0.010" rgba="0.68 0.77 0.82 1"/>
    <geom name="bathroom_shelf_mount_left" class="visual" mass="0" type="cylinder" pos="-0.073 {sy - .012:.6f} 0.052" size="0.007 0.007" quat="0.7071068 0.7071068 0 0" rgba="0.40 0.46 0.50 1"/>
    <geom name="bathroom_shelf_mount_right" class="visual" mass="0" type="cylinder" pos="0.073 {sy - .012:.6f} 0.052" size="0.007 0.007" quat="0.7071068 0.7071068 0 0" rgba="0.40 0.46 0.50 1"/>
    <geom name="collision_bottom" class="collision" type="box" pos="0 0 0" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>
    <geom name="collision_back" class="collision" type="box" pos="0 {sy - .006:.6f} 0.030" size="{sx:.6f} 0.006 0.030"/>
    <geom name="collision_left" class="collision" type="box" pos="{-sx + .006:.6f} 0 0.026" size="0.006 {sy:.6f} 0.026"/>
    <geom name="collision_right" class="collision" type="box" pos="{sx - .006:.6f} 0 0.026" size="0.006 {sy:.6f} 0.026"/>
    <geom name="collision_front_lip" class="collision" type="box" pos="0 {-sy + .004:.6f} 0.010" size="{sx:.6f} 0.004 0.010"/>'''
    elif asset.selection_id == "VLA82-016" and asset.semantic_class == "书籍":
        # A conventional hardcover book. All visible details stay within the
        # honest rectangular envelope; the only collision is the full book.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="book_pages" class="visual" type="box" pos="0.002 0 0" size="{sx - .005:.6f} {sy - .003:.6f} {max(sz - .003, .002):.6f}" rgba="0.94 0.88 0.72 1"/>
    <geom name="book_cover_top" class="visual" type="box" pos="0 0 {sz - .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.22 0.38 0.68 1"/>
    <geom name="book_cover_bottom" class="visual" type="box" pos="0 0 {-sz + .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.12 0.24 0.48 1"/>
    <geom name="book_spine" class="visual" type="box" pos="{-sx + .004:.6f} 0 0" size="0.004 {sy:.6f} {sz:.6f}" rgba="0.10 0.20 0.42 1"/>
    <geom name="book_title_label" class="visual" type="box" pos="0.012 0 {sz + .00025:.6f}" size="0.026 0.014 0.00025" rgba="0.95 0.89 0.62 1"/>
    <geom name="book_title_bar" class="visual" type="box" pos="0.012 0 {sz + .00055:.6f}" size="0.018 0.002 0.00020" rgba="0.48 0.18 0.12 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-016" and asset.semantic_class == "床头柜":
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="nightstand_body" class="visual" mass="0" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}" rgba="0.48 0.25 0.10 1"/>
    <geom name="nightstand_top" class="visual" mass="0" type="box" pos="0 0 {sz - .003:.6f}" size="{sx:.6f} {sy:.6f} 0.003" rgba="0.68 0.38 0.16 1"/>
    <geom name="nightstand_drawer_front" class="visual" mass="0" type="box" pos="0 {-sy - .001:.6f} 0.006" size="{sx - .010:.6f} 0.001 {sz - .011:.6f}" rgba="0.58 0.31 0.13 1"/>
    <geom name="nightstand_handle" class="visual" mass="0" type="capsule" pos="0 {-sy - .004:.6f} 0.011" size="0.004 0.025" quat="0.7071068 0.7071068 0 0" rgba="0.18 0.12 0.08 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-016" and asset.semantic_class == "床":
        # A compact but unmistakable bedroom bed: timber frame, upholstered
        # mattress, headboard, pillow and blanket. The mattress collision is a
        # solid floor-supported volume whose top exactly matches the spawn Z.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="bed_frame" class="visual" mass="0" type="box" pos="0 0 -0.075" size="{sx:.6f} {sy:.6f} {max(sz - .075, .04):.6f}" rgba="0.35 0.18 0.08 1"/>
    <geom name="bed_mattress" class="visual" mass="0" type="box" pos="0 0 {sz - .055:.6f}" size="{sx - .010:.6f} {sy - .012:.6f} 0.055" rgba="0.93 0.91 0.84 1"/>
    <geom name="bed_blanket" class="visual" mass="0" type="box" pos="0 0.090 {sz + .001:.6f}" size="{sx - .016:.6f} {sy - .110:.6f} 0.003" rgba="0.32 0.55 0.72 1"/>
    <geom name="bed_headboard" class="visual" mass="0" type="box" pos="0 {sy - .018:.6f} 0.135" size="{sx:.6f} 0.018 {sz + .060:.6f}" rgba="0.42 0.22 0.10 1"/>
    <geom name="bed_pillow" class="visual" mass="0" type="ellipsoid" pos="0 {sy - .105:.6f} {sz + .025:.6f}" size="{sx * .58:.6f} 0.085 0.025" rgba="0.98 0.98 0.96 1"/>
    <geom name="collision_mattress" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-046" and asset.semantic_class == "文件夹":
        # A conventional closed document folder: every visible and collision
        # part stays inside one rectangular envelope. There is deliberately no
        # raised grasp block or hidden handle.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="folder_pages" class="visual" type="box" pos="0.003 0 0" size="{sx - .007:.6f} {sy - .004:.6f} {max(sz - .002, .002):.6f}" rgba="0.94 0.91 0.82 1"/>
    <geom name="folder_cover_top" class="visual" type="box" pos="0 0 {sz - .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.10 0.34 0.72 1"/>
    <geom name="folder_cover_bottom" class="visual" type="box" pos="0 0 {-sz + .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.07 0.25 0.58 1"/>
    <geom name="folder_spine" class="visual" type="box" pos="{-sx + .005:.6f} 0 0" size="0.005 {sy:.6f} {sz:.6f}" rgba="0.05 0.20 0.50 1"/>
    <geom name="folder_label" class="visual" type="box" pos="0.014 0 {sz + .00025:.6f}" size="0.030 {max(sy - .006, .010):.6f} 0.00025" rgba="0.96 0.97 0.94 1"/>
    <geom name="folder_label_bar" class="visual" type="box" pos="0.014 0 {sz + .00055:.6f}" size="0.021 0.002 0.00020" rgba="0.18 0.48 0.82 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-048" and asset.semantic_class == "笔记本":
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="notebook_pages" class="visual" type="box" pos="0.003 0 0" size="{sx - .006:.6f} {sy - .003:.6f} {max(sz - .002, .002):.6f}" rgba="0.94 0.90 0.80 1"/>
    <geom name="notebook_cover_top" class="visual" type="box" pos="0 0 {sz - .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.18 0.58 0.36 1"/>
    <geom name="notebook_cover_bottom" class="visual" type="box" pos="0 0 {-sz + .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.10 0.38 0.23 1"/>
    <geom name="notebook_spine" class="visual" type="box" pos="{-sx + .004:.6f} 0 0" size="0.004 {sy:.6f} {sz:.6f}" rgba="0.08 0.24 0.16 1"/>
    <geom name="notebook_label" class="visual" type="box" pos="0.010 0 {sz + .00025:.6f}" size="0.020 0.010 0.00025" rgba="0.97 0.96 0.88 1"/>
    <geom name="notebook_elastic" class="visual" type="box" pos="0.032 0 {sz + .00055:.6f}" size="0.0015 {sy:.6f} 0.00020" rgba="0.94 0.73 0.18 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-048" and asset.semantic_class == "文件夹":
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="target_folder_pages" class="visual" type="box" pos="0.003 0 0" size="{sx - .006:.6f} {sy - .003:.6f} {max(sz - .002, .002):.6f}" rgba="0.95 0.91 0.82 1"/>
    <geom name="target_folder_cover" class="visual" type="box" pos="0 0 {sz - .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.82 0.30 0.18 1"/>
    <geom name="target_folder_bottom" class="visual" type="box" pos="0 0 {-sz + .001:.6f}" size="{sx:.6f} {sy:.6f} 0.001" rgba="0.58 0.16 0.10 1"/>
    <geom name="target_folder_spine" class="visual" type="box" pos="{-sx + .004:.6f} 0 0" size="0.004 {sy:.6f} {sz:.6f}" rgba="0.45 0.10 0.07 1"/>
    <geom name="target_folder_label" class="visual" type="box" pos="0.020 0 {sz + .0007:.6f}" size="0.026 0.016 0.0007" rgba="0.98 0.95 0.86 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-055" and asset.semantic_class == "卫生巾盒":
        # A compact retail hygiene box: retain the proven one-box collision,
        # while a blush package, pale front label, and winged-pad pictogram
        # make the object recognizable from the audit camera's front view.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="hygiene_box_body" class="visual" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}" rgba="0.92 0.34 0.55 1"/>
    <geom name="hygiene_front_label" class="visual" type="box" pos="0 {-sy - .0010:.6f} 0" size="{sx * .84:.6f} 0.0010 {sz * .78:.6f}" rgba="1.00 0.92 0.96 1"/>
    <geom name="pad_center" class="visual" type="capsule" pos="0 {-sy - .0023:.6f} 0" size="0.0060 0.0120" rgba="0.99 0.99 1.00 1"/>
    <geom name="pad_left_wing" class="visual" type="ellipsoid" pos="-0.0100 {-sy - .0024:.6f} 0" size="0.0100 0.0018 0.0060" rgba="0.99 0.99 1.00 1"/>
    <geom name="pad_right_wing" class="visual" type="ellipsoid" pos="0.0100 {-sy - .0024:.6f} 0" size="0.0100 0.0018 0.0060" rgba="0.99 0.99 1.00 1"/>
    <geom name="package_wave_top" class="visual" type="box" pos="-0.0480 {-sy - .0022:.6f} 0.0130" size="0.0200 0.0013 0.0020" euler="0 0 -0.14" rgba="0.67 0.43 0.78 1"/>
    <geom name="package_wave_bottom" class="visual" type="box" pos="0.0480 {-sy - .0022:.6f} -0.0130" size="0.0200 0.0013 0.0020" euler="0 0 -0.14" rgba="0.31 0.74 0.72 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-060":
        bbox = "0.054000 0.040000 0.127000"
        visual_and_collision = '''    <geom name="soap_bottle" class="visual" type="cylinder" pos="0 0 -0.012" size="0.034 0.082" rgba="0.53 0.91 0.73 0.58"/>
    <geom name="liquid_layer" class="visual" type="cylinder" pos="0 0 -0.034" size="0.032 0.058" rgba="0.25 0.79 0.55 0.72"/>
    <geom name="soap_shoulder" class="visual" type="ellipsoid" pos="0 0 0.069" size="0.034 0.034 0.019" rgba="0.58 0.94 0.77 0.66"/>
    <geom name="pump_stem" class="visual" type="cylinder" pos="0 0 0.094" size="0.009 0.019" rgba="0.94 0.96 0.95 1"/>
    <geom name="pump_head" class="visual" type="box" pos="0.007 0 0.113" size="0.026 0.012 0.007" rgba="0.95 0.97 0.96 1"/>
    <geom name="pump_nozzle" class="visual" type="box" pos="0.040 0 0.113" size="0.018 0.007 0.005" rgba="0.92 0.94 0.93 1"/>
    <geom name="collision" class="collision" type="box" size="0.035000 0.035000 0.100000"/>'''
    elif asset.selection_id == "VLA82-058" and asset.semantic_class == "牙刷":
        # Keep the proven box collision envelope, but render the semantic
        # features that distinguish a toothbrush from a plain rod.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        bristle_lines: list[str] = []
        white_index = 0
        blue_index = 0
        for row, z_pos in enumerate((.054, .062, .070, .078)):
            for column, x_pos in enumerate((-.006, -.003, 0., .003, .006)):
                blue = row == 1 or (row == 2 and column in {1, 3})
                color = "0.42 0.78 1.00 1" if blue else "0.97 0.99 1.00 1"
                prefix = "blue" if blue else "white"
                index = blue_index if blue else white_index
                bristle_lines.append(
                    f'    <geom name="toothbrush_bristle_{prefix}_{index}" class="visual" '
                    f'type="cylinder" pos="{x_pos:.4f} -0.01075 {z_pos:.4f}" '
                    f'size="0.00125 0.00625" quat="0.7071068 0.7071068 0 0" rgba="{color}"/>'
                )
                if blue:
                    blue_index += 1
                else:
                    white_index += 1
        bristles = "\n".join(bristle_lines)
        visual_and_collision = f'''    <geom name="toothbrush_handle_visual" class="visual" type="cylinder" pos="0 0 -0.0285" size="0.0075 0.0535" rgba="0.10 0.43 0.86 1"/>
    <geom name="toothbrush_grip_inlay_visual" class="visual" type="cylinder" pos="0 -0.006 -0.032" size="0.0022 0.024" rgba="0.93 0.97 1.00 1"/>
    <geom name="toothbrush_neck_visual" class="visual" type="cylinder" pos="0 0 0.036" size="0.0045 0.016" rgba="0.91 0.96 1.00 1"/>
    <geom name="toothbrush_head_visual" class="visual" type="ellipsoid" pos="0 0 0.066" size="0.010 0.0055 0.020" rgba="0.12 0.50 0.91 1"/>
{bristles}
    <geom name="collision" class="collision" type="box" size="0.010000 0.010000 0.090000"/>'''
    elif asset.selection_id == "VLA82-058" and asset.semantic_class == "漱口杯":
        # Approximate a smooth, genuinely open round cup with dense visual
        # capsule segments.  The five proven box collisions remain unchanged.
        segment_count = 28
        wall_radius = max(.012, sx - .004)
        wall_lines: list[str] = []
        rim_lines: list[str] = []
        points = [
            (
                wall_radius * float(np.cos(2.0 * np.pi * index / segment_count)),
                wall_radius * float(np.sin(2.0 * np.pi * index / segment_count)),
            )
            for index in range(segment_count)
        ]
        for index, (x_pos, y_pos) in enumerate(points):
            wall_lines.append(
                f'    <geom name="cup_wall_visual_{index}" class="visual" type="cylinder" '
                f'pos="{x_pos:.6f} {y_pos:.6f} 0.002500" size="0.0046 {sz - .0055:.6f}" '
                'rgba="0.32 0.72 0.94 0.78"/>'
            )
            next_x, next_y = points[(index + 1) % segment_count]
            segment_dx = next_x - x_pos
            segment_dy = next_y - y_pos
            segment_length = float(np.hypot(segment_dx, segment_dy))
            unit_x = segment_dx / segment_length
            unit_y = segment_dy / segment_length
            rim_lines.append(
                f'    <geom name="cup_rim_visual_{index}" class="visual" type="cylinder" '
                f'pos="{(x_pos + next_x) / 2.0:.6f} {(y_pos + next_y) / 2.0:.6f} {sz:.6f}" '
                f'size="0.0030 {segment_length / 2.0:.6f}" '
                f'quat="0.7071068 {-unit_y * .7071068:.7f} {unit_x * .7071068:.7f} 0" '
                'rgba="0.72 0.91 1.00 1"/>'
            )
        walls = "\n".join(wall_lines)
        rim = "\n".join(rim_lines)
        visual_and_collision = f'''    <geom name="cup_bottom_visual" class="visual" type="cylinder" pos="0 0 {-sz + .003:.6f}" size="{sx:.6f} 0.004" rgba="0.18 0.58 0.86 0.95"/>
{walls}
{rim}
    <geom name="collision_bottom" class="collision" type="box" pos="0 0 {-sz:.6f}" size="{sx:.6f} {sy:.6f} 0.008"/>
    <geom name="collision_front" class="collision" type="box" pos="0 {-sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}"/>
    <geom name="collision_back" class="collision" type="box" pos="0 {sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}"/>
    <geom name="collision_left" class="collision" type="box" pos="{-sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}"/>
    <geom name="collision_right" class="collision" type="box" pos="{sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}"/>'''
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
    elif asset.selection_id == "VLA82-041":
        # Preserve the proven flat box collision and inertia while replacing
        # the featureless texture box with a recognizable metal soup spoon.
        ox, oy, oz = _vla82_037_oriented_half_extents(asset)
        bbox = f"{ox:.6f} {oy:.6f} {oz:.6f}"
        legacy_mass = 8.0 * ox * oy * oz * 1000.0
        visual_and_collision = f'''    <geom name="spoon_bowl" class="visual" mass="0" type="ellipsoid" pos="-0.075 0 0" size="0.032 0.011 0.0055" rgba="0.82 0.85 0.88 1"/>
    <geom name="spoon_bowl_inner" class="visual" mass="0" type="ellipsoid" pos="-0.075 0 0.003" size="0.025 0.008 0.002" rgba="0.38 0.42 0.47 1"/>
    <geom name="spoon_neck" class="visual" mass="0" type="capsule" pos="-0.040 0 0" size="0.0035 0.018" quat="0.7071068 0 0.7071068 0" rgba="0.72 0.76 0.80 1"/>
    <geom name="spoon_handle" class="visual" mass="0" type="cylinder" pos="0.034 0 0" size="0.0045 0.070" quat="0.7071068 0 0.7071068 0" rgba="0.68 0.72 0.77 1"/>
    <geom name="spoon_handle_highlight" class="visual" mass="0" type="box" pos="0.034 -0.004 0.002" size="0.066 0.0008 0.0008" rgba="0.95 0.97 0.99 1"/>
    <geom name="legacy_visual_mass_carrier" class="visual" type="box" size="{ox:.6f} {oy:.6f} {oz:.6f}" mass="{legacy_mass:.9f}" rgba="0 0 0 0"/>
    <geom name="collision" class="collision" type="box" size="{ox:.6f} {oy:.6f} {oz:.6f}"/>'''
    elif asset.selection_id == "VLA82-051" and asset.semantic_class == "铅笔":
        # A conventional yellow pencil with a sharpened graphite end and
        # eraser.  The former one-box visual mass is carried transparently so
        # the already-verified two-pad grasp sees unchanged body dynamics.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        legacy_mass = 8.0 * sx * sy * sz * 1000.0
        visual_and_collision = f'''    <geom name="pencil_body" class="visual" mass="0" type="cylinder" pos="0 0 -0.004" size="0.005 0.073" rgba="0.96 0.68 0.08 1"/>
    <geom name="pencil_flat_0" class="visual" mass="0" type="box" pos="0 -0.0045 -0.004" size="0.0038 0.0008 0.073" rgba="1.00 0.79 0.12 1"/>
    <geom name="pencil_flat_1" class="visual" mass="0" type="box" pos="0.0039 0.0023 -0.004" size="0.0008 0.0034 0.073" rgba="0.82 0.48 0.03 1"/>
    <geom name="pencil_wood" class="visual" mass="0" type="capsule" pos="0 0 0.078" size="0.0045 0.010" rgba="0.82 0.62 0.37 1"/>
    <geom name="pencil_graphite" class="visual" mass="0" type="cylinder" pos="0 0 0.092" size="0.0015 0.003" rgba="0.08 0.08 0.09 1"/>
    <geom name="pencil_ferrule" class="visual" mass="0" type="cylinder" pos="0 0 -0.083" size="0.0055 0.008" rgba="0.72 0.75 0.78 1"/>
    <geom name="pencil_eraser" class="visual" mass="0" type="cylinder" pos="0 0 -0.089" size="0.0052 0.006" rgba="0.87 0.22 0.25 1"/>
    <geom name="legacy_visual_mass_carrier" class="visual" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}" mass="{legacy_mass:.9f}" rgba="0 0 0 0"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-051" and asset.semantic_class == "笔盒":
        # Keep the same five physical walls and open cavity.  Only massless
        # lining, rim, and zipper details are added to the colored shell.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="visual_bottom" class="visual" type="box" pos="0 0 {-sz:.6f}" size="{sx:.6f} {sy:.6f} 0.008" rgba="0.08 0.30 0.68 1"/>
    <geom name="visual_front" class="visual" type="box" pos="0 {-sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}" rgba="0.10 0.38 0.82 1"/>
    <geom name="visual_back" class="visual" type="box" pos="0 {sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}" rgba="0.10 0.38 0.82 1"/>
    <geom name="visual_left" class="visual" type="box" pos="{-sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}" rgba="0.08 0.30 0.68 1"/>
    <geom name="visual_right" class="visual" type="box" pos="{sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}" rgba="0.08 0.30 0.68 1"/>
    <geom name="case_lining" class="visual" mass="0" type="box" pos="0 0 {-sz + .009:.6f}" size="{sx - .010:.6f} {sy - .010:.6f} 0.002" rgba="0.72 0.84 0.96 1"/>
    <geom name="case_rim_front" class="visual" mass="0" type="box" pos="0 {-sy:.6f} {sz - .003:.6f}" size="{sx:.6f} 0.006 0.003" rgba="0.03 0.12 0.30 1"/>
    <geom name="case_rim_back" class="visual" mass="0" type="box" pos="0 {sy:.6f} {sz - .003:.6f}" size="{sx:.6f} 0.006 0.003" rgba="0.03 0.12 0.30 1"/>
    <geom name="case_rim_left" class="visual" mass="0" type="box" pos="{-sx:.6f} 0 {sz - .003:.6f}" size="0.006 {sy:.6f} 0.003" rgba="0.03 0.12 0.30 1"/>
    <geom name="case_rim_right" class="visual" mass="0" type="box" pos="{sx:.6f} 0 {sz - .003:.6f}" size="0.006 {sy:.6f} 0.003" rgba="0.03 0.12 0.30 1"/>
    <geom name="case_zip_pull" class="visual" mass="0" type="capsule" pos="{sx - .014:.6f} {-sy - .006:.6f} {sz - .003:.6f}" size="0.003 0.010" euler="0 1.5707963 0" rgba="0.82 0.84 0.86 1"/>
    <geom name="collision_bottom" class="collision" type="box" pos="0 0 {-sz:.6f}" size="{sx:.6f} {sy:.6f} 0.008"/>
    <geom name="collision_front" class="collision" type="box" pos="0 {-sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}"/>
    <geom name="collision_back" class="collision" type="box" pos="0 {sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}"/>
    <geom name="collision_left" class="collision" type="box" pos="{-sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}"/>
    <geom name="collision_right" class="collision" type="box" pos="{sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}"/>'''
    elif asset.selection_id == "VLA82-015":
        # Preserve a physical pan handle.  The 210 mm pan body is wider than
        # the Panda gripper aperture; the real affordance is its narrow handle.
        bbox = "0.175000 0.105000 0.035000"
        visual_and_collision = '''    <geom name="pan_body_visual" class="visual" type="cylinder" pos="-0.070 0 0" size="0.105000 0.035000" material="source_material"/>
    <geom name="pan_body_collision" class="collision" type="cylinder" pos="-0.070 0 0" size="0.105000 0.035000"/>
    <geom name="pan_handle_visual" class="visual" type="box" pos="0.105 0 0" size="0.070 0.018 0.012" material="source_material"/>
    <geom name="pan_handle_collision" class="collision" type="box" pos="0.105 0 0" size="0.070 0.018 0.012"/>'''
    elif asset.selection_id in {"VLA82-037", "VLA82-039", "VLA82-041"}:
        ox, oy, oz = _vla82_037_oriented_half_extents(asset)
        geom = f'type="box" size="{ox:.6f} {oy:.6f} {oz:.6f}"'
        bbox = f"{ox:.6f} {oy:.6f} {oz:.6f}"
        visual_and_collision = f'''    <geom name="visual" class="visual" {geom} material="source_material"/>
    <geom name="collision" class="collision" {geom}/>'''
    elif asset.selection_id == "VLA82-056" and asset.semantic_class == "固体香皂":
        # A bar of soap is a rounded rectangular solid, not a cylinder.  The
        # visible ellipsoid gives rounded edges while the slightly inset box
        # supplies a stable, honest 44-mm opposing-pad contact surface.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="soap_body" class="visual" mass="0" type="ellipsoid" size="{sx:.6f} {sy:.6f} {sz:.6f}" rgba="0.38 0.82 0.72 1"/>
    <geom name="soap_inset" class="visual" mass="0" type="ellipsoid" pos="0 0 {sz * .48:.6f}" size="{sx * .62:.6f} {sy * .58:.6f} {sz * .34:.6f}" rgba="0.72 0.95 0.88 1"/>
    <geom name="soap_groove_left" class="visual" mass="0" type="box" pos="{-sx * .30:.6f} 0 {sz - .001:.6f}" size="0.0012 {sy * .50:.6f} 0.001" rgba="0.18 0.58 0.50 1"/>
    <geom name="soap_groove_right" class="visual" mass="0" type="box" pos="{sx * .30:.6f} 0 {sz - .001:.6f}" size="0.0012 {sy * .50:.6f} 0.001" rgba="0.18 0.58 0.50 1"/>
    <geom name="collision" class="collision" type="box" size="{sx * .875:.6f} {min(sy, .022):.6f} {sz * .93:.6f}" mass="0.080"/>'''
    elif asset.geometry == "stapler":
        # A conventional desktop stapler is visually defined by a flat base,
        # a separate raised press arm, a rear hinge, and the bright metal
        # magazine between them.  Keep one box collision envelope so the
        # already-validated top grasp remains stable and inexpensive.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="stapler_base" class="visual" type="box" pos="0 0 {-sz * .55:.6f}" size="{sx:.6f} {sy:.6f} {sz * .28:.6f}" rgba="0.10 0.12 0.16 1"/>
    <geom name="stapler_metal_channel" class="visual" type="box" pos="{sx * .08:.6f} 0 {-sz * .10:.6f}" size="{sx * .78:.6f} {sy * .48:.6f} {sz * .12:.6f}" rgba="0.62 0.66 0.70 1"/>
    <geom name="stapler_upper_arm" class="visual" type="box" pos="{sx * .05:.6f} 0 {sz * .45:.6f}" size="{sx * .92:.6f} {sy * .82:.6f} {sz * .24:.6f}" euler="0 {-0.10:.6f} 0" rgba="0.16 0.20 0.27 1"/>
    <geom name="stapler_hinge" class="visual" type="cylinder" pos="{-sx * .82:.6f} 0 {sz * .08:.6f}" size="{sz * .24:.6f} {sy * .78:.6f}" quat="0.7071068 0.7071068 0 0" rgba="0.22 0.25 0.29 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.geometry == "mouse":
        # The source trajectory shows a compact dark computer mouse.  A
        # rounded shell, centre seam, and transverse scroll wheel remain
        # recognizable in the audit camera while one ellipsoid supplies
        # stable, ordinary MuJoCo grasp contacts.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        seam_x = sx * .40
        seam_length = sx * .50
        visual_and_collision = f'''    <geom name="visual" class="visual" type="ellipsoid" size="{sx:.6f} {sy:.6f} {sz:.6f}" rgba="0.10 0.11 0.13 1"/>
    <geom name="mouse_button_seam" class="visual" type="box" pos="{seam_x:.6f} 0 {sz * .90:.6f}" size="{seam_length:.6f} 0.0006 0.001" rgba="0.025 0.030 0.040 1"/>
    <geom name="mouse_wheel" class="visual" type="cylinder" pos="{sx * .26:.6f} 0 {sz:.6f}" size="0.0035 0.006" quat="0.7071068 0.7071068 0 0" rgba="0.34 0.36 0.39 1"/>
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.geometry == "keyboard":
        # Render a conventional dark keyboard with a visible 3x8 key grid.
        # The single base collision keeps it static and inexpensive while the
        # visual keys make the mouse's annotated destination unambiguous.
        bbox = f"{sx:.6f} {sy:.6f} {sz + .003:.6f}"
        key_lines = []
        for row in range(3):
            for column in range(8):
                x = -sx * .80 + column * (sx * 1.60 / 7.0)
                y = -sy * .62 + row * sy * .62
                key_lines.append(
                    f'    <geom name="keyboard_key_{row}_{column}" class="visual" '
                    f'type="box" pos="{x:.6f} {y:.6f} {sz + .0015:.6f}" '
                    f'size="{sx * .085:.6f} {sy * .115:.6f} 0.0015" rgba="0.72 0.74 0.77 1"/>'
                )
        keys = "\n".join(key_lines)
        visual_and_collision = f'''    <geom name="keyboard_base" class="visual" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}" rgba="0.055 0.065 0.080 1"/>
{keys}
    <geom name="collision" class="collision" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'''
    elif asset.geometry == "open_receptacle":
        visual_and_collision = f'''    <geom name="visual_bottom" class="visual" type="box" pos="0 0 {-sz:.6f}" size="{sx:.6f} {sy:.6f} 0.008" material="source_material"/>
    <geom name="visual_front" class="visual" type="box" pos="0 {-sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}" material="source_material"/>
    <geom name="visual_back" class="visual" type="box" pos="0 {sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}" material="source_material"/>
    <geom name="visual_left" class="visual" type="box" pos="{-sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}" material="source_material"/>
    <geom name="visual_right" class="visual" type="box" pos="{sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}" material="source_material"/>
    <geom name="collision_bottom" class="collision" type="box" pos="0 0 {-sz:.6f}" size="{sx:.6f} {sy:.6f} 0.008"/>
    <geom name="collision_front" class="collision" type="box" pos="0 {-sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}"/>
    <geom name="collision_back" class="collision" type="box" pos="0 {sy:.6f} 0" size="{sx:.6f} 0.008 {sz:.6f}"/>
    <geom name="collision_left" class="collision" type="box" pos="{-sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}"/>
    <geom name="collision_right" class="collision" type="box" pos="{sx:.6f} 0 0" size="0.008 {sy:.6f} {sz:.6f}"/>'''
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
    elif asset.geometry == "sponge":
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        core_x = max(0.001, sx - 0.004)
        visual_and_collision = f'''    <geom name="visual" class="visual" type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}" material="source_material"/>
    <geom name="collision_core" class="collision" type="box" size="{core_x:.6f} {sy:.6f} {sz:.6f}" mass="0.012"/>
    <geom name="collision_left_compliant" class="collision" type="box" pos="{-core_x:.6f} 0 0" size="0.004000 {sy:.6f} {sz:.6f}" mass="0.003"/>
    <geom name="collision_right_compliant" class="collision" type="box" pos="{core_x:.6f} 0 0" size="0.004000 {sy:.6f} {sz:.6f}" mass="0.003"/>'''
    elif asset.geometry in {"cylinder", "can", "cup", "dish", "pan", "soap"}:
        geom = f'type="cylinder" size="{sx:.6f} {sz:.6f}"'
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="visual" class="visual" {geom} material="source_material"/>
    <geom name="collision" class="collision" {geom}/>'''
    elif asset.geometry == "capsule":
        geom = f'type="capsule" fromto="0 0 {-sz:.6f} 0 0 {sz:.6f}" size="{sx:.6f}"'
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="visual" class="visual" {geom} material="source_material"/>
    <geom name="collision" class="collision" {geom}/>'''
    else:
        geom = f'type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"'
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        visual_and_collision = f'''    <geom name="visual" class="visual" {geom} material="source_material"/>
    <geom name="collision" class="collision" {geom}/>'''
    refreshed_visual_ids = {
        "VLA82-004", "VLA82-005", "VLA82-014", "VLA82-017", "VLA82-019",
        "VLA82-036", "VLA82-038", "VLA82-042", "VLA82-045", "VLA82-055", "VLA82-060",
    }
    visual_default_mass = ""
    if asset.selection_id in refreshed_visual_ids and not (
        asset.selection_id == "VLA82-055" and asset.semantic_class != "卫生巾盒"
    ):
        # MuJoCo visual geoms normally contribute their own density to body
        # mass and inertia even when contype/conaffinity are zero.  Keep all
        # new decorative parts massless, and reproduce the former one-geom
        # visual mass with one transparent carrier of the original shape.
        visual_default_mass = ' mass="0"'
        # Registration controls RoboCasa's object-wide scale.  It must stay
        # exactly equal to the authoritative pre-refresh dimensions even when
        # decorative handles, nozzles, or pump heads extend beyond it.
        bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        if asset.geometry in {"cylinder", "can", "cup", "dish", "pan", "soap"}:
            carrier_mass = float(np.pi * sx * sx * (2.0 * sz) * 1000.0)
            carrier_shape = f'type="cylinder" size="{sx:.6f} {sz:.6f}"'
        else:
            carrier_mass = float((2.0 * sx) * (2.0 * sy) * (2.0 * sz) * 1000.0)
            carrier_shape = f'type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}"'
        if asset.selection_id == "VLA82-055" and asset.semantic_class == "卫生巾盒":
            # A light paperboard hygiene package is roughly 150 g including
            # the 300 kg/m^3 collision body. The former volume-as-solid mass
            # made it over four times too heavy and caused pitch slip.
            carrier_mass = .03
        carrier = (
            f'    <geom name="legacy_visual_mass_carrier" class="visual" '
            f'{carrier_shape} mass="{carrier_mass:.9f}" rgba="0 0 0 0"/>'
        )
        visual_and_collision = visual_and_collision.replace(
            '    <geom name="collision"', f'{carrier}\n    <geom name="collision"', 1,
        )
    # The visual and collision geoms are separate so MuJoCo evaluates contacts
    # against a physical volume while appearance remains traceable to video.
    return f'''<mujoco model="{_safe_asset_name(asset.asset_key or asset.selection_id)}">
  <compiler angle="radian"/>
  <asset>
    <texture name="source_texture" type="2d" file="{texture_name}"/>
    <material name="source_material" texture="source_texture" rgba="1 1 1 1"/>
  </asset>
  <default>
    <default class="visual"><geom contype="0" conaffinity="0" group="1"{visual_default_mass}/></default>
    <default class="collision"><geom contype="1" conaffinity="1" group="0" density="{density}" friction="{friction}" solref="{solref}" solimp="{solimp}"/></default>
    <default class="region"><geom contype="0" conaffinity="0" group="1" rgba="0 1 0 0" mass="0"/></default>
  </default>
  <worldbody><body><body name="object">
{visual_and_collision}
    <geom name="reg_bbox" class="region" type="box" pos="0 0 0" size="{bbox}"/>
  </body></body></worldbody>
  <actuator/>
</mujoco>\n'''


def materialize_custom_asset(asset: AssetSpec) -> AssetSpec:
    """Create an exact-class MJCF body textured from the authoritative video."""
    if asset.kind != "custom_same_class":
        return asset
    model_path = Path(asset.asset_path_or_group)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    curated_texture = curated_appearance_texture_path(asset)
    texture_path = curated_texture or (model_path.parent / "source_texture.png")
    video = _source_video(asset.source_path)
    texture_ok = curated_texture is not None or _source_texture(video, texture_path)
    if not texture_ok:
        raise AssetResolutionError(f"no decodable source video for {asset.selection_id}")
    source_folder_name = Path(asset.source_path).parent.name
    if asset.asset_key in {"", asset.selection_id} and source_folder_name != asset.semantic_class:
        raise AssetResolutionError(f"source folder does not bind {asset.selection_id} to {asset.semantic_class}")
    model_text = _xml_for_asset(asset, texture_path.name)
    validate_vla82_037_drawer_asset_xml(asset, model_text)
    model_path.write_text(model_text, encoding="utf-8")
    if 'class="collision"' not in model_text:
        raise AssetResolutionError(f"collision body missing for {asset.selection_id}")
    evidence_path = model_path.parent / "asset_evidence.json"
    friction_profile = profile_for(asset.selection_id, asset.semantic_class)
    evidence_path.write_text(
        json.dumps(
            {
                "selection_id": asset.selection_id,
                "semantic_class": asset.semantic_class,
                "source_path": asset.source_path,
                "source_video": str(video.resolve()) if video else "",
                "source_video_sha256": sha256_file(video) if video else "",
                "source_frame": str(texture_path.resolve()),
                "source_frame_sha256": sha256_file(texture_path),
                "geometry": asset.geometry,
                "dimensions": list(asset.dimensions),
                "physics_profile": {"density": 65 if asset.geometry == "sponge" else 300, "friction": list(friction_profile.values), "friction_material": friction_profile.material, "friction_material_source": friction_profile.source, "friction_profile_version": friction_profile.version, "solref": [.02, 1.0], "solimp": [.88, .95, .003] if asset.geometry == "sponge" else [.9, .95, .001], "mass_basis": "18g composite sponge: 12g core + 3g per side" if asset.geometry == "sponge" else "geometry-density", "collision_model": "inner core plus two 4mm compliant side layers; no adhesion/equality constraints" if asset.geometry == "sponge" else "single collision geometry"},
                "collision_geom_count": model_text.count('class="collision"'),
                "model": str(model_path.resolve()),
                "model_sha256": sha256_file(model_path),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return replace(
        asset,
        exact_class=True,
        collision_validated=True,
        visible_validated=False,
        evidence_path=str(evidence_path.resolve()),
    )


def materialize_fixture_source_frame(asset: AssetSpec) -> dict[str, str]:
    """Extract and hash a real source-video frame for a native fixture audit."""
    if asset.kind != "fixture_part":
        raise AssetResolutionError("fixture source evidence is only valid for fixture_part assets")
    folder = CUSTOM_ASSET_ROOT / asset.selection_id
    folder.mkdir(parents=True, exist_ok=True)
    frame = folder / "fixture_source_frame.png"
    video = _source_video(asset.source_path)
    if not _source_texture(video, frame):
        raise AssetResolutionError(f"no decodable fixture source video for {asset.selection_id}")
    return {
        "source_video": str(video.resolve()) if video else "",
        "source_video_sha256": sha256_file(video) if video else "",
        "source_frame": str(frame.resolve()),
        "source_frame_sha256": sha256_file(frame),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
