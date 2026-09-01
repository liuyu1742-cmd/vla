"""Complete PandaOmron scene requests and read-only physics state extraction."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import gymnasium as gym

from tools.vla82_full_sim.annotations import OperationSpec
from tools.vla82_full_sim.assets import (
    AssetSpec, _default_geometry_and_size, _safe_asset_name, load_authoritative_manifest,
    materialize_custom_asset,
)
from tools.vla82_full_sim.friction import apply_object_friction, profile_for


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PLAN = PROJECT_ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
REQUIRED_CAMERAS = ("robot0_agentview_left", "robot0_eye_in_hand")


def vla82_trash_can_source_centers() -> tuple[tuple[float, float], ...]:
    """Reachable, separated source-can centres for the fixed VLA82-001 scene."""
    return ((-.01, .08), (.10, .08), (-.09, .16))


def vla82_trash_can_target_center() -> tuple[float, float]:
    """Table-supported XY centre for the exact fixed full-size receptacle."""
    return (.27, .42)


def vla82_trash_can_cavity_slot_centers(*, relative: bool = False) -> tuple[tuple[float, float], ...]:
    """Return the three non-overlapping can centres for the exact bin cavity.

    The relative form is the capacity contract; the absolute form is derived
    from the fixed source-faithful receptacle pose and is used by the physical
    transport expert.
    """
    offsets = ((-.05, -.045), (.05, -.045), (0., .045))
    if relative:
        return offsets
    target = np.asarray(vla82_trash_can_target_center(), dtype=float)
    return tuple(tuple(target + np.asarray(offset, dtype=float)) for offset in offsets)


def vla82_trash_can_whole_body_ik_config() -> dict[str, Any]:
    """Return a private, right-arm-only PandaOmron WholeBodyIK configuration.

    The upstream robot profile includes left-arm, head, and leg controllers.
    VLA82-001 only exposes the source task's right manipulator, torso, and
    stationary base; retaining the unrelated parts changes the public action
    surface and lets an expert accidentally command them.
    """
    config_path = PROJECT_ROOT / "third_party" / "robosuite" / "robosuite" / "controllers" / "config" / "robots" / "default_pandaomron_whole_body_ik.json"
    # Use RoboSuite's public loader so ``arms.right`` is normalized to the
    # runtime ``right`` controller part before this dict reaches Robot.reset.
    from robosuite.controllers import load_composite_controller_config
    config = copy.deepcopy(load_composite_controller_config(controller=str(config_path), robot="PandaOmron"))
    specific = config["composite_controller_specific_configs"]
    specific["ref_name"] = ["gripper0_right_grip_site"]
    specific["actuation_part_names"] = ["right"]
    specific["ik_input_ref_frame"] = "world"
    specific["ik_input_type"] = "keyboard"
    body_parts = config["body_parts"]
    config["body_parts"] = {name: body_parts[name] for name in ("right", "torso", "base")}
    return config


def vla82_open_support_controller_config() -> dict[str, Any]:
    """Return the standard 6-DoF PandaOmron action surface for tabletop skills.

    The open-support pick/place primitive emits Cartesian delta commands in
    RoboCasa's normal right-arm layout.  The trash-can task is exceptional: it
    deliberately uses its own WholeBodyIK action builder.  Reusing that
    7-DoF WholeBodyIK surface for ordinary tabletop placement makes a valid
    primitive fail before its first simulator step.
    """
    config_path = PROJECT_ROOT / "third_party" / "robosuite" / "robosuite" / "controllers" / "config" / "robots" / "default_pandaomron.json"
    from robosuite.controllers import load_composite_controller_config
    return copy.deepcopy(load_composite_controller_config(controller=str(config_path), robot="PandaOmron"))


def work_study_source_ranges(selection_id: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a reachable, semantically valid desktop source patch."""
    if str(selection_id) == "VLA82-053":
        # Keep the mouse separate from the keyboard while using the short carry
        # segment proven to retain opposed finger contact continuously.
        return ((.05, .06), (.035, .045))
    return ((.13, .15), (.10, .12))


def fixed_support_target_position(target_name: str) -> tuple[float, float, float]:
    """Place fixed receptacles in reachable, physically valid table regions."""
    positions = {
        "mouthwash_cup": (.14, -.12, .868),
        "pencil_case": (.14, -.12, .843),
        # The shelf remains distinct from the source patch while avoiding the
        # wrist-limit pose at the table's far edge.
        "bathroom_shelf": (.14, -.06, .815),
        # Bedroom furniture stands directly on the floor. The bed top is
        # z=.72 and the adjacent cabinet top is z=.68, matching ordinary
        # mattress / bedside-cabinet proportions and Panda reachability.
        "bed": (.00, .02, .360),
        "nightstand": (.355, .020, .340),
    }
    return positions.get(str(target_name), (0., -.11, .806))

# Task-2 native task classes encode the source and destination fixture roles.
# This is deliberately a closed registry: rollout capture must not select an
# arbitrary kitchen geom merely because it is visible in the scene.
TASK2_FIXTURE_SEMANTICS: dict[str, tuple[str, str, str]] = {
    "PickPlaceCabinetToCounter": ("cabinet", "counter", "on"),
    "PickPlaceCounterToCabinet": ("counter", "cabinet", "inside"),
    "PickPlaceCounterToDrawer": ("counter", "drawer", "inside"),
    "PickPlaceDrawerToCounter": ("drawer", "counter", "on"),
    "PickPlaceCounterToSink": ("counter", "sink", "inside"),
    "PickPlaceSinkToCounter": ("sink", "counter", "on"),
    "PickPlaceFridgeDrawerToShelf": ("fridge_drawer", "fridge_shelf", "on"),
}

# The source videos label these four work-study operations as placement on a
# desk / beside a display.  RoboCasa's kitchen-only task registry has no desk
# fixture, so an open physical counter support is the faithful simulator
# analogue.  Treating them as cabinet insertion makes the requested operation
# geometrically and semantically wrong.
WORK_STUDY_OPEN_SUPPORT_SELECTIONS = frozenset({
    "VLA82-046", "VLA82-050", "VLA82-052", "VLA82-053",
})
NOTEBOOK_STACK_SELECTIONS = frozenset({"VLA82-048"})
BEDSIDE_BOOK_PLACE_SELECTIONS = frozenset({"VLA82-016"})
PENCIL_CASE_INSERT_SELECTIONS = frozenset({"VLA82-051"})
TOOTHBRUSH_CUP_INSERT_SELECTIONS = frozenset({"VLA82-058"})
BATHROOM_SHELF_PLACE_SELECTIONS = frozenset({"VLA82-055", "VLA82-057"})

TARGET_PART_TOKENS: dict[str, tuple[str, ...]] = {
    "counter": ("top",),
    "drawer": ("bottom", "interior"),
    "cabinet": ("shelf", "interior"),
    "sink": ("bottom", "basin"),
    "fridge_shelf": ("shelf",),
}

# Selection-specific fixture roles from the authoritative operation labels.
# This avoids treating any kitchen fixture as proof of the requested operation.
FIXTURE_TARGETS = {
    "VLA82-007": ("stove", "knob"), "VLA82-008": ("stove", "door"),
    "VLA82-009": ("toasteroven", "rack"), "VLA82-010": ("stove", "rack"),
    "VLA82-026": ("fridge", "shelf"), "VLA82-027": ("fridge", "drawer"),
    "VLA82-028": ("fridge", "door"), "VLA82-029": ("microwave", "door"),
    "VLA82-030": ("drawer", "door"), "VLA82-031": ("cabinet", "door"),
    "VLA82-032": ("dishwasher", "rack"), "VLA82-033": ("dishwasher", "door"),
    "VLA82-034": ("stove", "door"), "VLA82-035": ("toasteroven", "door"),
}

# These close actions are evaluated from a visibly open, physically normal
# starting pose.  A fully-open appliance door can place its handle behind the
# cabinet edge, turning a manipulation test into a long collision-avoidance
# navigation task before the actual closing primitive starts.
PARTIALLY_OPEN_CLOSE_SELECTIONS = frozenset({
    "VLA82-008", "VLA82-029", "VLA82-031", "VLA82-033", "VLA82-034",
})


class EnvironmentValidationError(RuntimeError):
    """Raised when a requested scene lacks full robot, cameras, or exact objects."""


@contextmanager
def seeded_rng_scope(seed: int):
    """Seed Python/legacy NumPy construction paths and restore callers' state.

    RoboCasa owns a Generator for task choices, but its model/layout helpers
    also use Python's and NumPy's legacy global generators during construction.
    A fresh-process scene therefore must bracket *both* create and reset with
    this scope to make a supplied episode seed meaningful.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclass(frozen=True)
class SceneRequest:
    selection_id: str
    task_class: str
    robot_name: str
    camera_names: tuple[str, ...]
    primary_asset: AssetSpec
    manipulated_objects: tuple[str, ...]
    object_assets: tuple[AssetSpec, ...]
    fixture_requirements: tuple[str, ...]
    forbidden_direct_state_writes: bool = True
    # Explicit Task-2 source/target semantics, retained in the request passed
    # to live scene assembly.  Empty values mean no ordinary-object target may
    # be inferred and therefore cause capture to fail closed.
    source_fixture: str = ""
    target_fixture: str = ""
    target_relation: str = ""


def projected_counter_object_center(
    *,
    robot_base_world: Sequence[float],
    counter_world: Sequence[float],
    counter_yaw: float,
    reset_region_offset: Sequence[float],
    reset_region_size: Sequence[float],
    object_footprint: Sequence[float],
    edge_clearance: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a robot base onto a counter's local safe object-centre region.

    ``reset_region_*`` uses RoboCasa fixture-local coordinates.  The returned
    centre is therefore directly usable as a reset-time sampler target while
    ``lower`` / ``upper`` provide auditable physical bounds for the object
    centre (object half extent plus clearance remains inside the counter).
    """
    base = np.asarray(robot_base_world, dtype=float)[:2]
    counter = np.asarray(counter_world, dtype=float)[:2]
    region_offset = np.asarray(reset_region_offset, dtype=float)[:2]
    region_size = np.asarray(reset_region_size, dtype=float)[:2]
    footprint = np.asarray(object_footprint, dtype=float)[:2]
    if np.any(region_size <= 0.0) or np.any(footprint <= 0.0):
        raise EnvironmentValidationError("counter reset region and object footprint must be positive")
    rotation = np.array(
        [[np.cos(counter_yaw), -np.sin(counter_yaw)], [np.sin(counter_yaw), np.cos(counter_yaw)]],
        dtype=float,
    )
    projected_local = rotation.T @ (base - counter)
    safe_half_extent = region_size / 2.0 - footprint / 2.0 - float(edge_clearance)
    if np.any(safe_half_extent <= 0.0):
        raise EnvironmentValidationError("object footprint does not fit inside counter reset region with clearance")
    lower = region_offset - safe_half_extent
    upper = region_offset + safe_half_extent
    return np.clip(projected_local, lower, upper), lower, upper


def native_asset_reset_footprint(asset: AssetSpec) -> tuple[float, float]:
    """Return the physical counter footprint that the reset sampler must fit."""
    if asset.selection_id == "VLA82-015":
        # The centred pan spans 350 mm including its handle; the source record
        # stores only the 210 mm circular body.  Reset must fit the whole pan.
        return (.35, .21)
    fallback = tuple(float(value) for value in asset.dimensions[:2])
    if asset.kind != "robocasa_native":
        return fallback
    model_path = Path(asset.asset_path_or_group)
    try:
        root = ET.parse(model_path).getroot()
        bbox = next(
            geom for geom in root.iter("geom")
            if geom.get("name") == "reg_bbox" and geom.get("type") == "box"
        )
        half_extents = np.fromstring(bbox.attrib["size"], sep=" ", dtype=float)
        if half_extents.shape[0] >= 2 and np.all(half_extents[:2] > 0.0):
            return tuple(float(value) for value in 2.0 * half_extents[:2])
    except (ET.ParseError, FileNotFoundError, KeyError, StopIteration, ValueError):
        pass
    return fallback


def counter_source_placement_size(
    selection_id: str, object_footprint: Sequence[float],
) -> tuple[float, float]:
    """Choose a reset sampler extent after a safe counter centre is known."""
    if str(selection_id) == "VLA82-015":
        return (0.0, 0.0)
    return tuple(float(value) for value in object_footprint[:2])


def uses_projected_counter_source_placement(selection_id: str) -> bool:
    """Use the collision-audited front counter region for low-reach sources."""
    return str(selection_id) in {"VLA82-002", "VLA82-003", "VLA82-015", "VLA82-040"}


def source_object_rotation_override(selection_id: str, default_rotation: Any) -> Any:
    """Align a rectangular carton so opposing gripper pads meet broad faces."""
    return 0.0 if str(selection_id) in {"VLA82-018", "VLA82-055"} else default_rotation


def drawer_source_placement_override(selection_id: str, placement: dict[str, Any]) -> dict[str, Any]:
    """Keep VLA82-037 inside the drawer but within the collision-free arm workspace.

    The native centre/back sample forces an object-aligned Omron base through
    the adjacent cabinet.  Sampling the tongs at the normal front-left drawer
    area preserves the task semantics while allowing RoboCasa's own safe base
    anchor to reach the grasp.
    """
    if str(selection_id) != "VLA82-037":
        return placement
    adjusted = copy.deepcopy(placement)
    adjusted["pos"] = (0.0, 0.40)
    adjusted["size"] = (0.22, 0.08)
    adjusted["offset"] = (-0.12, 0.0)
    return adjusted


@dataclass(frozen=True)
class ContactEvidence:
    """A contact bound to named scene entities, never inferred from text tokens."""

    actor: str
    counterpart: str
    object_id: str | None
    target_id: str | None
    fixture_id: str | None
    joint_id: str | None
    distance: float


@dataclass(frozen=True)
class TargetGeometry:
    """Physical target contract supplied by scene assembly for one operation."""

    target_id: str
    fixture_id: str
    min_corner: tuple[float, float, float] | np.ndarray
    max_corner: tuple[float, float, float] | np.ndarray
    support_id: str
    spatial_relation: str


@dataclass(frozen=True)
class PhysicsCaptureContract:
    """Scene-assembly bindings required to turn raw MuJoCo contacts into evidence.

    No predicate guesses identifiers from string fragments: rollout assembly must
    provide the exact geoms/joints selected for the source operation.
    """

    object_geom_names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    target_geom_names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    target_geometries: dict[str, TargetGeometry] = field(default_factory=dict)
    fixture_contact_joints: dict[str, str] = field(default_factory=dict)
    fixture_joint_ids: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicsSnapshot:
    step: int
    robot_qpos: np.ndarray
    gripper_qpos: np.ndarray
    body_poses: dict[str, np.ndarray]
    joint_positions: dict[str, float]
    contacts: tuple[tuple[str, str, float], ...]
    dirt_fraction: float
    spray_coverage: float
    dispensed_amount: float
    # The legacy ``contacts`` tuple remains raw MuJoCo provenance.  Predicates
    # must use this structured channel, which binds every contact to exact
    # object/target/fixture/joint identities established by scene assembly.
    contact_evidence: tuple[ContactEvidence, ...] = ()
    target_geometries: dict[str, TargetGeometry] = field(default_factory=dict)
    # Required joint identity by source phase (for example, the selected
    # drawer's joint, not an arbitrary same-class drawer in the scene).
    fixture_joint_ids: dict[str, str] = field(default_factory=dict)


def _mapping_for(selection_id: str) -> Mapping[str, Any]:
    payload = json.loads(MAPPING_PLAN.read_text(encoding="utf-8"))
    for mapping in payload["mappings"]:
        if mapping["selection_id"] == selection_id:
            return mapping
    raise EnvironmentValidationError(f"selection is absent from mapping plan: {selection_id}")


def _fixture_requirements(spec: OperationSpec) -> tuple[str, ...]:
    phases = set(spec.phases)
    requirements = {"work_surface"}
    if phases & {"wipe", "scrub", "spray"}:
        requirements.add("cleaning_surface")
    if phases & {"insert", "deposit"}:
        requirements.add("receptacle")
    if spec.selection_id == "VLA82-001" or "垃圾桶" in spec.object_name:
        requirements.add("trash_receptacle")
    if phases & {"push", "pull"}:
        requirements.add("slider")
    if "turn_knob" in phases:
        requirements.add("knob")
    if phases & {"open", "close", "close_lid"}:
        requirements.add("hinge")
    if "press" in phases:
        requirements.add("pressable_control")
    return tuple(sorted(requirements))


_ENGLISH_OBJECTS = (
    ("can of soda", "汽水罐"),
    ("sandal", "凉鞋"),
    ("paperback book", "平装书"),
    ("book", "书籍"),
    ("folder", "文件夹"),
    ("pencil case", "笔盒"),
    ("pencil", "铅笔"),
    ("pen", "笔"),
    ("stapler", "订书机"),
    ("laptop", "笔记本电脑"),
    ("keyboard", "键盘"),
    ("mouse", "鼠标"),
    ("toothbrush", "牙刷"),
    ("toothpaste", "牙膏"),
    ("soap dispenser", "皂液器"),
)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "both": 2}


def authoritative_manipulated_objects(spec: OperationSpec) -> tuple[str, ...]:
    """Recover all explicitly named objects from source JSON, not task proxies."""
    texts = [spec.operation_text]
    folder = Path(spec.source_path).parent
    for filename in ("operation.json", "instruction.json", "source_manifest.json"):
        path = folder / filename
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
    text = "\n".join(texts).lower()
    objects = list(spec.manipulated_objects or (spec.object_name,))
    for phrase, canonical in _ENGLISH_OBJECTS:
        match = re.search(rf"(?:(one|two|three|four|five|six|both)\s+)?{re.escape(phrase)}s?\b", text)
        if not match:
            continue
        count = _COUNT_WORDS.get(match.group(1) or "one", 1)
        # The primary source object is already represented; every additional
        # explicitly quantified occurrence must become its own scene asset.
        existing = sum(item == canonical for item in objects)
        objects.extend([canonical] * max(0, count - existing))
    return tuple(objects)


def _auxiliary_assets(spec: OperationSpec, primary: AssetSpec, objects: tuple[str, ...]) -> tuple[AssetSpec, ...]:
    assets = [primary]
    for index, object_name in enumerate(objects[1:], start=1):
        key = f"{spec.selection_id}__aux_{index:02d}"
        geometry, dimensions = _default_geometry_and_size(object_name)
        assets.append(
            AssetSpec(
                selection_id=spec.selection_id,
                semantic_class=object_name,
                kind="custom_same_class",
                asset_path_or_group=str(
                    PROJECT_ROOT / "assets" / "vla82_source_textured" / spec.selection_id / f"aux-{index:02d}" / "model.xml"
                ),
                exact_class=False,
                collision_validated=False,
                visible_validated=False,
                affordances=tuple(spec.phases),
                source_path=spec.source_path,
                dimensions=dimensions,
                geometry=geometry,
                asset_key=key,
            )
        )
    return tuple(assets)


def build_scene_request(spec: OperationSpec, asset: AssetSpec) -> SceneRequest:
    if asset.selection_id != spec.selection_id or asset.semantic_class != spec.object_name:
        raise EnvironmentValidationError("scene primary asset does not match source operation")
    if asset.semantic_class != spec.object_name:
        raise EnvironmentValidationError("scene request semantic class does not match the authoritative source")
    # The operation spec is compiled from the selected object's real-video
    # annotation and is authoritative for this episode.  Legacy asset-catalog
    # affordances describe reusable geometry and may contain generic labels
    # such as ``composite`` / ``grasp`` (notably the oven door).  Carrying
    # those labels into the physics contract drops the real close/open joint
    # phase and makes genuine fixture motion impossible to verify.
    asset = replace(asset, affordances=tuple(spec.phases))
    mapping = _mapping_for(spec.selection_id)
    # Task 2's native-furniture mappings (stove knob, appliance door, rack,
    # etc.) are operated fixture parts.  Their source label is still retained
    # in ``asset`` for manifest validation, but injecting it as a free custom
    # object would both break the native task constructor and falsify the
    # contact/joint provenance used by the evaluator.
    if str(mapping.get("mapping_mode", "")) == "native_furniture_task":
        asset = replace(asset, kind="fixture_part", exact_class=True)
    task_class = "VLA82TrashCanDeposit" if spec.selection_id == "VLA82-001" else str(mapping["task_class"])
    if spec.selection_id == "VLA82-003":
        # The source label is a dish-brush countertop cleaning operation.  The
        # legacy cabinet-to-counter mapping put both the brush and clean region
        # above the Panda's usable workspace; use the native counter source so
        # grasp, scrub coverage, and return remain physically executable.
        task_class = "PickPlaceCounterToCabinet"
    if spec.selection_id in WORK_STUDY_OPEN_SUPPORT_SELECTIONS:
        task_class = "VLA82WorkStudyOpenSupport"
    if spec.selection_id in NOTEBOOK_STACK_SELECTIONS:
        task_class = "VLA82NotebookStack"
    if spec.selection_id in BEDSIDE_BOOK_PLACE_SELECTIONS:
        task_class = "VLA82BedsideBookPlace"
    if spec.selection_id in PENCIL_CASE_INSERT_SELECTIONS:
        task_class = "VLA82PencilCaseInsert"
    if spec.selection_id in TOOTHBRUSH_CUP_INSERT_SELECTIONS:
        task_class = "VLA82ToothbrushCupInsert"
    if spec.selection_id in BATHROOM_SHELF_PLACE_SELECTIONS:
        task_class = "VLA82BathroomShelfPlace"
    source_fixture, target_fixture, target_relation = TASK2_FIXTURE_SEMANTICS.get(task_class, ("", "", ""))
    if spec.selection_id in WORK_STUDY_OPEN_SUPPORT_SELECTIONS:
        source_fixture, target_fixture, target_relation = ("table", "table", "on")
    if spec.selection_id in NOTEBOOK_STACK_SELECTIONS:
        source_fixture, target_fixture, target_relation = ("table", "folder", "on")
    if spec.selection_id in BEDSIDE_BOOK_PLACE_SELECTIONS:
        source_fixture, target_fixture, target_relation = ("bed", "nightstand", "on")
    if spec.selection_id in PENCIL_CASE_INSERT_SELECTIONS:
        source_fixture, target_fixture, target_relation = ("table", "pencil_case", "inside")
    if spec.selection_id in TOOTHBRUSH_CUP_INSERT_SELECTIONS:
        source_fixture, target_fixture, target_relation = ("table", "mouthwash_cup", "inside")
    if spec.selection_id in BATHROOM_SHELF_PLACE_SELECTIONS:
        source_fixture, target_fixture, target_relation = ("table", "bathroom_shelf", "on")
    if task_class == "VLA82TrashCanDeposit":
        # Source video: three cans are manipulated independently and must end
        # inside the source-textured trash-can cavity.
        source_fixture, target_fixture, target_relation = ("cans", "trash_can", "inside")
    manifest = load_authoritative_manifest()[spec.selection_id]
    profile = manifest.get("object_profile", {})
    if tuple(profile.get("dimensions", ())) != tuple(asset.dimensions) or profile.get("geometry") != asset.geometry:
        raise EnvironmentValidationError("runtime asset does not match authoritative object profile")
    if task_class == "VLA82TrashCanDeposit":
        # Use the original source-textured full-size five-wall receptacle.  The
        # compact construction variant cannot physically contain all three
        # complete source cans and therefore cannot satisfy the video task.
        expected_path = PROJECT_ROOT / "assets" / "vla82_source_textured" / spec.selection_id / "model.xml"
        if Path(asset.asset_path_or_group).resolve() != expected_path.resolve():
            raise EnvironmentValidationError("trash-can target must use the original full-size source asset")
    manipulated_objects = tuple(str(item) for item in manifest.get("manipulated_objects", ()))
    if not manipulated_objects:
        raise EnvironmentValidationError("authoritative manifest does not name manipulated objects")
    object_assets = _auxiliary_assets(spec, asset, manipulated_objects)
    if spec.selection_id in BEDSIDE_BOOK_PLACE_SELECTIONS:
        bed = AssetSpec(
            selection_id=spec.selection_id,
            semantic_class="床",
            kind="custom_same_class",
            asset_path_or_group=str(
                PROJECT_ROOT / "assets" / "vla82_source_textured" / spec.selection_id
                / "bed" / "model.xml"
            ),
            exact_class=True,
            collision_validated=True,
            visible_validated=True,
            affordances=("support",),
            source_path=spec.source_path,
            dimensions=(.240, .420, .360),
            geometry="bed",
            asset_key=f"{spec.selection_id}__bed",
        )
        nightstand = AssetSpec(
            selection_id=spec.selection_id,
            semantic_class="床头柜",
            kind="custom_same_class",
            asset_path_or_group=str(
                PROJECT_ROOT / "assets" / "vla82_source_textured" / spec.selection_id
                / "nightstand" / "model.xml"
            ),
            exact_class=True,
            collision_validated=True,
            visible_validated=True,
            affordances=("support",),
            source_path=spec.source_path,
            dimensions=(.110, .100, .340),
            geometry="nightstand",
            asset_key=f"{spec.selection_id}__nightstand",
        )
        manipulated_objects = (spec.object_name,)
        object_assets = (asset, bed, nightstand)
    if spec.selection_id in NOTEBOOK_STACK_SELECTIONS:
        folder = next(
            (item for item in object_assets if item.semantic_class == "文件夹"), None,
        )
        if folder is None:
            raise EnvironmentValidationError("VLA82-048 source annotation has no folder target")
        folder = replace(
            folder,
            dimensions=(.070, .045, .006),
            geometry="folder",
            affordances=("support",),
        )
        manipulated_objects = (spec.object_name, "文件夹")
        object_assets = (asset, folder)
    if spec.selection_id in PENCIL_CASE_INSERT_SELECTIONS:
        pencil_case = next(
            (item for item in object_assets if item.semantic_class == "笔盒"), None,
        )
        if pencil_case is None:
            raise EnvironmentValidationError("VLA82-051 source annotation has no pencil-case target")
        pencil_case = replace(
            pencil_case,
            geometry="open_receptacle",
            # A normal 230 x 120 mm case leaves enough interior length for
            # the 190-mm source pencil after accounting for physical walls.
            dimensions=(.060, .115, .035),
            affordances=("contain",),
        )
        manipulated_objects = (spec.object_name, "笔盒")
        object_assets = (asset, pencil_case)
    if spec.selection_id in TOOTHBRUSH_CUP_INSERT_SELECTIONS:
        mouthwash_cup = AssetSpec(
            selection_id=spec.selection_id,
            semantic_class="漱口杯",
            kind="custom_same_class",
            asset_path_or_group=str(
                PROJECT_ROOT / "assets" / "vla82_source_textured" / spec.selection_id
                / "mouthwash_cup" / "model.xml"
            ),
            exact_class=True,
            collision_validated=True,
            visible_validated=False,
            affordances=("contain",),
            source_path=spec.source_path,
            dimensions=(.045, .045, .060),
            geometry="open_receptacle",
            asset_key=f"{spec.selection_id}__mouthwash_cup",
        )
        manipulated_objects = (spec.object_name, "漱口杯")
        object_assets = (asset, mouthwash_cup)
    if spec.selection_id in BATHROOM_SHELF_PLACE_SELECTIONS:
        bathroom_shelf = AssetSpec(
            selection_id=spec.selection_id,
            semantic_class="卫浴搁板",
            kind="custom_same_class",
            asset_path_or_group=str(
                PROJECT_ROOT / "assets" / "vla82_source_textured" / "VLA82-054" / "model.xml"
            ),
            exact_class=True,
            collision_validated=True,
            visible_validated=True,
            affordances=("support",),
            source_path=spec.source_path,
            dimensions=(.120, .070, .015),
            geometry="fixture",
            asset_key=f"{spec.selection_id}__bathroom_shelf",
        )
        manipulated_objects = (spec.object_name,)
        object_assets = (asset, bathroom_shelf)
    return SceneRequest(
        selection_id=spec.selection_id,
        task_class=task_class,
        robot_name="PandaOmron",
        camera_names=REQUIRED_CAMERAS,
        primary_asset=asset,
        manipulated_objects=manipulated_objects,
        object_assets=object_assets,
        fixture_requirements=_fixture_requirements(spec),
        forbidden_direct_state_writes=True,
        source_fixture=source_fixture,
        target_fixture=target_fixture,
        target_relation=target_relation,
    )


def _register_custom_category(asset: AssetSpec) -> str:
    """Register one project-local category so RoboCasa loads its real MJCF model."""
    from robocasa.models.objects.kitchen_object_utils import ObjCat
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS

    materialize_custom_asset(asset)
    category = _safe_asset_name(asset.asset_key or asset.selection_id)
    asset_path = Path(asset.asset_path_or_group).resolve()
    category_root = asset_path.parent.parent
    selected_folder = asset_path.parent.name
    excluded_folders = tuple(
        candidate.name for candidate in category_root.iterdir()
        if candidate.is_dir() and candidate.name != selected_folder
    )
    expected_path = str(asset_path)
    existing = OBJ_CATEGORIES.get(category, {}).get("vla82_custom") if category in OBJ_CATEGORIES else None
    # ObjCat samples every subdirectory of model_folders.  Registering the
    # shared VLA82 root without exclusions silently replaces a selected sponge
    # with an unrelated random VLA object, so exact source identity must be
    # rechecked even when this process registered the category earlier.
    if existing is None or tuple(map(str, getattr(existing, "mjcf_paths", ())) ) != (expected_path,):
        OBJ_CATEGORIES[category] = {
            "vla82_custom": ObjCat(
                name=category,
                types=("tool",),
                model_folders=[str(category_root)],
                exclude=list(excluded_folders),
                graspable=True,
                washable=True,
                reg_type="vla82_custom",
            )
        }
    OBJ_GROUPS[category] = [category]
    return category


def required_task_auxiliary_object_names(task_class: str) -> tuple[str, ...]:
    """Keep native structural objects referenced unconditionally by reward code."""
    return ("container",) if str(task_class) == "PickPlaceSinkToCounter" else ()


def uses_source_fixture_reset_anchor(request: SceneRequest | Any) -> bool:
    """Anchor selected navigation-free tests at their sampled source fixture."""
    return str(getattr(request, "selection_id", "")) in {
        # Native counter-to-cabinet source objects.  Anchor the mobile base
        # at the sampler-selected source fixture so these tests validate
        # manipulation rather than an unrelated random kitchen aisle.
        "VLA82-012", "VLA82-013", "VLA82-020", "VLA82-046", "VLA82-050", "VLA82-052", "VLA82-053",
        "VLA82-056", "VLA82-058", "VLA82-059",
        "VLA82-015", "VLA82-017", "VLA82-018", "VLA82-021", "VLA82-037",
    }


def _customize_object_configs(raw: Any, categories: tuple[str, ...], request: SceneRequest) -> None:
    """Replace a task's target with the exact asset without pose/state writes."""
    original = raw._get_obj_cfgs

    def configured(self: Any) -> list[dict[str, Any]]:
        configs = copy.deepcopy(original())
        target_index = next((i for i, cfg in enumerate(configs) if cfg.get("name") == "obj"), 0)
        if not configs:
            raise EnvironmentValidationError("task did not define an object placement contract")
        configs[target_index]["obj_groups"] = categories[0]
        configs[target_index]["exclude_obj_groups"] = None
        configs[target_index]["graspable"] = True
        configs[target_index]["placement"] = drawer_source_placement_override(
            request.selection_id, configs[target_index]["placement"],
        )
        # The acceptance scope is source-labelled object manipulation.  Keep
        # the mobile base at the source fixture selected by RoboCasa's own
        # reset logic so an unrelated random kitchen aisle or closed cabinet
        # door cannot turn a grasp/place test into a navigation test.  This
        # runs before reset, changes no MuJoCo state, and preserves the
        # authoritative source / target placement contracts.
        if uses_source_fixture_reset_anchor(request):
            source_fixture = configs[target_index].get("placement", {}).get("fixture")
            source_name = getattr(source_fixture, "name", "")
            if not source_name:
                raise EnvironmentValidationError(
                    f"source fixture unavailable for reset anchor: {request.selection_id}"
                )
            self.init_robot_base_ref = source_name
            self.robot_spawn_deviation_pos_x = 0.0
            self.robot_spawn_deviation_pos_y = 0.0
            self.robot_spawn_deviation_rot = 0.0
            self._vla82_robot_spawn_diagnostics = {
                "selection_id": request.selection_id,
                "mode": "source_fixture_anchor",
                "source_fixture": source_name,
                "task_initially_solved": False,
            }
        if request.selection_id == "VLA82-039":
            # The source record is a long wooden spoon in a drawer.  Align it
            # with the drawer depth before reset: a random diagonal pose
            # leaves no finger clearance beside either inner side wall.
            configs[target_index]["placement"]["rotation"] = np.pi / 2.0
        configs[target_index]["placement"]["rotation"] = source_object_rotation_override(
            request.selection_id,
            configs[target_index]["placement"].get("rotation", 0.0),
        )
        if uses_projected_counter_source_placement(request.selection_id):
            # These source objects are sampled at the closest physically safe
            # point of the live counter top. This changes only RoboCasa's
            # pre-reset sampler; no post-reset pose is written.
            placement = configs[target_index]["placement"]
            counter = placement["fixture"]
            # Object configs are built before robosuite creates a live robot
            # body.  Use the same RoboCasa base-placement routine that later
            # installs the robot, anchored to this source counter.
            from robocasa.utils.env_utils import compute_robot_base_placement_pose

            self.init_robot_base_ref = counter.name
            robot_base, _ = compute_robot_base_placement_pose(self, ref_fixture=counter)
            robot_base = np.asarray(robot_base, dtype=float)
            object_footprint = native_asset_reset_footprint(request.primary_asset)
            # Ask the counter for all physical top geoms and select the
            # reachable one, rather than treating its full fixture bounds as
            # a single continuous surface.
            regions = counter.get_reset_regions(
                env=self,
                top_size=tuple(value + 0.04 for value in object_footprint),
            )
            local_to_world = np.array(
                [[np.cos(counter.rot), -np.sin(counter.rot)], [np.sin(counter.rot), np.cos(counter.rot)]],
                dtype=float,
            )
            candidates: list[tuple[float, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
            for region in regions.values():
                center, lower, upper = projected_counter_object_center(
                    robot_base_world=robot_base,
                    counter_world=counter.pos,
                    counter_yaw=float(counter.rot),
                    reset_region_offset=region["offset"],
                    reset_region_size=region["size"],
                    object_footprint=object_footprint,
                )
                candidate_world = np.asarray(counter.pos, dtype=float)[:2] + local_to_world @ center
                candidates.append((float(np.linalg.norm(candidate_world - robot_base[:2])), region, center, lower, upper))
            if not candidates:
                raise EnvironmentValidationError("counter has no sponge-sized physical top region")
            _, reset_region, center, lower, upper = min(candidates, key=lambda item: item[0])
            # A near-point sampler preserves the projected target while the
            # explicit clearance in ``projected_counter_object_center`` keeps
            # the complete physical source object within the reset region.
            placement["sample_region_kwargs"] = {}
            placement["size"] = counter_source_placement_size(
                request.selection_id, object_footprint,
            )
            placement["pos"] = (0.0, 0.0)
            placement["offset"] = tuple(float(center[i] - reset_region["offset"][i]) for i in range(2))
            placement["margin"] = 0.0
            placement["rotation"] = 0.0
            configs[target_index]["reset_region"] = copy.deepcopy(reset_region)
            sampled_target_world = np.asarray(counter.pos, dtype=float)[:2] + local_to_world @ center
            self._vla82_placement_diagnostics = {
                "selection_id": request.selection_id,
                "robot_base_world": robot_base[:2].tolist(),
                "counter_world": np.asarray(counter.pos, dtype=float)[:2].tolist(),
                "computed_local_offset": center.tolist(),
                "counter_lower_bound": lower.tolist(),
                "counter_upper_bound": upper.tolist(),
                "world_object_base_target_distance": float(np.linalg.norm(sampled_target_world - robot_base[:2])),
            }
        annotated = [configs[target_index]]
        required_auxiliary = set(required_task_auxiliary_object_names(request.task_class))
        annotated.extend(
            copy.deepcopy(config)
            for index, config in enumerate(configs)
            if index != target_index and str(config.get("name", "")) in required_auxiliary
        )
        # Every object named by authoritative source JSON is injected as a
        # distinct MuJoCo object with its own source-backed asset category.
        for index, category in enumerate(categories[1:], start=1):
            extra = copy.deepcopy(configs[target_index])
            extra["name"] = f"annotated_{index:02d}"
            extra["obj_groups"] = category
            extra["placement"] = copy.deepcopy(extra["placement"])
            extra["placement"]["offset"] = (0.03 * index, 0.04 * index)
            annotated.append(extra)
        return annotated

    raw._get_obj_cfgs = MethodType(configured, raw)


def source_cabinet_robot_anchor(
    *, cabinet_center: Sequence[float], native_anchor: Sequence[float],
    object_center: Sequence[float], normal_standoff: float = .56,
) -> np.ndarray:
    """Align a reset-time mobile-base anchor with an object inside a cabinet.

    RoboCasa's native anchor selects the accessible side of the source fixture.
    Preserve that outward direction and only (1) use the collision-audited Panda
    standoff and (2) align along the cabinet front with the sampled object.
    """
    cabinet = np.asarray(cabinet_center, dtype=float)
    native = np.asarray(native_anchor, dtype=float)
    obj = np.asarray(object_center, dtype=float)
    outward = native[:2] - cabinet[:2]
    norm = float(np.linalg.norm(outward))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise EnvironmentValidationError("cabinet robot anchor has no outward direction")
    outward /= norm
    tangent = np.array((-outward[1], outward[0]), dtype=float)
    tangential_offset = float(np.dot(obj[:2] - cabinet[:2], tangent))
    result = native.copy()
    result[:2] = cabinet[:2] + outward * float(normal_standoff) + tangent * tangential_offset
    return result


def counter_source_normal_standoff(selection_id: str) -> float:
    """Use the measured collision-free counter boundary for the VLA82-020 mug."""
    return .45 if str(selection_id) == "VLA82-020" else .56


def source_drawer_robot_anchor(
    *, drawer_center: Sequence[float], native_anchor: Sequence[float],
    object_center: Sequence[float], slide_axis_world: Sequence[float],
) -> np.ndarray:
    """Align across drawer width while preserving the native front standoff."""
    drawer = np.asarray(drawer_center, dtype=float)
    native = np.asarray(native_anchor, dtype=float)
    obj = np.asarray(object_center, dtype=float)
    slide = np.asarray(slide_axis_world, dtype=float)[:2]
    norm = float(np.linalg.norm(slide))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise EnvironmentValidationError("drawer slide axis has no horizontal direction")
    slide /= norm
    tangent = np.array((-slide[1], slide[0]), dtype=float)
    native_slide = float(np.dot(native[:2] - drawer[:2], slide))
    object_tangent = float(np.dot(obj[:2] - drawer[:2], tangent))
    result = native.copy()
    result[:2] = drawer[:2] + slide * native_slide + tangent * object_tangent
    return result


def source_drawer_native_anchor_is_reachable(
    selection_id: str, *, native_base: Sequence[float], object_center: Sequence[float],
    max_distance: float = .60,
) -> bool:
    """Use RoboCasa's safe anchor when the drawer-open tongs will be reachable.

    This gate runs before reset settling completes, while the recorded source
    pose is still up to one drawer travel (0.30 m) behind its first-step pose.
    """
    return (
        str(selection_id) == "VLA82-037"
        and float(np.linalg.norm(
            np.asarray(native_base, dtype=float)[:2]
            - np.asarray(object_center, dtype=float)[:2]
        )) <= float(max_distance)
    )


def open_drawer_robot_anchor(
    *, drawer_low: Sequence[float], drawer_high: Sequence[float],
    native_anchor: Sequence[float], standoff: float = .15,
) -> np.ndarray:
    """Place the reset base outside the open edge of the live drawer bbox."""
    low = np.asarray(drawer_low, dtype=float)
    high = np.asarray(drawer_high, dtype=float)
    native = np.asarray(native_anchor, dtype=float)
    result = native.copy()
    outward_axis = int(np.argmax(high[:2] - low[:2]))
    low_distance = abs(float(native[outward_axis] - low[outward_axis]))
    high_distance = abs(float(native[outward_axis] - high[outward_axis]))
    outward_sign = -1.0 if low_distance <= high_distance else 1.0
    edge = low[outward_axis] if outward_sign < 0 else high[outward_axis]
    result[:2] = (low[:2] + high[:2]) / 2.0
    orthogonal_axis = 1 - outward_axis
    result[orthogonal_axis] += np.sign(native[orthogonal_axis] - result[orthogonal_axis]) * .025
    result[outward_axis] = float(edge) + outward_sign * float(standoff)
    return result


def uses_cabinet_source_alignment(request: SceneRequest | Any) -> bool:
    """Select reset alignment from source semantics, not a short ID allowlist."""
    return (
        str(getattr(request, "source_fixture", "")) == "cabinet"
        or str(getattr(request, "selection_id", "")) == "VLA82-021"
    )


def uses_counter_source_alignment(request: SceneRequest | Any) -> bool:
    """Select object-aligned reset only for the recovered native counter cases."""
    return (
        str(getattr(request, "source_fixture", "")) == "counter"
        and str(getattr(request, "selection_id", "")) in {"VLA82-012", "VLA82-013", "VLA82-020", "VLA82-056", "VLA82-058", "VLA82-059"}
    )


def uses_drawer_source_alignment(request: SceneRequest | Any) -> bool:
    """Select reset alignment from authoritative drawer-source semantics."""
    return str(getattr(request, "source_fixture", "")) == "drawer"


def _configure_drawer_source_robot_spawn(raw: Any, request: SceneRequest) -> None:
    """Align the reset base across an open source drawer without approaching it."""
    if not uses_drawer_source_alignment(request):
        return
    original = raw._setup_scene

    def configured(self: Any) -> None:
        original()
        self.sim.forward()
        try:
            from robocasa.utils.env_utils import detect_robot_collision, set_robot_base, set_robot_to_position

            # At this setup hook the free object's MuJoCo body has not yet
            # received its sampled pose, so ``body_xpos`` is still zero.
            # ``object_placements`` is the authoritative pre-settle source;
            # the reach gate below accounts for the drawer's pending travel.
            object_center = np.asarray(self.object_placements["obj"][0], dtype=float)
            prefix = str(self.drawer.naming_prefix)
            joint_id = next(
                index for index in range(int(self.sim.model.njnt))
                if str(self.sim.model.joint_id2name(index) or "").startswith(prefix)
                and int(self.sim.model.jnt_type[index]) == 2
            )
            body_id = int(self.sim.model.jnt_bodyid[joint_id])
            body_rotation = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
            slide_axis_world = body_rotation @ np.asarray(self.sim.model.jnt_axis[joint_id], dtype=float)
            native_anchor = np.asarray(self.init_robot_base_pos_anchor, dtype=float).copy()
            # The nominal source-drawer anchor intersects the open drawer, so
            # RoboCasa expands its random placement range until it finds a
            # collision-free base pose.  Reproduce that search without
            # consuming RNG state, then preserve its safe slide-axis standoff
            # while correcting only the drawer-width coordinate.
            native_sim_state = self.sim.get_state()
            native_rng_state = copy.deepcopy(self.rng.bit_generator.state)
            native_base = np.asarray(
                set_robot_base(
                    env=self,
                    anchor_pos=native_anchor,
                    anchor_ori=self.init_robot_base_ori_anchor,
                    rot_dev=self.robot_spawn_deviation_rot,
                    pos_dev_x=self.robot_spawn_deviation_pos_x,
                    pos_dev_y=self.robot_spawn_deviation_pos_y,
                ),
                dtype=float,
            )
            self.sim.set_state(native_sim_state)
            self.sim.forward()
            self.rng.bit_generator.state = native_rng_state
            aligned_base = source_drawer_robot_anchor(
                drawer_center=self.drawer.pos,
                native_anchor=native_base,
                object_center=object_center,
                slide_axis_world=slide_axis_world,
            )
            slide_xy = np.asarray(slide_axis_world[:2], dtype=float)
            slide_xy /= float(np.linalg.norm(slide_xy))
            native_slide_projection = float(
                np.dot(native_base[:2] - np.asarray(self.drawer.pos, dtype=float)[:2], slide_xy)
            )
            outward_sign = -1.0 if native_slide_projection < 0.0 else 1.0
            anchor = None
            spawn_mode = "source_drawer_tangent_alignment"
            clearance_offset = None
            for offset in np.arange(0.0, 0.151, 0.005):
                candidate = aligned_base.copy()
                candidate[:2] += slide_xy * outward_sign * float(offset)
                set_robot_to_position(self, candidate)
                self.sim.forward()
                if not detect_robot_collision(self):
                    anchor = candidate
                    clearance_offset = float(offset)
                    break
            self.sim.set_state(native_sim_state)
            self.sim.forward()
            if anchor is None:
                if source_drawer_native_anchor_is_reachable(
                    request.selection_id,
                    native_base=native_base,
                    object_center=object_center,
                ):
                    anchor = native_base
                    clearance_offset = None
                    spawn_mode = "source_drawer_native_reachable"
                elif request.selection_id == "VLA82-039":
                    # The spoon's depth-aligned pose changes the safe tangent
                    # band used for tongs. Retain RoboCasa's own valid drawer
                    # reference instead of forcing that unrelated alignment.
                    anchor = native_base
                    clearance_offset = None
                else:
                    raise EnvironmentValidationError(
                        f"no collision-free source-drawer base pose: {request.selection_id}"
                    )
        except (AttributeError, KeyError, IndexError, StopIteration, TypeError, ValueError) as error:
            raise EnvironmentValidationError(
                f"cannot configure source-drawer robot spawn: {request.selection_id}"
            ) from error
        self.init_robot_base_pos_anchor = anchor
        self.robot_spawn_deviation_pos_x = 0.0
        self.robot_spawn_deviation_pos_y = 0.0
        self.robot_spawn_deviation_rot = 0.0
        self._vla82_robot_spawn_diagnostics = {
            "selection_id": request.selection_id,
            "mode": spawn_mode,
            "native_anchor": native_anchor.tolist(),
            "anchor": anchor.tolist(),
            "native_base": native_base.tolist(),
            "aligned_base": aligned_base.tolist(),
            "clearance_offset": clearance_offset,
            "object_center": object_center.tolist(),
            "slide_axis_world": slide_axis_world.tolist(),
            "native_slide_projection": native_slide_projection,
            "aligned_object_distance": float(np.linalg.norm(aligned_base[:2] - object_center[:2])),
            "task_initially_solved": False,
        }

    raw._setup_scene = MethodType(configured, raw)


def _configure_cabinet_source_robot_spawn(raw: Any, request: SceneRequest) -> None:
    """Install a source-led cabinet-front anchor before RoboCasa places the robot.

    The hook runs from ``_setup_scene`` after each hard-reset model rebuild and
    before ``set_robot_base``.  It changes no object, fixture, joint, or episode
    state and therefore cannot manufacture task success.
    """
    if not uses_cabinet_source_alignment(request):
        return
    original = raw._setup_scene

    def configured(self: Any) -> None:
        original()
        try:
            object_center = np.asarray(self.object_placements["obj"][0], dtype=float)
            if request.selection_id == "VLA82-021":
                # This custom source asset is sampled on a concrete counter
                # region after the generic fixture anchor is chosen.  Anchor
                # the base to that actual pre-reset sampled source, preserving
                # all object physics while keeping the arm in its verified
                # reach envelope.
                anchor = object_center.copy()
                anchor[0] += .42
                anchor[2] = 0.0
            else:
                anchor = source_cabinet_robot_anchor(
                    cabinet_center=self.cab.pos,
                    native_anchor=self.init_robot_base_pos_anchor,
                    object_center=object_center,
                )
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise EnvironmentValidationError(
                f"cannot configure source-cabinet robot spawn: {request.selection_id}"
            ) from error
        self.init_robot_base_pos_anchor = anchor
        self.robot_spawn_deviation_pos_x = 0.0
        self.robot_spawn_deviation_pos_y = 0.0
        self.robot_spawn_deviation_rot = 0.0
        self._vla82_robot_spawn_diagnostics = {
            "selection_id": request.selection_id,
            "mode": "source_cabinet_front_alignment",
            "anchor": anchor.tolist(),
            "object_center": object_center.tolist(),
            "task_initially_solved": False,
        }

    raw._setup_scene = MethodType(configured, raw)


def _configure_counter_source_robot_spawn(raw: Any, request: SceneRequest) -> None:
    """Align the reset base with a sampled counter object before physics begins."""
    if not uses_counter_source_alignment(request):
        return
    original = raw._setup_scene

    def configured(self: Any) -> None:
        original()
        try:
            object_center = np.asarray(self.object_placements["obj"][0], dtype=float)
            anchor = source_cabinet_robot_anchor(
                cabinet_center=self.counter.pos,
                native_anchor=self.init_robot_base_pos_anchor,
                object_center=object_center,
                normal_standoff=counter_source_normal_standoff(request.selection_id),
            )
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise EnvironmentValidationError(
                f"cannot configure source-counter robot spawn: {request.selection_id}"
            ) from error
        self.init_robot_base_pos_anchor = anchor
        self.robot_spawn_deviation_pos_x = 0.0
        self.robot_spawn_deviation_pos_y = 0.0
        self.robot_spawn_deviation_rot = 0.0
        self._vla82_robot_spawn_diagnostics = {
            "selection_id": request.selection_id,
            "mode": "source_counter_object_alignment",
            "anchor": anchor.tolist(),
            "object_center": object_center.tolist(),
            "task_initially_solved": False,
        }

    raw._setup_scene = MethodType(configured, raw)


def is_counter_to_drawer_request(request: SceneRequest | Any) -> bool:
    """Select the shared physical drawer profile from authoritative semantics."""
    return (
        str(getattr(request, "task_class", "")) == "PickPlaceCounterToDrawer"
        and str(getattr(request, "source_fixture", "")) == "counter"
        and str(getattr(request, "target_fixture", "")) == "drawer"
        and str(getattr(request, "target_relation", "")) == "inside"
    )


def _configure_vla005_robot_spawn(raw: Any, request: SceneRequest) -> None:
    """Install a collision-audited reset pose for a counter-to-drawer task."""
    if not is_counter_to_drawer_request(request):
        return
    original = raw._setup_scene

    def configured(self: Any) -> None:
        original()
        self.sim.forward()
        prefix = str(self.drawer.naming_prefix)
        bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for geom_id in range(int(self.sim.model.ngeom)):
            name = str(self.sim.model.geom_id2name(geom_id) or "")
            if not name.startswith(prefix) or not any(token in name for token in ("inner_bottom", "inner_left", "inner_right", "inner_back")):
                continue
            size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)
            rotation = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            extent = np.abs(rotation) @ size
            center = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
            bounds.append((center - extent, center + extent))
        if not bounds:
            raise EnvironmentValidationError("opened drawer exposes no inner collision geometry")
        drawer_low = np.min(np.vstack([item[0] for item in bounds]), axis=0)
        drawer_high = np.max(np.vstack([item[1] for item in bounds]), axis=0)
        anchor = open_drawer_robot_anchor(
            drawer_low=drawer_low, drawer_high=drawer_high,
            native_anchor=self.init_robot_base_pos_anchor,
        )
        self.init_robot_base_pos_anchor = anchor
        self.robot_spawn_deviation_pos_x = 0.0
        self.robot_spawn_deviation_pos_y = 0.0
        self.robot_spawn_deviation_rot = 0.0
        self._vla82_robot_spawn_diagnostics = {
            "selection_id": request.selection_id,
            "mode": "open_drawer_outer_edge",
            "anchor": anchor.tolist(),
            "task_initially_solved": False,
        }

    raw._setup_scene = MethodType(configured, raw)


def fixture_close_initial_open_fraction(selection_id: str) -> float:
    """Return the normal safe reset opening fraction for a door-close task."""
    if str(selection_id) == "VLA82-029":
        return .10
    if str(selection_id) == "VLA82-031":
        return .25
    return .32


def fixture_close_reset_hinge_damping(selection_id: str) -> float | None:
    """Return a source-fixture damper needed to hold a normal partly-open pose."""
    # The imported oven door has unit damping and swings fully open under zero
    # action. A real appliance hinge resists that free swing; this reset-time
    # damper preserves a visibly open but physically stable start state.
    if str(selection_id) == "VLA82-008":
        return 100.0
    return 20.0 if str(selection_id) == "VLA82-031" else None


def fixture_close_reset_hinge_friction(selection_id: str) -> float | None:
    """Return the static hinge friction for a stable source-fixture reset."""
    if str(selection_id) == "VLA82-008":
        return 20.0
    return 5.0 if str(selection_id) == "VLA82-031" else None


def _configure_fixture_close_partial_open_reset(raw: Any, request: SceneRequest) -> None:
    """Reset selected appliance doors to a normal, visibly open start angle.

    This hook is deliberately reset-only.  It never runs during expert action
    rollout, so success still requires physical gripper contact and an actual
    joint transition through ``env.step``.  The selected doors start partly
    open rather than fully open, which is a normal appliance configuration and
    leaves their handles in the robot's reachable working envelope.
    """
    if str(request.selection_id) not in PARTIALLY_OPEN_CLOSE_SELECTIONS:
        return
    original = raw.reset

    def configured(self: Any, *args: Any, **kwargs: Any) -> Any:
        observation = original(*args, **kwargs)
        binding = fixture_binding(self, request)
        joint_id = str(binding.get("joint", ""))
        if not joint_id:
            raise EnvironmentValidationError(
                f"close-task fixture has no reset joint: {request.selection_id}"
            )
        joint_index = int(self.sim.model.joint_name2id(joint_id))
        lower, upper = (float(value) for value in self.sim.model.jnt_range[joint_index])
        open_bound = lower if abs(lower) >= abs(upper) else upper
        # Retain a visibly open state without crossing the fixture's own
        # native limits.  The microwave's door sweeps through the Panda arm
        # at ~30 degrees in this official kitchen layout, so it starts at a
        # narrower but still visibly open angle.  This is reset-time layout
        # configuration only; all subsequent joint movement is physical.
        open_fraction = fixture_close_initial_open_fraction(request.selection_id)
        partially_open = float(open_bound) * open_fraction
        self.sim.data.set_joint_qpos(joint_id, partially_open)
        damping = fixture_close_reset_hinge_damping(request.selection_id)
        friction = fixture_close_reset_hinge_friction(request.selection_id)
        if damping is not None or friction is not None:
            dof_index = int(self.sim.model.jnt_dofadr[joint_index])
            if damping is not None:
                self.sim.model.dof_damping[dof_index] = max(
                    float(self.sim.model.dof_damping[dof_index]), float(damping),
                )
            if friction is not None:
                self.sim.model.dof_frictionloss[dof_index] = max(
                    float(self.sim.model.dof_frictionloss[dof_index]), float(friction),
                )
            self.sim.data.set_joint_qvel(joint_id, 0.0)
        if str(request.selection_id) == "VLA82-029":
            # In the native microwave layout the stock base spawn places arm
            # links inside a partly-open panel.  Shift the mobile base 45 cm
            # toward the handle-facing workspace before the episode begins.
            # This is a normal reset pose, not an in-episode motion or a
            # write to the evaluated microwave hinge.
            base_side_joint = "mobilebase0_joint_mobile_side"
            current_side = float(self.sim.data.get_joint_qpos(base_side_joint))
            self.sim.data.set_joint_qpos(base_side_joint, current_side - .45)
        self.sim.forward()
        # The upstream task computes observations before its reset method
        # returns. Recompute them after the allowed reset-only joint setup so
        # the first captured frame and the live MuJoCo state agree.
        try:
            return self._get_observations(force_update=True)
        except (AttributeError, TypeError):
            return observation

    raw.reset = MethodType(configured, raw)


def gripper_type_for_request(request: SceneRequest) -> str:
    """Choose a source-object-compatible official simulated parallel gripper."""
    return "RobotiqThreeFingerGripper" if request.selection_id == "VLA82-021" else "PandaGripper"


def robocasa_environment_kwargs(
    request: SceneRequest, seed: int, *, obj_groups: str | None = None,
    camera_width: int = 256, camera_height: int = 256,
    has_renderer: bool = False, renderer: str = "mjviewer",
) -> dict[str, Any]:
    """Build constructor kwargs without sending object-only args to fixture tasks."""
    kwargs: dict[str, Any] = {
        "robots": "PandaOmron",
        "gripper_types": gripper_type_for_request(request),
        "camera_names": list(request.camera_names),
        "camera_widths": int(camera_width),
        "camera_heights": int(camera_height),
        "horizon": 1000,
        "seed": seed,
        "split": "pretrain",
    }
    if request.selection_id == "VLA82-015":
        # A pan including its handle needs a full counter segment.  Constrain
        # only this reset-time scene choice to RoboCasa's official 5x1 layouts
        # rather than spending thousands of samples in narrow random layouts.
        kwargs["split"] = None
        kwargs["layout_and_style_ids"] = (
            (11, 34), (15, 34), (18, 34), (40, 34), (50, 34),
        )
    if request.primary_asset.kind != "fixture_part":
        kwargs["obj_registries"] = ("objaverse", "lightwheel", "aigen", "vla82_custom")
        if not obj_groups:
            raise EnvironmentValidationError("object scene requires an exact object group")
        kwargs["obj_groups"] = obj_groups
    if has_renderer:
        # RoboCasa's create_env owns the downstream ``has_renderer`` keyword.
        # Passing it through here duplicates that explicit argument; its public
        # on-screen switch is ``render_onscreen`` instead.
        kwargs.update({"render_onscreen": True, "render_offscreen": True, "renderer": str(renderer)})
    return kwargs


def _seed_raw_reset_rng(raw: Any, seed: int) -> None:
    """Bind every ObjectPlay reset sampler to the declared episode generator.

    ``ObjectPlayEnv`` does not expose a constructor seed and its
    ``UniformRandomSampler`` otherwise creates an entropy-seeded Generator of
    its own.  Resetting only ``raw.rng`` therefore left trash-can placement
    and Panda initialization different for the same requested seed.
    """
    generator = np.random.default_rng(int(seed))
    raw.rng = generator
    placement = getattr(raw, "placement_initializer", None)
    if placement is not None:
        if hasattr(placement, "rng"):
            placement.rng = generator
        for sampler in getattr(placement, "samplers", {}).values():
            if hasattr(sampler, "rng"):
                sampler.rng = generator


class _RoboCasaGymAdapter(gym.Env):
    """Small Gymnasium-compatible adapter over a real RoboCasa simulation."""

    def __init__(self, raw: Any, request: SceneRequest, initial_observation: Mapping[str, Any] | None = None, episode_seed: int | None = None):
        super().__init__()
        self._raw = raw
        self.request = request
        self._initial_observation = initial_observation
        self._episode_seed = episode_seed
        lower, upper = raw.action_spec
        self.action_space = gym.spaces.Box(
            low=np.asarray(lower, dtype=np.float32), high=np.asarray(upper, dtype=np.float32), dtype=np.float32
        )
        # Gymnasium owns ``unwrapped`` as a read-only property returning this
        # adapter; retaining the raw simulator on ``env`` preserves the same
        # access surface used by RoboCasa's Gym wrapper.
        self.env = raw

    def _wrap_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        wrapped = dict(observation)
        for camera in self.request.camera_names:
            source = f"{camera}_image"
            if source in observation:
                wrapped[f"video.{camera}"] = np.asarray(observation[source], dtype=np.uint8)
        return wrapped

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self._initial_observation is not None:
            observation = self._initial_observation
            self._initial_observation = None
            return self._wrap_observation(observation), {"seed": seed}
        effective_seed = self._episode_seed if seed is None else int(seed)
        if effective_seed is None:
            observation = self._raw.reset()
            return self._wrap_observation(observation), {"seed": seed}
        with seeded_rng_scope(effective_seed):
            _seed_raw_reset_rng(self._raw, effective_seed)
            observation = self._raw.reset()
        return self._wrap_observation(observation), {"seed": effective_seed}

    def step(self, action: np.ndarray):
        observation, reward, done, info = self._raw.step(np.asarray(action, dtype=np.float32))
        return self._wrap_observation(observation), float(reward), bool(done), False, dict(info)

    def close(self) -> None:
        self._raw.close()


class _TrashCanDepositEnvironment:
    """A source-matched MuJoCo receptacle scene with the full PandaOmron arm.

    It uses RoboCasa's ObjectPlayEnv rather than coercing the authoritative
    trash-can deposit annotation into an unrelated sink task.  The open custom
    MJCF trash can carries the source-video texture and real collision walls.
    """

    @staticmethod
    def build(
        asset: AssetSpec,
        request: SceneRequest,
        seed: int,
        *,
        camera_width: int = 256,
        camera_height: int = 256,
        has_renderer: bool = False,
        renderer: str = "mjviewer",
    ):
        from robocasa.utils.model_zoo.object_play_env import ObjectPlayEnv

        class SourceTexturedObjectPlayEnv(ObjectPlayEnv):
            def __init__(self, *, object_paths: tuple[str, ...], **kwargs: Any):
                self._object_paths = object_paths
                super().__init__(obj_mjcf_path=object_paths[0], num_objects=len(object_paths), **kwargs)

            def set_cameras(self) -> None:
                if self._bed_mjcf_path:
                    # Pull the audit camera back so the bed, the adjacent
                    # floor-standing nightstand and the full arm stay visible
                    # throughout the transfer rather than cropping the target.
                    self._cam_configs["agentview"] = dict(
                        self._cam_configs["agentview"], pos=[.15, -.90, 1.50],
                    )
                agentview = dict(self._cam_configs["agentview"])
                self._cam_configs["robot0_agentview_left"] = agentview
                super().set_cameras()

            def _load_model(self) -> None:
                # ObjectPlayEnv only repeats one MJCF; this source-faithful
                # variant loads every independently annotated object asset.
                from robocasa.models.objects.objects import MJCFObject
                from robosuite.models.arenas import TableArena
                from robosuite.models.objects import MujocoXMLObject
                from robosuite.models.tasks import ManipulationTask
                from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
                import robosuite.utils.transform_utils as transform

                super(ObjectPlayEnv, self)._load_model()
                for robot, rotation, offset in zip(self.robots, (-np.pi / 2,), (0,)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    orientation = np.array((0, 0, rotation))
                    xpos = transform.euler2mat(orientation) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos + np.array((0, offset, 0)))
                    robot.robot_model.set_base_ori(orientation)
                self.mujoco_arena = TableArena(
                    table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset
                )
                self.mujoco_arena.set_origin([0, 0, 0])
                self.set_cameras()
                # Asset 0 is the source-textured receptacle.  It is assembled
                # as a static scene body, never as a free ObjectPlay object:
                # a free trash can falls / topples during a correct release
                # and invalidates the physical containment predicate.
                target = MujocoXMLObject(
                    fname=self._object_paths[0], name="MCJFObj_0", joints=None,
                    # Preserve the original source asset at unit scale in all
                    # axes; reducing Z silently halves the usable cavity and
                    # no longer represents the authoritative object.
                    obj_type="all", duplicate_collision_geoms=False, scale=(1, 1, 1),
                )
                # Keep the source-path provenance surface used by the strict
                # runtime asset-identity audit (MujocoXMLObject omits it).
                target.mjcf_path = self._object_paths[0]
                # Construct the fixed receptacle in the Panda's reachable Y
                # band.  Source cans are sampled strictly to its +X exterior,
                # so this is a genuine cross-table transfer rather than an
                # initial-in-container shortcut.
                # The held-can response model has a materially stronger
                # controllable +Y component than long -X transport.  Build
                # the static target in that reachable direction: its outer
                # wall remains 1cm beyond the source-can radius envelope.
                self._trash_can_world_pos = np.array((*vla82_trash_can_target_center(), .96), dtype=float)
                target.set_pos(self._trash_can_world_pos)
                target.set_euler((0.0, 0.0, 0.0))
                movable = [
                    MJCFObject(name=f"MCJFObj_{index}", mjcf_path=path, scale=self._obj_scale)
                    for index, path in enumerate(self._object_paths[1:], start=1)
                ]
                objects = [target, *movable]
                # The three authoritative cans retain independent source
                # identities and their free joints for grasp / deposit proof.
                self.objects = {
                    "trash_can" if index == 0 else "obj" if index == 1 else f"annotated_{index - 1:02d}": value
                    for index, value in enumerate(objects)
                }
                # Dedicated reset samplers keep the first tested source in
                # the verified close-arm band while the two remaining cans
                # occupy non-overlapping supported table lanes. All are at
                # least 45mm beyond the target cavity's +X edge.
                self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
                source_lanes = tuple(((x, x), (y, y)) for x, y in vla82_trash_can_source_centers())
                for index, (x_range, y_range) in enumerate(source_lanes):
                    self.placement_initializer.append_sampler(UniformRandomSampler(
                        name=f"SourceCan{index}Sampler", mujoco_objects=movable[index],
                        x_range=x_range, y_range=y_range,
                        rotation=self._rotation, ensure_object_boundary_in_range=False, ensure_valid_placement=True,
                        reference_pos=self.table_offset, z_offset=0.01,
                    ))
                self.model = ManipulationTask(
                    mujoco_arena=self.mujoco_arena, mujoco_robots=[robot.robot_model for robot in self.robots],
                    mujoco_objects=objects,
                )

            def _reset_internal(self) -> None:
                # Keep the target's static XML pose fixed.  This mirrors
                # ObjectPlayEnv's reset implementation but samples / writes
                # only movable cans through its public reset sampler.
                super(ObjectPlayEnv, self)._reset_internal()
                placements = self.placement_initializer.sample()
                for obj_pos, obj_quat, obj in placements.values():
                    if not tuple(getattr(obj, "joints", ())):
                        continue
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate((np.array(obj_pos), np.array(obj_quat))))

        assets = tuple(materialize_custom_asset(item) for item in request.object_assets)
        raw = SourceTexturedObjectPlayEnv(
            robots="PandaOmron",
            controller_configs=vla82_trash_can_whole_body_ik_config(),
            object_paths=tuple(item.asset_path_or_group for item in assets),
            camera_names=list(request.camera_names),
            camera_widths=int(camera_width),
            camera_heights=int(camera_height),
            has_renderer=bool(has_renderer),
            has_offscreen_renderer=True,
            ignore_done=True,
            # A hard reset rebuilds ``placement_initializer`` after we seed
            # it below, silently replacing its Generator with entropy.  The
            # source MJCF is already fully loaded for this per-episode env;
            # retain it and reseed its live sampler on every reset instead.
            hard_reset=False,
            # Keep the standard 0.8m X length (and therefore the Panda base-X
            # calibration) while widening only the safe transfer direction.
            table_full_size=(.8, 1.2, .05),
            x_range=(-0.28, 0.28),
            y_range=(-0.24, 0.24),
        )
        # SourceTexturedObjectPlayEnv bypasses RoboCasa's seeded ``create_env``
        # path.  Seed both its environment and its already-created sampler
        # before the very first reset captured by the adapter.
        _seed_raw_reset_rng(raw, seed)
        initial_observation = raw.reset()
        return _RoboCasaGymAdapter(raw, request, initial_observation, episode_seed=seed)


class _WorkStudyOpenSupportEnvironment:
    """A source-textured VLA object on a normal open RoboCasa work surface."""

    @staticmethod
    def build(
        asset: AssetSpec,
        request: SceneRequest,
        seed: int,
        *,
        camera_width: int = 256,
        camera_height: int = 256,
        has_renderer: bool = False,
        renderer: str = "mjviewer",
    ):
        from robocasa.utils.model_zoo.object_play_env import ObjectPlayEnv

        class OpenSupportObjectPlayEnv(ObjectPlayEnv):
            def __init__(
                self, *, target_mjcf_path: str | None = None,
                target_object_name: str | None = None,
                bed_mjcf_path: str | None = None, **kwargs: Any,
            ):
                self._target_mjcf_path = target_mjcf_path
                self._target_object_name = target_object_name
                self._bed_mjcf_path = bed_mjcf_path
                super().__init__(**kwargs)

            def _load_model(self) -> None:
                from robocasa.models.objects.objects import MJCFObject
                from robosuite.models.arenas import EmptyArena, TableArena
                from robosuite.models.objects import MujocoXMLObject
                from robosuite.models.tasks import ManipulationTask
                from robosuite.utils.placement_samplers import UniformRandomSampler
                import robosuite.utils.transform_utils as transform

                super(ObjectPlayEnv, self)._load_model()
                for robot, rotation, offset in zip(self.robots, (-np.pi / 2,), (0,)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    orientation = np.array((0, 0, rotation))
                    robot.robot_model.set_base_xpos(transform.euler2mat(orientation) @ xpos + np.array((0, offset, 0)))
                    robot.robot_model.set_base_ori(orientation)
                bedroom_scene = bool(self._bed_mjcf_path)
                self.mujoco_arena = (
                    EmptyArena()
                    if bedroom_scene
                    else TableArena(
                        table_full_size=self.table_full_size,
                        table_friction=self.table_friction,
                        table_offset=self.table_offset,
                    )
                )
                self.mujoco_arena.set_origin([0, 0, 0])
                self.set_cameras()
                obj = MJCFObject(name="MCJFObj_0", mjcf_path=self._obj_mjcf_path, scale=self._obj_scale)
                objects = [obj]
                self.objects = {"obj": obj}
                if self._bed_mjcf_path:
                    bed_object = MujocoXMLObject(
                        fname=self._bed_mjcf_path, name="MCJFObj_bed", joints=None,
                        obj_type="all", duplicate_collision_geoms=False, scale=(1, 1, 1),
                    )
                    bed_object.mjcf_path = self._bed_mjcf_path
                    bed_object.set_pos(fixed_support_target_position("bed"))
                    bed_object.set_euler((0., 0., 0.))
                    self.objects["bed"] = bed_object
                    objects.append(bed_object)
                if self._target_mjcf_path:
                    target_name = str(self._target_object_name or "target")
                    target_object = MujocoXMLObject(
                        fname=self._target_mjcf_path, name=f"MCJFObj_{target_name}", joints=None,
                        obj_type="all", duplicate_collision_geoms=False, scale=(1, 1, 1),
                    )
                    target_object.mjcf_path = self._target_mjcf_path
                    target_object.set_pos(fixed_support_target_position(target_name))
                    target_object.set_euler((0., 0., 0.))
                    self.objects[target_name] = target_object
                    objects.append(target_object)
                self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler", mujoco_objects=[obj],
                    x_range=self._x_range, y_range=self._y_range,
                    rotation=np.pi / 2 if self._target_object_name == "pencil_case" else self._rotation,
                    rotation_axis="x" if self._target_object_name == "pencil_case" else "z",
                    ensure_object_boundary_in_range=False, ensure_valid_placement=True,
                    reference_pos=(0., 0., .720) if bedroom_scene else self.table_offset,
                    z_offset=.005 if bedroom_scene else .01,
                )
                self.model = ManipulationTask(
                    mujoco_arena=self.mujoco_arena,
                    mujoco_robots=[robot.robot_model for robot in self.robots],
                    mujoco_objects=objects,
                )

            def set_cameras(self) -> None:
                agentview = dict(self._cam_configs["agentview"])
                self._cam_configs["robot0_agentview_left"] = agentview
                super().set_cameras()

        source_asset = materialize_custom_asset(asset)
        target_object_name: str | None = None
        target_asset: AssetSpec | None = None
        bed_asset: AssetSpec | None = None
        if request.task_class == "VLA82PencilCaseInsert":
            target_object_name = "pencil_case"
            target_asset = materialize_custom_asset(request.object_assets[1])
        elif request.task_class == "VLA82ToothbrushCupInsert":
            target_object_name = "mouthwash_cup"
            target_asset = materialize_custom_asset(request.object_assets[1])
        elif request.task_class == "VLA82BathroomShelfPlace":
            target_object_name = "bathroom_shelf"
            target_asset = materialize_custom_asset(request.object_assets[1])
        elif request.selection_id == "VLA82-053":
            target_object_name = "keyboard"
            target_asset = materialize_custom_asset(next(
                item for item in request.object_assets if item.semantic_class == "键盘"
            ))
        elif request.task_class == "VLA82NotebookStack":
            target_object_name = "folder"
            target_asset = materialize_custom_asset(next(
                item for item in request.object_assets if item.semantic_class == "文件夹"
            ))
        elif request.task_class == "VLA82BedsideBookPlace":
            target_object_name = "nightstand"
            target_asset = materialize_custom_asset(next(
                item for item in request.object_assets if item.semantic_class == "床头柜"
            ))
            bed_asset = materialize_custom_asset(next(
                item for item in request.object_assets if item.semantic_class == "床"
            ))
        source_x_range, source_y_range = work_study_source_ranges(request.selection_id)
        raw = OpenSupportObjectPlayEnv(
            robots="PandaOmron",
            controller_configs=vla82_open_support_controller_config(),
            obj_mjcf_path=source_asset.asset_path_or_group,
            target_mjcf_path=target_asset.asset_path_or_group if target_asset else None,
            target_object_name=target_object_name,
            bed_mjcf_path=bed_asset.asset_path_or_group if bed_asset else None,
            num_objects=1,
            camera_names=list(request.camera_names),
            camera_widths=int(camera_width),
            camera_heights=int(camera_height),
            has_renderer=bool(has_renderer),
            has_offscreen_renderer=True,
            ignore_done=True,
            hard_reset=False,
            table_full_size=(.8, 1.2, .05),
            # Reset on the robot-facing half of the same unobstructed desk.
            # PandaOmron's initial arm centre is near (0.01, 0.10); placing a
            # flat folder beside that reachable patch avoids a mobile-base
            # command before grasp.  The table centre remains a distinct
            # physical placement destination.
            x_range=source_x_range,
            y_range=source_y_range,
        )
        _seed_raw_reset_rng(raw, seed)
        initial_observation = raw.reset()
        return _RoboCasaGymAdapter(raw, request, initial_observation, episode_seed=seed)


def _build_registered_or_custom_robocasa_task(
    request: SceneRequest,
    seed: int,
    *,
    camera_width: int = 256,
    camera_height: int = 256,
    has_renderer: bool = False,
    renderer: str = "mjviewer",
):
    from robocasa.utils.env_utils import create_env

    if request.robot_name != "PandaOmron":
        raise EnvironmentValidationError("only complete PandaOmron scenes are permitted")
    if request.task_class == "VLA82TrashCanDeposit":
        return _TrashCanDepositEnvironment.build(
            request.primary_asset, request, seed, camera_width=camera_width,
            camera_height=camera_height, has_renderer=has_renderer, renderer=renderer,
        )
    if request.task_class in {"VLA82WorkStudyOpenSupport", "VLA82NotebookStack", "VLA82BedsideBookPlace"}:
        return _WorkStudyOpenSupportEnvironment.build(
            request.primary_asset, request, seed, camera_width=camera_width,
            camera_height=camera_height, has_renderer=has_renderer, renderer=renderer,
        )
    if request.task_class in {
        "VLA82PencilCaseInsert", "VLA82ToothbrushCupInsert", "VLA82BathroomShelfPlace",
    }:
        return _WorkStudyOpenSupportEnvironment.build(
            request.primary_asset, request, seed, camera_width=camera_width,
            camera_height=camera_height, has_renderer=has_renderer, renderer=renderer,
        )
    if request.primary_asset.kind == "fixture_part":
        raw = create_env(
            request.task_class,
            **robocasa_environment_kwargs(
                request, seed, camera_width=camera_width, camera_height=camera_height,
                has_renderer=has_renderer, renderer=renderer,
            ),
        )
        _configure_fixture_close_partial_open_reset(raw, request)
        initial_observation = raw.reset()
        return _RoboCasaGymAdapter(raw, request, initial_observation, episode_seed=seed)
    if request.primary_asset.kind == "robocasa_native":
        raw = create_env(
            request.task_class,
            **robocasa_environment_kwargs(
                request, seed, obj_groups=request.primary_asset.asset_path_or_group,
                camera_width=camera_width, camera_height=camera_height,
                has_renderer=has_renderer, renderer=renderer,
            ),
        )
        # Apply the same reset-time source-counter placement contract to the
        # exact native asset; this only changes RoboCasa's sampler config.
        _customize_object_configs(raw, (request.primary_asset.asset_path_or_group,), request)
        initial_observation = raw.reset()
        return _RoboCasaGymAdapter(raw, request, initial_observation, episode_seed=seed)

    validated_assets = tuple(materialize_custom_asset(asset) for asset in request.object_assets)
    categories = tuple(_register_custom_category(asset) for asset in validated_assets)
    raw = create_env(
        request.task_class,
        **robocasa_environment_kwargs(
            request, seed, obj_groups=categories[0], camera_width=camera_width,
            camera_height=camera_height, has_renderer=has_renderer, renderer=renderer,
        ),
    )
    _customize_object_configs(raw, categories, request)
    _configure_cabinet_source_robot_spawn(raw, request)
    _configure_counter_source_robot_spawn(raw, request)
    _configure_drawer_source_robot_spawn(raw, request)
    _configure_vla005_robot_spawn(raw, request)
    # RoboCasa allocates controllers and the 12-D full-arm action surface during
    # reset.  The object-config override must be installed before this reset.
    initial_observation = raw.reset()
    return _RoboCasaGymAdapter(raw, request, initial_observation, episode_seed=seed)


def _raw_environment(environment: Any) -> Any:
    return getattr(getattr(environment, "unwrapped", environment), "env", environment)


def _flatten_qpos_indexes(indexes: Any) -> np.ndarray:
    """Normalize robosuite list and PandaOmron hand-group index surfaces."""
    if isinstance(indexes, Mapping):
        return np.asarray([item for group in indexes.values() for item in group], dtype=int)
    return np.asarray(indexes, dtype=int)


def preferred_fixture_joint_names(
    *, task_class: str, discovered_joint_names: Sequence[str],
    fridge_door_joint_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Bind CloseFridge to the doors opened by the official task reset."""
    discovered = tuple(str(name) for name in discovered_joint_names)
    if str(task_class) != "CloseFridge":
        return discovered
    allowed = set(str(name) for name in fridge_door_joint_names)
    return tuple(name for name in discovered if name in allowed)


def fixture_binding(raw: Any, request: SceneRequest) -> dict[str, str]:
    """Resolve exactly one authoritative fixture and its concrete MuJoCo IDs."""
    target = FIXTURE_TARGETS.get(request.selection_id)
    if target is None:
        raise EnvironmentValidationError(f"fixture target is not declared for {request.selection_id}")
    fixture_token, part_token = target
    # Atomic appliance environments may sample a concrete part at reset (for
    # example ``raw.knob == 'rear_center'``).  Bind that source-declared live
    # part, not the first knob/body in MuJoCo name order.
    live_part = getattr(raw, part_token, None)
    if isinstance(live_part, str) and live_part:
        part_token = live_part
    active_fixture = getattr(raw, "fxtr", None)
    if active_fixture is None:
        # Rack / appliance tasks expose their selected fixture under a task
        # attribute (for example SlideOvenRack.oven), rather than ``fxtr``.
        for attribute in ("stove", "oven", "toaster_oven", "fridge", "microwave", "drawer", "cabinet", "dishwasher"):
            candidate = getattr(raw, attribute, None)
            if hasattr(candidate, "naming_prefix"):
                active_fixture = candidate
                break
    if active_fixture is not None:
        # Atomic RoboCasa appliance tasks expose the exact sampled fixture in
        # ``fxtr``. It is more precise than matching another same-type fixture.
        candidates = [(name, fixture) for name, fixture in getattr(raw, "fixtures", {}).items() if fixture is active_fixture]
    else:
        candidates = [
            (name, fixture) for name, fixture in getattr(raw, "fixtures", {}).items()
            if fixture_token in f"{name} {type(fixture).__name__}".lower().replace("_", "")
        ]
    if len(candidates) != 1:
        raise EnvironmentValidationError(f"fixture target is ambiguous or absent for {request.selection_id}: {fixture_token}")
    name, fixture = candidates[0]
    prefix = str(getattr(fixture, "naming_prefix", ""))
    model = raw.sim.model
    geoms = [
        model.geom_id2name(index) or "" for index in range(int(model.ngeom))
        if (model.geom_id2name(index) or "").startswith(prefix) and part_token in (model.geom_id2name(index) or "").lower()
    ]
    joints = [
        model.joint_id2name(index) or "" for index in range(int(model.njnt))
        if (model.joint_id2name(index) or "").startswith(prefix) and part_token in (model.joint_id2name(index) or "").lower()
    ]
    if not joints:
        joints = [
            model.joint_id2name(index) or "" for index in range(int(model.njnt))
            if (model.joint_id2name(index) or "").startswith(prefix)
        ]
    joints = list(preferred_fixture_joint_names(
        task_class=request.task_class,
        discovered_joint_names=joints,
        fridge_door_joint_names=tuple(getattr(active_fixture, "_fridge_door_joint_names", ())),
    ))
    if request.task_class == "CloseFridge" and joints:
        joint_token = joints[0][len(prefix):] if joints[0].startswith(prefix) else joints[0]
        joint_token = joint_token.removesuffix("_joint")
        matching_geoms = [name for name in geoms if joint_token in name]
        if matching_geoms:
            geoms = matching_geoms
    if not geoms:
        raise EnvironmentValidationError(f"fixture target has no {part_token} geom for {request.selection_id}")
    semantic_geom = geoms[0]
    body_id = int(model.geom_bodyid[model.geom_name2id(semantic_geom)])
    visible_geoms = [
        model.geom_id2name(index) or "" for index in range(int(model.ngeom))
        if int(model.geom_bodyid[index]) == body_id and float(model.geom_rgba[index][3]) > 0.01
    ]
    if not visible_geoms:
        raise EnvironmentValidationError(f"fixture target body has no visible geom for {request.selection_id}")
    joint_required = bool({"slider", "knob", "hinge", "pressable_control"} & set(request.fixture_requirements))
    if joint_required and not joints:
        raise EnvironmentValidationError(f"operable fixture target has no joint for {request.selection_id}")
    return {
        "fixture_name": str(name), "fixture_prefix": prefix, "body": model.body_id2name(body_id) or "",
        "semantic_geom": semantic_geom, "geom": visible_geoms[0], "joint": joints[0] if joints else "",
        "joint_required": str(joint_required).lower(),
    }


def audit_camera_target_object_id(task_class: str) -> str:
    """Return the source object that must remain visible in an audit video."""
    return "trash_can" if str(task_class) == "VLA82TrashCanDeposit" else "obj"


def audit_camera_offsets(task_class: str, selection_id: str = "") -> tuple[tuple[float, float, float], ...]:
    """Return camera candidates sized for the selected physical operation."""
    if str(task_class) == "VLA82TrashCanDeposit":
        # The receptacle is tall and the deposit path is cross-table; a wider
        # oblique view keeps the complete trash can in frame instead of
        # passing a four-pixel segmentation check with the rim cropped out.
        return ((.85, -1.10, .55), (-.85, -1.10, .55), (.95, .35, .65))
    if str(selection_id) == "VLA82-007":
        # The stove's cookware can hide a low front view of the selected
        # rotary control.  A closer elevated view from the side opposite the
        # pan keeps the finger pads and the selected central knob legible,
        # while the two fallbacks retain a complete action view if occluded.
        return ((-.85, .55, .75), (.95, .55, .78), (-1.00, -.72, .85))
    if str(selection_id) == "VLA82-030":
        return ((-1.25, -1.35, .85), (1.25, -1.35, .85), (-1.35, .65, .90))
    if str(selection_id) == "VLA82-029":
        # A low diagonal camera looks through the microwave's moving door
        # frame.  An elevated, wider side view keeps the Panda wrist, handle
        # and closing panel visible throughout the full hinge sweep.
        return ((1.65, -1.75, 1.35), (-1.65, -1.75, 1.35), (1.85, .45, 1.50))
    if str(selection_id) == "VLA82-004":
        # This close side-front orbit stays within the kitchen rather than
        # looking through the cabinet's exterior wall.  Replay locks this
        # view in world space while the mobile base extracts the bottle.
        return ((-.75, .35, .70), (-.55, -.75, .55), (-.75, -.30, .70))
    if str(selection_id) in {"VLA82-005", "VLA82-041"}:
        # Keep the counter pickup, open drawer, gripper and complete transfer
        # above the counter edge in one elevated diagonal audit view.
        return ((1.25, -1.35, .85), (-1.25, -1.35, .85), (1.35, .65, .90))
    if str(selection_id) == "VLA82-027":
        # The fridge's opened outer door hides its lower drawer from the
        # right-side diagonal.  The mirrored elevated view exposes the
        # drawer face, closing contact, and final seated position together.
        return ((-1.25, -1.35, .85), (1.25, -1.35, .85), (-1.35, .65, .90))
    if str(selection_id) in {"VLA82-017", "VLA82-019", "VLA82-020", "VLA82-027", "VLA82-030", "VLA82-035", "VLA82-036", "VLA82-045", "VLA82-060"}:
        # Cabinet / drawer panels can fully occlude a close front view.  Use
        # a distant elevated diagonal with room for the robot, panel, and
        # contact point, then try the mirrored side if needed.
        return ((1.25, -1.35, .85), (-1.25, -1.35, .85), (1.35, .65, .90))
    return ((.35, -.55, .06), (-.35, -.55, .06), (.55, .20, .06), (-.55, .20, .06), (0, .55, .04))


def audit_camera_focus(
    selection_id: str,
    *,
    target: Sequence[float],
    eef: Sequence[float],
    destination: Sequence[float] | None = None,
) -> np.ndarray:
    """Keep both hand and target within a wide unobstructed operation view."""
    target_point = np.asarray(target, dtype=float)
    if str(selection_id) in {"VLA82-005", "VLA82-041", "VLA82-045"} and destination is not None:
        return .5 * (target_point + np.asarray(destination, dtype=float))
    if str(selection_id) in {"VLA82-004", "VLA82-017", "VLA82-019", "VLA82-020", "VLA82-030", "VLA82-035", "VLA82-036", "VLA82-060"}:
        return .5 * (target_point + np.asarray(eef, dtype=float))
    return target_point


def configure_audit_camera(
    raw: Any,
    request: SceneRequest,
    *,
    contract: PhysicsCaptureContract | None = None,
) -> dict[str, Any]:
    """Aim the required free audit camera at the exact target without moving physics state."""
    if request.primary_asset.kind == "fixture_part":
        binding = fixture_binding(raw, request)
        geom_id = raw.sim.model.geom_name2id(binding["geom"])
        target = np.asarray(raw.sim.data.geom_xpos[geom_id], dtype=float)
        target_geom_ids = {geom_id}
    else:
        # The VLA82 trash-can scene has a movable source can in ``obj`` and
        # a static, source-labelled receptacle in ``trash_can``.  The latter
        # is the operation target and must be visible in the result video.
        obj = raw.objects[audit_camera_target_object_id(request.task_class)]
        body_id = raw.sim.model.body_name2id(obj.root_body)
        target = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
        prefix = str(getattr(obj, "naming_prefix", "obj_"))
        target_geom_ids = {
            index for index in range(int(raw.sim.model.ngeom))
            if (raw.sim.model.geom_id2name(index) or "").startswith(prefix)
        }
    camera = "robot0_agentview_left"
    camera_id = raw.sim.model.camera_name2id(camera)
    from robosuite.utils import transform_utils as transform
    parent_id = int(raw.sim.model.cam_bodyid[camera_id])
    parent_position = np.asarray(raw.sim.data.body_xpos[parent_id], dtype=float)
    parent_rotation = np.asarray(raw.sim.data.body_xmat[parent_id], dtype=float).reshape(3, 3)
    original_position = raw.sim.model.cam_pos[camera_id].copy()
    original_quaternion = raw.sim.model.cam_quat[camera_id].copy()
    selected: tuple[np.ndarray, int] | None = None
    robot = raw.robots[0]
    eef = np.asarray(raw.sim.data.site_xpos[robot.eef_site_id["right"]], dtype=float)
    destination = None
    if contract is not None and contract.target_geometries:
        target_geometry = next(iter(contract.target_geometries.values()))
        destination = .5 * (
            np.asarray(target_geometry.min_corner, dtype=float)
            + np.asarray(target_geometry.max_corner, dtype=float)
        )
    focus = audit_camera_focus(
        request.selection_id,
        target=target,
        eef=eef,
        destination=destination,
    )
    for offset in audit_camera_offsets(request.task_class, request.selection_id):
        position = focus + np.asarray(offset)
        forward = (focus - position) / np.linalg.norm(focus - position)
        right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
        right /= np.linalg.norm(right)
        rotation = np.column_stack((right, np.cross(right, forward), -forward))
        raw.sim.model.cam_pos[camera_id] = parent_rotation.T @ (position - parent_position)
        raw.sim.model.cam_quat[camera_id] = transform.convert_quat(transform.mat2quat(parent_rotation.T @ rotation), to="wxyz")
        raw.sim.forward()
        segmentation = raw.sim.render(width=256, height=256, camera_name=camera, segmentation=True)
        pixels = sum(int((segmentation[:, :, 1] == geom_id).sum()) for geom_id in target_geom_ids)
        if pixels >= 4:
            selected = (position, pixels)
            break
    if selected is None:
        # Some source placement contracts intentionally place the object behind
        # a fixture. Preserve the task's original required camera if it is the
        # only view that exposes the exact object.
        raw.sim.model.cam_pos[camera_id] = original_position
        raw.sim.model.cam_quat[camera_id] = original_quaternion
        raw.sim.forward()
        segmentation = raw.sim.render(width=256, height=256, camera_name=camera, segmentation=True)
        pixels = sum(int((segmentation[:, :, 1] == geom_id).sum()) for geom_id in target_geom_ids)
        selected = (np.asarray(raw.sim.data.cam_xpos[camera_id], dtype=float), pixels)
    # This is a camera extrinsic only: no qpos, body pose, joint, or action is written.
    return {"camera_name": camera, "position": selected[0].tolist(), "target": target.tolist(), "focus": focus.tolist(), "segmentation_pixels": selected[1]}


def validate_scene_objects(raw: Any, request: SceneRequest) -> None:
    """Prove that every source-annotated object was instantiated in MuJoCo."""
    actual = set(getattr(raw, "objects", {}))
    # The work-study primitive deliberately uses one exact VLA source asset on
    # a single open desktop.  Its catalog companions are annotations from the
    # original recording, not additional scene objects; requiring them here
    # would validate an unrelated bookcase/cabinet layout instead.
    if request.task_class in {"VLA82WorkStudyOpenSupport", "VLA82NotebookStack", "VLA82BedsideBookPlace"}:
        expected = (
            {"obj", "folder"} if request.task_class == "VLA82NotebookStack"
            else {"obj", "bed", "nightstand"} if request.task_class == "VLA82BedsideBookPlace"
            else {"obj", "keyboard"} if request.selection_id == "VLA82-053"
            else {"obj"}
        )
    elif request.task_class == "VLA82PencilCaseInsert":
        expected = {"obj", "pencil_case"}
    elif request.task_class == "VLA82ToothbrushCupInsert":
        expected = {"obj", "mouthwash_cup"}
    elif request.task_class == "VLA82BathroomShelfPlace":
        expected = {"obj", "bathroom_shelf"}
    else:
        expected = {"obj", *(f"annotated_{index:02d}" for index in range(1, len(request.object_assets)))}
    missing = sorted(expected - actual)
    if missing:
        raise EnvironmentValidationError(f"scene is missing annotated MuJoCo object(s): {', '.join(missing)}")


def runtime_primary_asset_identity(raw: Any, request: SceneRequest) -> dict[str, Any]:
    """Fail closed unless the live primary object is this selection's MJCF.

    RoboCasa categories sample a model directory at runtime; catalog and source
    evidence alone are therefore insufficient proof of object identity.
    """
    asset = request.primary_asset
    identity: dict[str, Any] = {
        "selection_id": request.selection_id,
        "asset_key": asset.asset_key or asset.selection_id,
        "asset_kind": asset.kind,
    }
    if asset.kind != "custom_same_class":
        identity.update({"matches": True, "loaded_mjcf_path": "", "expected_mjcf_path": ""})
        return identity
    # VLA82-001's primary source asset is the receptacle; ``obj`` is the
    # first independently annotated can and must not replace that identity.
    primary_object_id = "trash_can" if request.task_class == "VLA82TrashCanDeposit" else "obj"
    loaded = getattr(getattr(raw, "objects", {}).get(primary_object_id), "mjcf_path", "")
    expected = asset.asset_path_or_group
    if not loaded or Path(str(loaded)).resolve() != Path(str(expected)).resolve():
        raise EnvironmentValidationError(
            f"runtime primary MJCF mismatch for {request.selection_id}: loaded={loaded} expected={expected}"
        )
    identity.update({
        "matches": True,
        "loaded_mjcf_path": str(Path(str(loaded)).resolve()),
        "expected_mjcf_path": str(Path(str(expected)).resolve()),
    })
    return identity


def validate_trash_can_deposit_scene(raw: Any, request: SceneRequest) -> None:
    """Require one target receptacle and the three source-annotated cans."""
    if (request.source_fixture, request.target_fixture, request.target_relation) != ("cans", "trash_can", "inside"):
        raise EnvironmentValidationError("trash-can source-target semantics are not explicit")
    actual = set(getattr(raw, "objects", {}))
    expected = {"trash_can", "obj", "annotated_01", "annotated_02"}
    missing = sorted(expected - actual)
    if missing:
        raise EnvironmentValidationError(f"trash-can scene is missing target/can object(s): {', '.join(missing)}")


def make_environment(
    request: SceneRequest,
    seed: int,
    *,
    camera_width: int = 256,
    camera_height: int = 256,
    has_renderer: bool = False,
    renderer: str = "mjviewer",
):
    """Instantiate and validate a full arm, two-camera, physics-backed environment."""
    with seeded_rng_scope(seed):
        environment = _build_registered_or_custom_robocasa_task(
            request, seed, camera_width=camera_width, camera_height=camera_height,
            has_renderer=has_renderer, renderer=renderer,
        )
    raw = _raw_environment(environment)
    robots = getattr(raw, "robots", ())
    if not robots or len(getattr(robots[0], "robot_joints", ())) < 7:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        raise EnvironmentValidationError("complete PandaOmron robot requires at least seven robot joints")
    cameras = set(getattr(raw, "camera_names", ()))
    if not set(request.camera_names).issubset(cameras):
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        raise EnvironmentValidationError("complete PandaOmron scene is missing required cameras")
    if request.task_class == "VLA82TrashCanDeposit":
        try:
            validate_trash_can_deposit_scene(raw, request)
        except EnvironmentValidationError:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
            raise
    elif request.primary_asset.kind == "fixture_part":
        try:
            fixture_binding(raw, request)
        except EnvironmentValidationError:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
            raise
    else:
        try:
            validate_scene_objects(raw, request)
        except EnvironmentValidationError:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
            raise
    try:
        environment.runtime_asset_identity = runtime_primary_asset_identity(raw, request)
    except EnvironmentValidationError:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        raise
    contract = build_physics_capture_contract(environment, request)
    selected_geoms = tuple(contract.object_geom_names.get("obj", ()))
    friction_profile = profile_for(request.selection_id, request.primary_asset.semantic_class)
    if selected_geoms:
        environment.friction_evidence = apply_object_friction(
            raw,
            selected_geoms,
            friction_profile,
        )
    else:
        environment.friction_evidence = {
            "version": friction_profile.version,
            "material": friction_profile.material,
            "material_source": "not_applicable_fixture_part",
            "geom_names": [],
            "before": {},
            "after": {},
            "application_stage": "post_load_pre_step",
        }
    environment.audit_camera = configure_audit_camera(raw, request, contract=contract)
    environment.runtime_scene_fingerprint = scene_runtime_fingerprint(environment, request)
    return environment


def _trash_can_capture_contract(raw: Any, request: SceneRequest) -> PhysicsCaptureContract:
    """Bind the source trash-can walls and open inner volume, never a table."""
    if (request.source_fixture, request.target_fixture, request.target_relation) != ("cans", "trash_can", "inside"):
        raise EnvironmentValidationError("trash-can capture contract has no explicit source-target semantics")
    model = raw.sim.model
    object_geoms: dict[str, tuple[str, ...]] = {}
    for object_id in ("obj", "annotated_01", "annotated_02"):
        obj = raw.objects.get(object_id)
        prefix = str(getattr(obj, "naming_prefix", ""))
        geoms = tuple(
            name for index in range(int(model.ngeom))
            if (name := model.geom_id2name(index) or "").startswith(prefix)
        )
        if not geoms:
            raise EnvironmentValidationError(f"trash-can source can has no geom: {object_id}")
        object_geoms[object_id] = geoms
    receptacle = raw.objects.get("trash_can")
    prefix = str(getattr(receptacle, "naming_prefix", ""))
    wall_geoms = tuple(
        name for index in range(int(model.ngeom))
        if (name := model.geom_id2name(index) or "").startswith(prefix) and "collision_" in name
    )
    volume_geom = next(
        (name for index in range(int(model.ngeom))
         if (name := model.geom_id2name(index) or "").startswith(prefix) and "reg_bbox" in name),
        None,
    )
    if len(wall_geoms) < 5 or volume_geom is None:
        raise EnvironmentValidationError("trash-can target has no exact walls/open-cavity volume")
    volume_id = model.geom_name2id(volume_geom)
    center = np.asarray(raw.sim.data.geom_xpos[volume_id], dtype=float)
    extent = np.asarray(model.geom_size[volume_id], dtype=float)[:3]
    # The visible/collision model has 8-mm walls and an open top. Keep the
    # interior clear of side walls and require a can below the rim plane.
    inner = np.maximum(extent - np.array((0.012, 0.012, 0.0)), np.array((0.02, 0.02, 0.02)))
    target_id = "trash_can:inner_cavity"
    target = TargetGeometry(
        target_id, "trash_can", tuple(center - np.array((inner[0], inner[1], extent[2]))),
        tuple(center + np.array((inner[0], inner[1], extent[2]))), target_id, "inside",
    )
    return PhysicsCaptureContract(
        object_geom_names=object_geoms,
        target_geom_names={target_id: wall_geoms},
        target_geometries={target_id: target},
    )


def _work_study_open_support_capture_contract(raw: Any, request: SceneRequest) -> PhysicsCaptureContract:
    """Bind the exact source object and physical desktop collision surface."""
    obj = raw.objects.get("obj")
    prefix = str(getattr(obj, "naming_prefix", ""))
    object_geoms = tuple(
        name for index in range(int(raw.sim.model.ngeom))
        if (name := raw.sim.model.geom_id2name(index) or "").startswith(prefix)
    )
    if not object_geoms:
        raise EnvironmentValidationError("open-support scene has no source object collision geometry")
    table_geoms = tuple(
        name for index in range(int(raw.sim.model.ngeom))
        if (name := raw.sim.model.geom_id2name(index) or "")
        and "table" in name.lower()
        and "visual" not in name.lower()
        and int(raw.sim.model.geom_contype[index]) != 0
    )
    if not table_geoms:
        raise EnvironmentValidationError("open-support scene has no physical desktop geometry")
    ids = tuple(raw.sim.model.geom_name2id(name) for name in table_geoms)
    centers = np.asarray([raw.sim.data.geom_xpos[index] for index in ids], dtype=float)
    sizes = np.asarray([raw.sim.model.geom_size[index] for index in ids], dtype=float)
    extents = np.maximum(sizes[:, :3], np.array((.02, .02, .02)))
    target_id = "open_support_table:tabletop"
    target = TargetGeometry(
        target_id,
        "open_support_table",
        tuple(np.min(centers - extents, axis=0)),
        tuple(np.max(centers + extents, axis=0)),
        target_id,
        "on",
    )
    return PhysicsCaptureContract(
        {"obj": object_geoms},
        {target_id: table_geoms},
        {target_id: target},
    )


def _fixed_object_support_capture_contract(
    raw: Any, request: SceneRequest, *, target_object_name: str, fixture_id: str,
) -> PhysicsCaptureContract:
    """Bind a movable source to one exact static object used as its support."""
    model = raw.sim.model
    source = raw.objects.get("obj")
    target_object = raw.objects.get(target_object_name)
    source_prefix = str(getattr(source, "naming_prefix", ""))
    target_prefix = str(getattr(target_object, "naming_prefix", ""))
    source_geoms = tuple(
        name for index in range(int(model.ngeom))
        if (name := model.geom_id2name(index) or "").startswith(source_prefix)
    )
    target_collision_geoms = tuple(
        name for index in range(int(model.ngeom))
        if (name := model.geom_id2name(index) or "").startswith(target_prefix)
        and "visual" not in name.lower()
        and (int(model.geom_contype[index]) or int(model.geom_conaffinity[index]))
    )
    target_geoms = (
        tuple(name for name in target_collision_geoms if "collision_bottom" in name)
        if fixture_id == "bathroom_shelf"
        else target_collision_geoms
    )
    if not source_geoms or not target_geoms:
        raise EnvironmentValidationError("fixed-support scene lacks exact source or target collision geometry")
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for name in target_geoms:
        geom_id = model.geom_name2id(name)
        rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        extent = np.abs(rotation) @ np.asarray(model.geom_size[geom_id], dtype=float)
        center = np.asarray(raw.sim.data.geom_xpos[geom_id], dtype=float)
        lows.append(center - extent)
        highs.append(center + extent)
    target_id = f"{fixture_id}:tabletop"
    target = TargetGeometry(
        target_id, fixture_id, tuple(np.min(lows, axis=0)), tuple(np.max(highs, axis=0)),
        target_id, "on",
    )
    return PhysicsCaptureContract(
        {"obj": source_geoms}, {target_id: target_geoms}, {target_id: target},
    )


def _open_receptacle_insert_capture_contract(
    raw: Any,
    request: SceneRequest,
    *,
    target_object_name: str,
    target_fixture: str,
) -> PhysicsCaptureContract:
    """Bind one physical source and one exact five-wall open receptacle."""
    if (request.source_fixture, request.target_fixture, request.target_relation) != (
        "table", target_fixture, "inside",
    ):
        raise EnvironmentValidationError("open-receptacle insert has no explicit source-target semantics")
    model = raw.sim.model
    pencil = raw.objects.get("obj")
    pencil_prefix = str(getattr(pencil, "naming_prefix", ""))
    object_geoms = tuple(
        name for index in range(int(model.ngeom))
        if (name := model.geom_id2name(index) or "").startswith(pencil_prefix)
    )
    case = raw.objects.get(target_object_name)
    case_prefix = str(getattr(case, "naming_prefix", ""))
    wall_geoms = tuple(
        name for index in range(int(model.ngeom))
        if (name := model.geom_id2name(index) or "").startswith(case_prefix)
        and "collision_" in name
    )
    volume_geom = next(
        (name for index in range(int(model.ngeom))
         if (name := model.geom_id2name(index) or "").startswith(case_prefix) and "reg_bbox" in name),
        None,
    )
    if not object_geoms or len(wall_geoms) < 5 or volume_geom is None:
        raise EnvironmentValidationError("insert scene lacks source object or open-cavity collision geometry")
    volume_id = model.geom_name2id(volume_geom)
    center = np.asarray(raw.sim.data.geom_xpos[volume_id], dtype=float)
    extent = np.asarray(model.geom_size[volume_id], dtype=float)[:3]
    inner = np.maximum(extent - np.array((.010, .010, 0.0)), np.array((.02, .02, .01)))
    target_id = f"{target_fixture}:inner_cavity"
    floor_top = float(center[2] - extent[2] + .008)
    upper = center + np.array((inner[0], inner[1], extent[2]))
    upper[2] += open_receptacle_vertical_protrusion_allowance(target_fixture)
    target = TargetGeometry(
        target_id, target_fixture,
        (float(center[0] - inner[0]), float(center[1] - inner[1]), floor_top),
        tuple(upper),
        target_id, "inside",
    )
    return PhysicsCaptureContract(
        {"obj": object_geoms},
        {target_id: wall_geoms},
        {target_id: target},
    )


def open_receptacle_vertical_protrusion_allowance(target_fixture: str) -> float:
    """Allow a tall toothbrush COM above the rim while its lower part is in-cup."""
    return .08 if str(target_fixture) == "mouthwash_cup" else 0.0


def _pencil_case_insert_capture_contract(raw: Any, request: SceneRequest) -> PhysicsCaptureContract:
    """Bind the physical pencil and the five-wall pencil-case cavity."""
    return _open_receptacle_insert_capture_contract(
        raw, request, target_object_name="pencil_case", target_fixture="pencil_case",
    )


def _toothbrush_cup_insert_capture_contract(raw: Any, request: SceneRequest) -> PhysicsCaptureContract:
    """Bind the physical toothbrush and the five-wall mouthwash-cup cavity."""
    return _open_receptacle_insert_capture_contract(
        raw, request, target_object_name="mouthwash_cup", target_fixture="mouthwash_cup",
    )


def _ordinary_target_binding(raw: Any, request: SceneRequest, fixture_role: str | None = None) -> tuple[str, tuple[str, ...], str]:
    """Resolve the declared Task-2 destination fixture and semantic target part.

    This deliberately has no first-visible-geom or arbitrary-fixture fallback:
    an unbound source/target relation is not physical evidence.
    """
    role = fixture_role or request.target_fixture
    if not role or not request.target_relation:
        raise EnvironmentValidationError(f"ordinary task has no declared source-target contract: {request.selection_id}")
    # RoboCasa's public task fields use ``cab`` while source semantics use the
    # stable word ``cabinet``.  This is an explicit alias, not a fixture scan.
    aliases = {"cabinet": ("cabinet", "cab")}
    attribute = next((name for name in aliases.get(role, (role,)) if getattr(raw, name, None) is not None), role)
    fixture = getattr(raw, attribute, None)
    prefix = str(getattr(fixture, "naming_prefix", ""))
    if not prefix:
        raise EnvironmentValidationError(
                f"declared target fixture {role} is absent for {request.selection_id}"
        )
    matching = [
        str(name) for name, value in getattr(raw, "fixtures", {}).items()
        if value is fixture
    ]
    if len(matching) != 1:
        raise EnvironmentValidationError(
                f"declared target fixture {role} is ambiguous for {request.selection_id}"
        )
    tokens = TARGET_PART_TOKENS.get(role)
    if not tokens:
        raise EnvironmentValidationError(
            f"declared target fixture has no semantic-part contract: {role}"
        )
    model = raw.sim.model
    geoms = tuple(
        name for index in range(int(model.ngeom))
        if (name := model.geom_id2name(index) or "").startswith(prefix)
        and any(token in name.lower() for token in tokens)
    )
    if not geoms:
        raise EnvironmentValidationError(
            f"declared target fixture has no semantic {request.target_fixture} geom for {request.selection_id}"
        )
    return matching[0], geoms, prefix


def cabinet_inside_cavity_geometry(raw: Any, fixture_prefix: str) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Bind a cabinet insertion to the lowest real support cavity.

    Cabinet ``reg_level`` markers are invisible placement volumes rather than
    collision supports.  The cabinet bottom is the first physical support and
    the first shelf is its ceiling; cabinets without an exposed bottom retain
    the first-shelf / next-shelf fallback.
    """
    bottom: tuple[str, np.ndarray, np.ndarray] | None = None
    shelves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index in range(int(raw.sim.model.ngeom)):
        name = str(raw.sim.model.geom_id2name(index) or "")
        is_bottom = name == f"{fixture_prefix}bottom"
        is_shelf = name.startswith(fixture_prefix) and name.endswith("_shelf")
        if not (is_bottom or is_shelf):
            continue
        if "visual" in name or not (
            int(raw.sim.model.geom_contype[index]) or int(raw.sim.model.geom_conaffinity[index])
        ):
            continue
        rotation = np.asarray(raw.sim.data.geom_xmat[index], dtype=float).reshape(3, 3)
        extent = np.abs(rotation) @ np.asarray(raw.sim.model.geom_size[index], dtype=float)
        center = np.asarray(raw.sim.data.geom_xpos[index], dtype=float)
        item = (name, center - extent, center + extent)
        if is_bottom:
            bottom = item
        else:
            shelves.append(item)
    shelves.sort(key=lambda item: float(item[1][2]))
    if bottom is not None and shelves:
        floor_name, floor_low, floor_high = bottom
        _, ceiling_low, ceiling_high = shelves[0]
    elif len(shelves) >= 2:
        floor_name, floor_low, floor_high = shelves[0]
        _, ceiling_low, ceiling_high = shelves[1]
    else:
        raise EnvironmentValidationError("cabinet insertion requires a physical floor and ceiling")
    lower = floor_low.copy()
    upper = floor_high.copy()
    lower[2] = float(floor_high[2])
    upper[2] = float(ceiling_low[2])
    lower[:2] = np.maximum(floor_low[:2], ceiling_low[:2])
    upper[:2] = np.minimum(floor_high[:2], ceiling_high[:2])
    if np.any(upper <= lower):
        raise EnvironmentValidationError("cabinet shelf cavity has no positive volume")
    return (floor_name,), lower, upper


def fixture_contact_geometry_names(*, semantic_geom: str, all_geom_names: Sequence[str]) -> tuple[str, ...]:
    """Bind the selected movable part and its explicitly named handle geometry.

    RoboCasa models the front panel and handle as sibling bodies.  Restricting
    evidence to the panel body loses a real gripper--handle contact; selecting
    the semantic ``*_door`` stem keeps the binding exact to this one fixture.
    """
    semantic = str(semantic_geom)
    generated = re.match(r"^(.*)_g\d+$", semantic)
    # Only a terminal MuJoCo ``_gN`` suffix denotes one generated collision
    # primitive.  Tokens such as ``main_group_reg_rack0`` are semantic names;
    # splitting them at the final textual ``_g`` broadens a rack binding to
    # the entire appliance and makes unrelated shell/door contacts admissible.
    if generated is None:
        return ()
    stem = generated.group(1)
    return tuple(name for name in all_geom_names if str(name).startswith(stem))


def build_physics_capture_contract(environment: Any, request: SceneRequest | None = None) -> PhysicsCaptureContract:
    """Bind exact source objects, target regions, supports and joints from MuJoCo."""
    raw = _raw_environment(environment)
    request = request or getattr(environment, "request", None)
    if request is None:
        raise EnvironmentValidationError("physics capture requires the originating SceneRequest")
    if request.task_class == "VLA82TrashCanDeposit":
        return _trash_can_capture_contract(raw, request)
    if request.task_class == "VLA82WorkStudyOpenSupport":
        return _work_study_open_support_capture_contract(raw, request)
    if request.task_class == "VLA82NotebookStack":
        return _fixed_object_support_capture_contract(
            raw, request, target_object_name="folder", fixture_id="folder",
        )
    if request.task_class == "VLA82BedsideBookPlace":
        return _fixed_object_support_capture_contract(
            raw, request, target_object_name="nightstand", fixture_id="nightstand",
        )
    if request.task_class == "VLA82PencilCaseInsert":
        return _pencil_case_insert_capture_contract(raw, request)
    if request.task_class == "VLA82ToothbrushCupInsert":
        return _toothbrush_cup_insert_capture_contract(raw, request)
    if request.task_class == "VLA82BathroomShelfPlace":
        return _fixed_object_support_capture_contract(
            raw, request, target_object_name="bathroom_shelf", fixture_id="bathroom_shelf",
        )
    object_geoms: dict[str, tuple[str, ...]] = {}
    for object_id, obj in getattr(raw, "objects", {}).items():
        prefix = str(getattr(obj, "naming_prefix", ""))
        names = tuple(
            name for index in range(int(raw.sim.model.ngeom))
            if (name := raw.sim.model.geom_id2name(index) or "") and prefix and name.startswith(prefix)
        )
        if names:
            object_geoms[str(object_id)] = names
    binding: dict[str, str] | None = None
    fixture_contact_geoms: tuple[str, ...] = ()
    if request.primary_asset.kind == "fixture_part":
        binding = fixture_binding(raw, request)
        # The selected movable fixture *part* is the manipulated entity.  Do
        # not bind every geom under an appliance prefix: a stove can contain
        # several knobs and doors, and only the body selected by Task 2 is
        # admissible evidence for this operation.
        semantic_body = int(raw.sim.model.geom_bodyid[raw.sim.model.geom_name2id(binding["semantic_geom"])])
        fixture_geoms = tuple(
            name for index in range(int(raw.sim.model.ngeom))
            if (name := raw.sim.model.geom_id2name(index) or "")
            and int(raw.sim.model.geom_bodyid[index]) == semantic_body
            and float(raw.sim.model.geom_rgba[index][3]) > 0.01
        )
        # Drawer handles are physically separate child bodies.  Include only
        # the selected semantic door stem, never a same-class fixture elsewhere
        # in the kitchen, so contact evidence follows the controller target.
        semantic_stem_geoms = fixture_contact_geometry_names(
            semantic_geom=binding["semantic_geom"],
            all_geom_names=tuple(raw.sim.model.geom_id2name(index) or "" for index in range(int(raw.sim.model.ngeom))),
        )
        if semantic_stem_geoms:
            fixture_geoms = semantic_stem_geoms
        if not fixture_geoms:
            raise EnvironmentValidationError(f"fixture operated body has no visible geom for {request.selection_id}")
        object_geoms["obj"] = fixture_geoms
        fixture_contact_geoms = fixture_geoms
    if "obj" not in object_geoms:
        raise EnvironmentValidationError("physics capture cannot bind primary source object geom")
    target_geoms: dict[str, tuple[str, ...]] = {}
    targets: dict[str, TargetGeometry] = {}
    joint_ids: dict[str, str] = {}
    target_names: tuple[str, ...] = ()
    target_bounds: tuple[np.ndarray, np.ndarray] | None = None
    target_role = request.target_fixture
    if binding is None:
        # Source-labelled cleaning and return phases operate on the live work
        # surface, not the generic pick-place task's cabinet destination.
        if set(request.primary_asset.affordances) & {"wipe", "scrub", "spray"}:
            target_role = request.source_fixture or "counter"
        fixture_name, target_names, fixture_prefix = _ordinary_target_binding(raw, request, target_role)
        if target_role == "drawer" and request.target_relation == "inside":
            # A drawer's generic "bottom" token also matches outer carcass
            # and decorative geometry.  Its union truncates the admissible
            # Z band at the thin bottom plate, rejecting an object that is
            # physically resting inside. Bind only the live collision walls
            # that actually define the opened inner cavity.
            target_names = tuple(
                name for index in range(int(raw.sim.model.ngeom))
                if (name := raw.sim.model.geom_id2name(index) or "").startswith(fixture_prefix)
                and "inner_" in name
                and "visual" not in name
                and any(token in name for token in ("inner_bottom", "inner_left", "inner_right", "inner_back"))
            )
            if not target_names:
                raise EnvironmentValidationError(
                    f"declared drawer target has no live inner collision cavity for {request.selection_id}"
                )
        elif target_role == "cabinet" and request.target_relation == "inside":
            target_names, lower, upper = cabinet_inside_cavity_geometry(raw, fixture_prefix)
            target_bounds = (lower, upper)
        binding = {
            "fixture_name": fixture_name,
            "fixture_prefix": fixture_prefix,
            "semantic_geom": target_names[0], "geom": target_names[0], "joint": "",
        }
    if binding is not None:
        target_names = target_names or (binding["semantic_geom"],)
        geom_ids = tuple(raw.sim.model.geom_name2id(geom) for geom in target_names)
        centers = np.asarray([raw.sim.data.geom_xpos[geom_id] for geom_id in geom_ids], dtype=float)
        sizes = np.asarray([raw.sim.model.geom_size[geom_id] for geom_id in geom_ids], dtype=float)
        if target_role == "drawer" and request.target_relation == "inside":
            rotations = np.asarray([
                raw.sim.data.geom_xmat[geom_id].reshape(3, 3) for geom_id in geom_ids
            ], dtype=float)
            extents = np.asarray([
                np.abs(rotation) @ size[:3] for rotation, size in zip(rotations, sizes)
            ], dtype=float)
        else:
            extents = np.maximum(sizes[:, :3], np.array((0.02, 0.02, 0.02)))
        min_corner = np.min(centers - extents, axis=0)
        max_corner = np.max(centers + extents, axis=0)
        if target_bounds is not None:
            min_corner, max_corner = target_bounds
        target_id = (
            f"{binding['fixture_name']}:{target_role}:tabletop"
            if request.primary_asset.kind != "fixture_part"
            else f"{binding['fixture_name']}:{binding['semantic_geom']}"
        )
        target_geoms[target_id] = target_names
        phases = set(request.primary_asset.affordances)
        relation = (
            "clean-region" if phases & {"wipe", "scrub", "spray"}
            else request.target_relation if request.primary_asset.kind != "fixture_part"
            else "joint-target" if binding["joint"]
            else "clean-region" if phases & {"wipe", "scrub", "spray"}
            else "above" if "stack" in phases
            else "beside" if "arrange" in phases
            else "inside" if phases & {"insert", "deposit"}
            else "on"
        )
        if relation == "clean-region":
            # A cleaning region is the physical support *and* the admissible
            # centre-of-mass band for a tool resting on it. The raw tabletop
            # geom itself is thin, while a sponge's body origin is above it.
            min_corner[2] -= 0.02
            max_corner[2] += 0.20
        targets[target_id] = TargetGeometry(
            target_id, binding["fixture_name"], tuple(min_corner), tuple(max_corner), target_id, relation
        )
        if binding["joint"]:
            for phase in request.primary_asset.affordances:
                if phase in {"open", "close", "close_lid", "pull", "push", "turn_knob"}:
                    joint_ids[phase] = binding["joint"]
    if not targets:
        raise EnvironmentValidationError(f"physics capture target binding absent for {request.selection_id}")
    # For appliance actions contact must occur on the exact selected movable
    # body before the selected MuJoCo joint changes.  Mapping its full visible
    # body (rather than merely one decorative geom) preserves evidence without
    # accepting contact with another same-class fixture part.
    contact_joints = {
        geom: joint
        for joint in joint_ids.values()
        for geom in fixture_contact_geoms
    }
    return PhysicsCaptureContract(object_geoms, target_geoms, targets, contact_joints, joint_ids)


def scene_runtime_fingerprint(environment: Any, request: SceneRequest | None = None) -> dict[str, Any]:
    """Hash live reset provenance without mutating any MuJoCo state."""
    raw = _raw_environment(environment)
    request = request or getattr(environment, "request", None)
    if request is None:
        raise EnvironmentValidationError("runtime scene fingerprint requires SceneRequest")
    contract = build_physics_capture_contract(environment, request)
    robot = raw.robots[0]
    model = raw.sim.model
    eef = np.asarray(raw.sim.data.site_xpos[robot.eef_site_id["right"]], dtype=float)
    selected_geoms = tuple(contract.object_geom_names.get("obj", ()))
    geom_positions = {
        name: np.asarray(raw.sim.data.geom_xpos[model.geom_name2id(name)], dtype=float).round(9).tolist()
        for name in selected_geoms
    }
    target_geometries = {
        target_id: {
            "fixture_id": target.fixture_id,
            "min_corner": np.asarray(target.min_corner, dtype=float).round(9).tolist(),
            "max_corner": np.asarray(target.max_corner, dtype=float).round(9).tolist(),
            "spatial_relation": target.spatial_relation,
        }
        for target_id, target in sorted(contract.target_geometries.items())
    }
    joint_values = {
        joint: float(raw.sim.data.qpos[int(model.jnt_qposadr[model.joint_name2id(joint)])])
        for joint in set(contract.fixture_joint_ids.values()) if joint
    }
    fixture = fixture_binding(raw, request) if request.primary_asset.kind == "fixture_part" else {}
    model_surface = {
        "geom_names": [model.geom_id2name(index) or "" for index in range(int(model.ngeom))],
        "joint_names": [model.joint_id2name(index) or "" for index in range(int(model.njnt))],
        "body_names": [model.body_id2name(index) or "" for index in range(int(model.nbody))],
    }
    model_sha = hashlib.sha256(json.dumps(model_surface, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    payload = {
        "selection_id": request.selection_id,
        "layout_id": getattr(raw, "layout_id", None),
        "style_id": getattr(raw, "style_id", None),
        "task_refs": getattr(raw, "_ep_meta", {}).get("task_refs", {}),
        "live_knob": getattr(raw, "knob", None),
        "fixture": fixture,
        "fixture_joint_ids": dict(contract.fixture_joint_ids),
        "fixture_joint_values": joint_values,
        "selected_geoms": geom_positions,
        "target_geometries": target_geometries,
        "eef_world": eef.round(9).tolist(),
        "robot_qpos": np.asarray(raw.sim.data.qpos, dtype=float)[:13].round(9).tolist(),
        "model_sha256": model_sha,
        "friction_evidence": getattr(environment, "friction_evidence", {}),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return {**payload, "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def snapshot_from_environment(environment: Any, step: int, *, contract: PhysicsCaptureContract | None = None) -> PhysicsSnapshot:
    """Extract physics observables only; it never mutates MuJoCo state."""
    raw = _raw_environment(environment)
    robot = raw.robots[0]
    contract = contract or build_physics_capture_contract(environment)
    qpos = np.asarray(raw.sim.data.qpos)
    robot_indexes = _flatten_qpos_indexes(getattr(robot, "_ref_joint_pos_indexes", ()))
    gripper_indexes = _flatten_qpos_indexes(getattr(robot, "_ref_gripper_joint_pos_indexes", ()))
    body_poses: dict[str, np.ndarray] = {}
    for name, obj in getattr(raw, "objects", {}).items():
        body_id = raw.sim.model.body_name2id(obj.root_body)
        body_poses[str(name)] = np.concatenate(
            (np.asarray(raw.sim.data.body_xpos[body_id]), np.asarray(raw.sim.data.body_xquat[body_id]))
        )
    # Fixture-operated selections have no entry in ``raw.objects``.  Capture
    # the selected exact fixture body as ``obj`` so all position, stability,
    # and joint predicates consume real MuJoCo state rather than an empty
    # placeholder pose.
    if "obj" not in body_poses and contract.object_geom_names.get("obj"):
        geom_id = raw.sim.model.geom_name2id(contract.object_geom_names["obj"][0])
        body_id = int(raw.sim.model.geom_bodyid[geom_id])
        body_poses["obj"] = np.concatenate(
            (np.asarray(raw.sim.data.body_xpos[body_id]), np.asarray(raw.sim.data.body_xquat[body_id]))
        )
    contacts: list[tuple[str, str, float]] = []
    named_contacts: list[ContactEvidence] = []
    for index in range(int(raw.sim.data.ncon)):
        contact = raw.sim.data.contact[index]
        geom_a = raw.sim.model.geom_id2name(contact.geom1) or str(contact.geom1)
        geom_b = raw.sim.model.geom_id2name(contact.geom2) or str(contact.geom2)
        contacts.append((geom_a, geom_b, float(contact.dist)))
        names = {geom_a, geom_b}
        object_id = next((key for key, geoms in contract.object_geom_names.items() if names & set(geoms)), None)
        target_id = next((key for key, geoms in contract.target_geom_names.items() if names & set(geoms)), None)
        joint_id = next((joint for geom, joint in contract.fixture_contact_joints.items() if geom in names), None)
        fixture_id = contract.target_geometries[target_id].fixture_id if target_id in contract.target_geometries else None
        named_contacts.append(ContactEvidence(geom_a, geom_b, object_id, target_id, fixture_id, joint_id, float(contact.dist)))
    joint_positions = {
        raw.sim.model.joint_id2name(index) or str(index): float(qpos[address])
        for index, address in enumerate(np.asarray(raw.sim.model.jnt_qposadr))
    }
    return PhysicsSnapshot(
        step=step,
        robot_qpos=qpos[robot_indexes].copy(),
        gripper_qpos=qpos[gripper_indexes].copy(),
        body_poses=body_poses,
        joint_positions=joint_positions,
        contacts=tuple(contacts),
        dirt_fraction=float(getattr(raw, "dirt_fraction", 1.0)),
        spray_coverage=float(getattr(raw, "spray_coverage", 0.0)),
        dispensed_amount=float(getattr(raw, "dispensed_amount", 0.0)),
        contact_evidence=tuple(named_contacts),
        target_geometries=dict(contract.target_geometries),
        fixture_joint_ids=dict(contract.fixture_joint_ids),
    )
