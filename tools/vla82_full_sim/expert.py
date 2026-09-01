"""Step-only expert demonstrations for the VLA82 source operation contracts.

This module is deliberately an *expert data* producer.  Its artifacts always
carry ``training_expert=True`` and are rejected by acceptance code later in the
pipeline.  It never writes simulator state: every command is handed to the
public Gym/RoboCasa ``env.step`` API and every claimed result is re-evaluated
from captured MuJoCo observables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tools.pick_place_oracle.open_gripper import OpenGripperPickPlaceOracle
from tools.pick_place_oracle import PickPlaceSnapshot

from .annotations import OperationSpec, sha256_file
from .environment import (
    EnvironmentValidationError,
    PhysicsSnapshot,
    SceneRequest,
    build_physics_capture_contract,
    is_counter_to_drawer_request,
    make_environment,
    scene_runtime_fingerprint,
    snapshot_from_environment,
    uses_cabinet_source_alignment,
)
from .predicates import PredicateResult, evaluate_operation


EXPERT_SCHEMA_VERSION = 1
PHASE_INDEX = {
    "grasp": 0, "place": 1, "return": 2, "deposit": 3, "wipe": 4,
    "scrub": 5, "spray": 6, "insert": 7, "stack": 8, "arrange": 9,
    "open": 10, "close": 11, "pull": 12, "push": 13,
    "turn_knob": 14, "press": 15, "close_lid": 16,
}


@dataclass(frozen=True)
class ActionLayout:
    """Public-action slots derived from the live CompositeController."""

    arm: tuple[int, int]
    torso: int
    base: tuple[int, int]
    gripper: int
    base_mode: int

    @classmethod
    def from_env(cls, environment: Any) -> "ActionLayout":
        raw = _raw_environment(environment)
        controller = raw.robots[0].composite_controller
        splits = dict(controller._action_split_indexes)
        required = ("right", "torso", "base", "right_gripper")
        if any(name not in splits for name in required):
            raise ValueError(f"PandaOmron controller layout missing parts: {sorted(set(required) - set(splits))}")
        arm, torso, base, gripper = splits["right"], splits["torso"], splits["base"], splits["right_gripper"]
        action_dim = int(np.prod(environment.action_space.shape))
        if arm[1] - arm[0] != 6 or torso[1] - torso[0] != 1 or base[1] - base[0] != 3 or gripper[1] - gripper[0] != 1:
            raise ValueError(f"unexpected PandaOmron controller dimensions: {splits}")
        mode = action_dim - 1
        if max(arm[1], torso[1], base[1], gripper[1]) > mode:
            raise ValueError(f"invalid controller layout for {action_dim}-D action: {splits}")
        return cls(tuple(arm), int(torso[0]), tuple(base), int(gripper[0]), mode)

    @classmethod
    def canonical(cls, action_dim: int) -> "ActionLayout":
        if action_dim < 12:
            raise ValueError(f"PandaOmron action vector must have at least 12 values, got {action_dim}")
        return cls((0, 6), 6, (7, 10), 10, action_dim - 1)


@dataclass(frozen=True)
class WholeBodyActionLayout:
    """Action slots reported by RoboSuite's WholeBodyIK controller itself."""

    right: tuple[int, int]
    torso: tuple[int, int]
    base: tuple[int, int]
    right_gripper: tuple[int, int]

    @classmethod
    def from_controller(cls, controller: Any, *, action_dim: int) -> "WholeBodyActionLayout":
        splits = dict(getattr(controller, "_whole_body_controller_action_split_indexes", {}))
        required = ("right", "torso", "base", "right_gripper")
        if any(name not in splits for name in required):
            raise ValueError(f"WholeBodyIK layout missing parts: {sorted(set(required) - set(splits))}")
        layout = cls(*(tuple(int(item) for item in splits[name]) for name in required))
        expected = {"right": 6, "torso": 1, "base": 3, "right_gripper": 1}
        for name, span in zip(required, (layout.right, layout.torso, layout.base, layout.right_gripper)):
            if span[1] - span[0] != expected[name] or span[0] < 0 or span[1] > action_dim:
                raise ValueError(f"unexpected WholeBodyIK {name} split {span} for {action_dim}-D action")
        return layout


def whole_body_pose_action(
    controller: Any, layout: WholeBodyActionLayout, *, position: np.ndarray, axis_angle: np.ndarray,
    torso_qpos: float | np.ndarray, closed: bool,
) -> np.ndarray:
    """Build one public absolute WholeBodyIK command through RoboSuite's API."""
    right = np.concatenate((np.asarray(position, dtype=np.float32).reshape(3), np.asarray(axis_angle, dtype=np.float32).reshape(3)))
    torso = np.asarray(torso_qpos, dtype=np.float32).reshape(-1)
    action_dict = {
        "right": right,
        "torso": torso,
        "base": np.zeros(layout.base[1] - layout.base[0], dtype=np.float32),
        "right_gripper": np.full(layout.right_gripper[1] - layout.right_gripper[0], 1.0 if closed else -1.0, dtype=np.float32),
    }
    if torso.size != layout.torso[1] - layout.torso[0]:
        raise ValueError(f"WholeBodyIK torso target must have {layout.torso[1] - layout.torso[0]} values, got {torso.size}")
    action = np.asarray(controller.create_action_vector(action_dict), dtype=np.float32).reshape(-1)
    expected_dim = max(layout.right[1], layout.torso[1], layout.base[1], layout.right_gripper[1])
    if action.size != expected_dim:
        raise ValueError(f"WholeBodyIK action builder returned {action.size} values, expected {expected_dim}")
    return action


def expand_openvla_action(action: np.ndarray, action_dim: int, *, layout: ActionLayout | None = None) -> np.ndarray:
    """Map a public 7-D expert command into PandaOmron's 12-D vector.

    The live PandaOmron composite ordering is arm pose (0:6), torso (6),
    mobile base (7:10), gripper (10), then base-mode (11).  This explicit conversion
    prevents a source expert's gripper command being accidentally written to
    the base-mode slot.
    """
    source = np.asarray(action, dtype=np.float32).reshape(-1)
    if source.shape != (7,):
        raise ValueError(f"OpenVLA expert action must be 7-D, got {source.shape}")
    resolved = layout or ActionLayout.canonical(action_dim)
    if action_dim <= resolved.base_mode:
        raise ValueError(f"PandaOmron action vector does not contain declared base-mode slot {resolved.base_mode}")
    result = np.zeros(action_dim, dtype=np.float32)
    result[resolved.arm[0]:resolved.arm[1]] = source[:6]
    result[resolved.gripper] = source[6]
    result[resolved.base_mode] = -1.0
    return result


@dataclass(frozen=True)
class EpisodeReport:
    selection_id: str
    seed: int
    status: str
    training_expert: bool
    npz_path: str
    diagnostic_path: str
    predicate_success: bool
    errors: tuple[str, ...]
    spec_sha256: str
    scene_sha256: str
    asset_sha256: str
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> bytes:
    if is_dataclass(value):
        value = asdict(value)
    elif not isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
        # An opaque test/simulator object has no stable state contract.  Its
        # class identity is stable while repr() usually embeds memory address.
        value = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def spec_fingerprint(spec: OperationSpec) -> str:
    return _sha(asdict(spec))


def scene_fingerprint(scene: Any) -> str:
    return _sha(scene)


def asset_fingerprint(scene: Any) -> str:
    asset = getattr(scene, "primary_asset", None)
    evidence = getattr(asset, "evidence_path", "") if asset is not None else ""
    if evidence and Path(str(evidence)).is_file():
        return sha256_file(Path(str(evidence)))
    return _sha(asset if asset is not None else {"scene_type": type(scene).__name__})


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
    temporary.write_text(json.dumps(dict(data), ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frame(observation: Mapping[str, Any], camera: str, fallback: np.ndarray | None = None) -> np.ndarray:
    value = observation.get(f"video.{camera}")
    if value is None:
        value = observation.get(f"{camera}_image")
    if value is None:
        return np.zeros_like(fallback) if fallback is not None else np.zeros((64, 64, 3), dtype=np.uint8)
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"camera {camera} is not an RGB image")
    # MuJoCo's offscreen buffer uses a bottom-left image origin.  Stored MP4
    # evidence uses the conventional top-left origin expected by video tools.
    return np.flipud(array).copy()


def _proprio(snapshot: Any) -> np.ndarray:
    robot = np.asarray(getattr(snapshot, "robot_qpos", ()), dtype=np.float32).reshape(-1)
    gripper = np.asarray(getattr(snapshot, "gripper_qpos", ()), dtype=np.float32).reshape(-1)
    result = np.zeros(8, dtype=np.float32)
    result[: min(7, robot.size)] = robot[:7]
    result[7] = float(gripper.mean()) if gripper.size else 0.0
    return result


def _snapshot_json(snapshot: Any) -> dict[str, Any]:
    if hasattr(snapshot, "step"):
        return {
            "step": int(snapshot.step),
            "gripper_qpos": np.asarray(getattr(snapshot, "gripper_qpos", ()), dtype=float).round(8).tolist(),
            "object_pose": np.asarray(getattr(snapshot, "body_poses", {}).get("obj", ()), dtype=float).round(8).tolist(),
            "body_poses": {
                str(name): np.asarray(pose, dtype=float).round(8).tolist()
                for name, pose in getattr(snapshot, "body_poses", {}).items()
            },
            "contacts": [list(row) for row in getattr(snapshot, "contacts", ())],
            "joints": {str(k): float(v) for k, v in getattr(snapshot, "joint_positions", {}).items()},
            "dirt_fraction": float(getattr(snapshot, "dirt_fraction", 1.0)),
            "spray_coverage": float(getattr(snapshot, "spray_coverage", 0.0)),
            "dispensed_amount": float(getattr(snapshot, "dispensed_amount", 0.0)),
        }
    return {"step": int(snapshot.get("step", 0))} if isinstance(snapshot, dict) else {"snapshot": str(snapshot)}


def _action_bounds(environment: Any) -> tuple[np.ndarray, np.ndarray]:
    space = environment.action_space
    return np.asarray(space.low, dtype=np.float32), np.asarray(space.high, dtype=np.float32)


class CleaningCoverageTracker:
    """Read-only episode evidence derived from exact MuJoCo contact records.

    RoboCasa's generic pick/place scene has no mutable dirt field.  This
    tracker therefore does *not* touch ``raw`` or any environment metric.  It
    converts already-observed, exact object--declared-surface contacts into a
    2-cm capture-grid record.  The immutable ``PhysicsSnapshot`` returned to
    the predicate carries the corresponding remaining-area fraction, so every
    claimed reduction is auditable from the same raw contacts and poses.
    """

    def __init__(self, target_id: str, *, cell_size: float = 0.02, reduction_per_cell: float = 0.10):
        self.target_id = str(target_id)
        self.cell_size = float(cell_size)
        self.reduction_per_cell = float(reduction_per_cell)
        self._cells: set[tuple[int, int]] = set()
        self.evidence: list[dict[str, Any]] = []

    @property
    def covered_cells(self) -> int:
        return len(self._cells)

    def derive(self, snapshot: PhysicsSnapshot, *, credit: bool = True) -> PhysicsSnapshot:
        target = snapshot.target_geometries.get(self.target_id)
        pose = snapshot.body_poses.get("obj")
        exact_contact = any(
            contact.object_id == "obj" and contact.target_id == self.target_id
            for contact in snapshot.contact_evidence
        )
        if credit and exact_contact and target is not None and pose is not None:
            point = np.asarray(pose, dtype=float)[:3]
            lower = np.asarray(target.min_corner, dtype=float)
            upper = np.asarray(target.max_corner, dtype=float)
            if bool(np.all(point >= lower - 1e-6) and np.all(point <= upper + 1e-6)):
                cell = tuple(np.floor(point[:2] / self.cell_size).astype(int))
                new_cell = cell not in self._cells
                self._cells.add(cell)
                self.evidence.append({
                    "step": int(snapshot.step), "target_id": self.target_id,
                    "cell": list(cell), "object_world": point.round(6).tolist(),
                    "new_cell": new_cell,
                })
        remaining = max(0.0, 1.0 - self.reduction_per_cell * self.covered_cells)
        return replace(snapshot, dirt_fraction=float(remaining))


def cleaning_coverage_points(source_position: np.ndarray, target: Any) -> tuple[np.ndarray, ...]:
    """Return a bounded, separated wipe grid on the declared live surface.

    The controller must physically reach each point before it advances; these
    are targets only, never simulator-state writes.  Four corners provide
    redundancy while guaranteeing at least three 2-cm evidence cells for a
    normally sized counter cleaning region.
    """
    lower = np.asarray(target.min_corner, dtype=float)
    upper = np.asarray(target.max_corner, dtype=float)
    center = np.asarray(source_position, dtype=float).copy()
    safe_margin = min(0.04, max(0.005, float(np.min((upper[:2] - lower[:2]) / 5.0))))
    center[:2] = np.minimum(np.maximum(center[:2], lower[:2] + safe_margin), upper[:2] - safe_margin)
    half_span = np.minimum(0.05, np.maximum(0.012, (upper[:2] - lower[:2] - 2.0 * safe_margin) / 2.0))
    # The calibrated PandaOmron short probe retained the tool for a pure
    # world-X transit.  Start directly above the source contact point, then
    # sweep that proven axis rather than a diagonal reach.
    # First-contact cell plus two immediately adjacent 2.5-cm cells: enough
    # for the 3-cell predicate without spending grasp lifetime on a long sweep.
    half_span[0] = min(float(half_span[0]), 0.025)
    offsets = ((0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0))
    points: list[np.ndarray] = []
    for offset in offsets:
        point = center.copy()
        point[:2] += half_span * np.asarray(offset)
        point[:2] = np.minimum(np.maximum(point[:2], lower[:2] + 1e-4), upper[:2] - 1e-4)
        points.append(point)
    return tuple(points)


def cleaning_contact_target(
    point: Sequence[float], grasp_offset: Sequence[float], *,
    contact_eef_z: float, target_contact: bool, reseat_step: float = .0005,
) -> np.ndarray:
    """Keep wipe XY tracking while preserving or gently restoring surface contact."""
    desired = np.asarray(point, dtype=float) + np.asarray(grasp_offset, dtype=float)
    desired[2] = float(contact_eef_z) - (0.0 if bool(target_contact) else float(reseat_step))
    return desired


def should_recover_wipe_grasp(*, grasped: bool, exact_gripper_contact: bool, attempts: int, wipe_complete: bool = False) -> bool:
    """Permit at most two physical regrasp recoveries before wipe evidence."""
    return not bool(wipe_complete) and not (bool(grasped) and bool(exact_gripper_contact)) and int(attempts) < 2


def held_for_cleaning_credit(*, raw_grasp: bool, two_pad_contact: bool) -> bool:
    """Coverage is valid only for a live grasp with both physical finger pads."""
    return bool(raw_grasp) and bool(two_pad_contact)


def compliant_descend_limit(bottom_gap: float) -> float:
    """Two-stage bounded Z controller: coarse above 3 mm, micro at contact."""
    return 0.05 if float(bottom_gap) > 0.003 else 0.01


def compliant_descend_command(bottom_gap: float) -> float:
    """Gap-scheduled Panda Z command, from fast free-space to contact microstep."""
    if float(bottom_gap) > 0.01:
        return -0.70
    if float(bottom_gap) > 0.003:
        return -0.10
    return -0.01


def oracle_horizontal_action(
    controller: Any, eef: np.ndarray, target: np.ndarray, low: np.ndarray, high: np.ndarray,
    *, layout: ActionLayout | None = None, limit: float = 1.0,
) -> np.ndarray:
    """Exact OpenGripperPickPlaceOracle closed-transit action in 12-D layout."""
    horizontal_target = np.asarray(target, dtype=float).copy()
    horizontal_target[2] = float(np.asarray(eef, dtype=float)[2])
    oracle = OpenGripperPickPlaceOracle(horizontal_target, horizontal_target, horizontal_target, world_to_origin=controller.world_to_origin_frame)
    decision = oracle._motion(PickPlaceSnapshot(np.asarray(eef, dtype=float), horizontal_target, True, False), horizontal_target, closed=True)
    action = expand_openvla_action(np.asarray(decision.action, dtype=np.float32), int(low.size), layout=layout)
    action[:3] = np.clip(action[:3], -float(limit), float(limit))
    return np.clip(action, low, high)


class PrimitiveExpert:
    """Phase-labelled, bounded commands passed exclusively to ``env.step``.

    These neutral guarded motions provide an initial source-labelled dataset.
    They intentionally do not assert success: Task 3 predicates remain the
    authority and failed scenes are retained for later expert/DAgger repair.
    """

    def __init__(self, phase: str, index: int):
        self.phase = phase
        self.index = index

    def actions(self, environment: Any) -> Iterable[np.ndarray]:
        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        steps = 8 if self.phase in {"wipe", "scrub", "turn_knob", "open", "close", "pull", "push"} else 5
        for local in range(steps):
            action = np.zeros_like(low, dtype=np.float32)
            # RoboCasa PandaOmron accepts mobile-base / arm / gripper controls.
            # A phase-specific, bounded low-amplitude excitation is safer than
            # mutating qpos and creates authentic step provenance for diagnosis.
            direction = 1.0 if (local + self.index) % 2 == 0 else -1.0
            action[: min(6, action.size)] = direction * 0.12
            if action.size > layout.gripper:
                action[layout.gripper] = 0.35 if self.phase == "grasp" else -0.35
            action[layout.base_mode] = -1.0
            yield np.clip(action, low, high)


def _raw_environment(environment: Any) -> Any:
    return getattr(getattr(environment, "unwrapped", environment), "env", environment)


def _selected_geom_has_gripper_contact(raw: Any, geom_name: str | Sequence[str]) -> bool:
    """Read exact MuJoCo contact provenance without mutating simulator state."""
    try:
        selected = {geom_name} if isinstance(geom_name, str) else set(geom_name)
        for index in range(int(raw.sim.data.ncon)):
            contact = raw.sim.data.contact[index]
            names = {
                str(raw.sim.model.geom_id2name(contact.geom1) or ""),
                str(raw.sim.model.geom_id2name(contact.geom2) or ""),
            }
            if selected & names and any(("gripper" in name.lower() or "finger" in name.lower()) for name in names):
                return True
    except (AttributeError, TypeError):
        return False
    return False


def selected_geoms_have_contact(
    raw: Any, first_geoms: Sequence[str], second_geoms: Sequence[str],
) -> bool:
    """Read an exact contact between two named MuJoCo geometry sets."""
    first = {str(name) for name in first_geoms}
    second = {str(name) for name in second_geoms}
    try:
        for index in range(int(raw.sim.data.ncon)):
            contact = raw.sim.data.contact[index]
            name1 = str(raw.sim.model.geom_id2name(int(contact.geom1)) or "")
            name2 = str(raw.sim.model.geom_id2name(int(contact.geom2)) or "")
            if (name1 in first and name2 in second) or (name2 in first and name1 in second):
                return True
    except (AttributeError, TypeError, IndexError):
        return False
    return False


def is_mobile_base_fixed_fixture_contact(names: Sequence[str]) -> bool:
    """Whether one contact is the mobile base against fixed scene furniture.

    Ground, the manipulated object, and the robot's own geometry are excluded;
    every other named geometry is an obstacle for a mobile-base route.  Keeping
    this rule independent of fixture name prevents new scene fixtures (such as
    an oven) from silently bypassing a cabinet-only collision gate.
    """
    lowered = tuple(str(name or "").lower() for name in names)
    if not any("mobilebase" in name for name in lowered):
        return False
    ignored = ("mobilebase", "floor", "obj_", "robot", "panda", "gripper", "finger")
    return any(name and not any(token in name for token in ignored) for name in lowered)


def is_material_mobile_base_fixture_contact(
    names: Sequence[str], *, distance: float, penetration_threshold: float = -.001,
) -> bool:
    """Reject millimetre-scale base collisions, not solver-level grazing noise."""
    return bool(
        is_mobile_base_fixed_fixture_contact(names)
        and float(distance) <= float(penetration_threshold)
    )


def _selected_geom_has_two_finger_contacts(raw: Any, geom_name: str | Sequence[str]) -> bool:
    try:
        selected = {geom_name} if isinstance(geom_name, str) else set(geom_name)
        fingers: set[str] = set()
        for index in range(int(raw.sim.data.ncon)):
            contact = raw.sim.data.contact[index]
            names = (str(raw.sim.model.geom_id2name(contact.geom1) or ""), str(raw.sim.model.geom_id2name(contact.geom2) or ""))
            if selected.intersection(names):
                fingers.update(name for name in names if "finger" in name.lower() and "collision" in name.lower())
        return len(fingers) >= 2
    except (AttributeError, TypeError):
        return False


def is_gripper_pad_geometry_name(name: str) -> bool:
    """Recognize pad collision geoms from supported official grippers."""
    lowered = str(name).lower()
    return "pad_collision" in lowered and (
        "finger" in lowered or "f1_pad_collision" in lowered or "f2_pad_collision" in lowered or "f3_pad_collision" in lowered
    )


def make_selected_gripper_contact_detail(
    *, names: Sequence[str], normal: Sequence[float], distance: float,
    object_geoms: Sequence[str],
) -> dict[str, Any] | None:
    """Normalize one selected object/gripper contact for diagnostic traces."""
    first, second = str(names[0]), str(names[1])
    selected = set(object_geoms)
    vector = np.asarray(normal, dtype=float)
    if first.startswith("gripper") and second in selected:
        gripper_geom, object_geom = first, second
    elif second.startswith("gripper") and first in selected:
        gripper_geom, object_geom = second, first
        vector = -vector
    else:
        return None
    return {
        "gripper_geom": gripper_geom,
        "object_geom": object_geom,
        "normal_gripper_to_object": vector.round(6).tolist(),
        "distance": round(float(distance), 6),
    }


def selected_gripper_contact_details(raw: Any, object_geoms: Sequence[str]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    try:
        for index in range(int(raw.sim.data.ncon)):
            contact = raw.sim.data.contact[index]
            detail = make_selected_gripper_contact_detail(
                names=(
                    str(raw.sim.model.geom_id2name(contact.geom1) or ""),
                    str(raw.sim.model.geom_id2name(contact.geom2) or ""),
                ),
                normal=np.asarray(contact.frame[:3], dtype=float),
                distance=float(contact.dist), object_geoms=object_geoms,
            )
            if detail is not None:
                details.append(detail)
    except (AttributeError, IndexError, TypeError, ValueError):
        return []
    return details


def selected_geom_has_two_finger_pad_contacts(raw: Any, geom_names: Sequence[str]) -> bool:
    """Match the exact pad groups used by RoboSuite's native grasp check."""
    try:
        selected, pads = set(geom_names), set()
        for index in range(int(raw.sim.data.ncon)):
            contact = raw.sim.data.contact[index]
            names = (str(raw.sim.model.geom_id2name(contact.geom1) or ""), str(raw.sim.model.geom_id2name(contact.geom2) or ""))
            if selected.intersection(names):
                pads.update(name for name in names if is_gripper_pad_geometry_name(name))
        return len(pads) >= 2
    except (AttributeError, TypeError):
        return False


def needs_flat_tool_reorientation(raw: Any, object_geoms: Sequence[str]) -> bool:
    """Select the vertical-pinch preparation from live collision geometry."""
    try:
        sizes = np.asarray(
            [raw.sim.model.geom_size[raw.sim.model.geom_name2id(name)][:3] for name in object_geoms],
            dtype=float,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if sizes.size == 0:
        return False
    extent = np.max(sizes, axis=0)
    return bool(max(extent[0], extent[1]) >= 0.04 and extent[2] <= 0.03)


def knob_turn_should_stop(
    *, initial_joint: float, current_joint: float,
    required_delta: float, contact_observed: bool, exact_contact: bool = True,
) -> bool:
    """Stop emitting wrist motion once the contacted knob reaches its goal."""
    return bool(contact_observed) and bool(exact_contact) and abs(float(current_joint) - float(initial_joint)) >= abs(float(required_delta))


def knob_required_turn_delta(
    joint_range: Sequence[float], *, visible_fraction: float = .10, minimum_delta: float = .20,
) -> float:
    """Require a clearly visible fraction of the physical rotary range.

    A small fixed delta can make a controller trace look complete while leaving
    a wide-range appliance control visually indistinguishable from its start
    state.  The threshold is derived from MuJoCo's declared hinge range and is
    read-only; it does not change fixture state.
    """
    bounds = np.asarray(joint_range, dtype=float).reshape(-1)
    if bounds.size < 2 or not np.all(np.isfinite(bounds)):
        return float(minimum_delta)
    return max(float(minimum_delta), float(visible_fraction) * abs(float(bounds[1]) - float(bounds[0])))


def knob_collateral_is_stable(
    *, initial_position: Sequence[float], final_position: Sequence[float],
    maximum_horizontal_motion: float = .02, maximum_vertical_drop: float = .015,
) -> bool:
    """Check that nearby cookware was not displaced while operating a knob."""
    start = np.asarray(initial_position, dtype=float).reshape(-1)
    finish = np.asarray(final_position, dtype=float).reshape(-1)
    if start.size < 3 or finish.size < 3 or not np.all(np.isfinite(start[:3])) or not np.all(np.isfinite(finish[:3])):
        return False
    horizontal = float(np.linalg.norm(finish[:2] - start[:2]))
    vertical_drop = float(start[2] - finish[2])
    return horizontal <= float(maximum_horizontal_motion) and vertical_drop <= float(maximum_vertical_drop)


def knob_collateral_contact_free(history: Sequence[Any], *, object_key: str = "cookware") -> bool:
    """Reject any arm, hand, or finger contact with the protected cookware."""
    token = str(object_key).lower()
    for snapshot in history:
        for contact in getattr(snapshot, "contacts", ()):
            first, second = (str(item).lower() for item in contact[:2])
            object_contact = token in first or token in second
            robot_contact = any(marker in first or marker in second for marker in ("robot0", "gripper", "finger"))
            if object_contact and robot_contact:
                return False
    return True


def knob_cookware_avoidance_vector(
    *, knob_center: Sequence[float], cookware_center: Sequence[float],
) -> np.ndarray:
    """Return the horizontal unit direction from protected cookware to knob."""
    delta = np.asarray(knob_center, dtype=float)[:2] - np.asarray(cookware_center, dtype=float)[:2]
    norm = float(np.linalg.norm(delta))
    if not np.all(np.isfinite(delta)) or norm <= 1e-9:
        return np.zeros(2, dtype=float)
    return delta / norm


def knob_post_turn_retreat_vector(
    *, knob_center: Sequence[float], cookware_center: Sequence[float],
) -> np.ndarray:
    """Choose a short upward retreat away from cookware after a completed turn."""
    horizontal = knob_cookware_avoidance_vector(
        knob_center=knob_center, cookware_center=cookware_center,
    )
    if float(np.linalg.norm(horizontal)) <= 1e-9:
        horizontal = np.array((-1.0, 0.0), dtype=float)
    return np.array((.16 * horizontal[0], .16 * horizontal[1], .14), dtype=float)


def knob_turn_control_stage(
    *, exact_contact: bool, contact_observed: bool, goal_reached: bool,
) -> str:
    """Rotate only while the selected knob is in live gripper contact."""
    if bool(goal_reached):
        return "stop"
    if bool(exact_contact):
        return "rotate"
    return "reseat" if bool(contact_observed) else "preapproach"


def knob_reseat_target(
    *, knob_center: Sequence[float], measured_contact_offset: Sequence[float],
) -> np.ndarray:
    """Recover the measured surface contact pose instead of the knob center."""
    return np.asarray(knob_center, dtype=float) + np.asarray(measured_contact_offset, dtype=float)


def knob_wrist_target_for_pad_contact(
    *, eef: Sequence[float], pad_center: Sequence[float], knob_center: Sequence[float],
) -> np.ndarray:
    """Convert a desired finger-pad target into the corresponding wrist target."""
    return (
        np.asarray(eef, dtype=float)
        + np.asarray(knob_center, dtype=float)
        - np.asarray(pad_center, dtype=float)
    )


def knob_rotating_surface_target(
    *, knob_center: Sequence[float], initial_contact_offset: Sequence[float],
    initial_rotation: Sequence[Sequence[float]], current_rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Track a physical knob surface point as the knob turns about its centre."""
    initial = np.asarray(initial_rotation, dtype=float).reshape(3, 3)
    current = np.asarray(current_rotation, dtype=float).reshape(3, 3)
    offset = np.asarray(initial_contact_offset, dtype=float)
    return np.asarray(knob_center, dtype=float) + current @ initial.T @ offset


class KnobPrimitive:
    """Read-state closed loop: approach selected knob, contact, then rotate.

    It uses MuJoCo geometry only as privileged *training control* state.  The
    emitted values are normal PandaOmron controller actions; no fixture joint or
    body state is ever assigned here.
    """

    # Contact timing varies with the valid random kitchen layout.  This bound
    # reserves a full rotation segment after contact instead of terminating at
    # the first touch, while remaining a finite diagnostic episode.
    # The unobstructed centre stove control advances by approximately 0.9 mrad
    # per physical controller step under the Panda wrist-speed limit.  Keep one
    # continuous, contact-preserving trajectory long enough to reach the
    # visible-range gate instead of stitching several short partial turns.
    MAX_STEPS = 720

    def __init__(self, contract: Any):
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    def actions(self, environment: Any) -> Iterable[np.ndarray]:
        raw = _raw_environment(environment)
        selection_id = str(getattr(getattr(environment, "request", None), "selection_id", ""))
        try:
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            base_controller = robot.composite_controller.part_controllers["base"]
            eef_site = robot.eef_site_id["right"]
            geoms = tuple(getattr(self.contract, "object_geom_names", {}).get("obj", ()))
            selected = next((name for name in reversed(geoms) if "knob" in name and "main" in name), geoms[-1])
            geom_id = raw.sim.model.geom_name2id(selected)
            joint_id = str(getattr(self.contract, "fixture_joint_ids", {})["turn_knob"])
            initial_joint = float(raw.sim.data.get_joint_qpos(joint_id))
            model_joint_id = int(raw.sim.model.joint_name2id(joint_id))
            required_delta = knob_required_turn_delta(raw.sim.model.jnt_range[model_joint_id])
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            # The generic step-only fallback records a truthful failure if the
            # exact selected fixture binding is absent.
            yield from PrimitiveExpert("turn_knob", 0).actions(environment)
            return
        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        contact_observed = False
        measured_contact_offset: np.ndarray | None = None
        initial_knob_rotation: np.ndarray | None = None
        avoidance_pulses = 0
        for _ in range(self.MAX_STEPS):
            knob_center = np.asarray(raw.sim.data.geom_xpos[geom_id], dtype=float)
            knob_rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            pad_center = selected_gripper_pad_center(raw)
            reference = pad_center if pad_center is not None else eef
            target = (
                knob_wrist_target_for_pad_contact(
                    eef=eef, pad_center=pad_center, knob_center=knob_center,
                )
                if pad_center is not None else knob_center
            )
            error = np.asarray(controller.world_to_origin_frame(target), dtype=float) - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
            action = np.zeros_like(low, dtype=np.float32)
            action[layout.base_mode] = -1.0  # PandaOmron's stationary-base control mode.
            # The exact selected fixture part can have several collision geoms;
            # any geom in that one declared body is admissible, no other knob is.
            exact_contact = _selected_geom_has_gripper_contact(raw, geoms)
            if exact_contact and measured_contact_offset is None:
                measured_contact_offset = reference - knob_center
                initial_knob_rotation = knob_rotation.copy()
            contact_observed = contact_observed or exact_contact
            if contact_observed and measured_contact_offset is not None:
                desired_reference = (
                    knob_rotating_surface_target(
                        knob_center=knob_center,
                        initial_contact_offset=measured_contact_offset,
                        initial_rotation=initial_knob_rotation,
                        current_rotation=knob_rotation,
                    )
                    if initial_knob_rotation is not None
                    else knob_reseat_target(
                        knob_center=knob_center,
                        measured_contact_offset=measured_contact_offset,
                    )
                )
                target = (
                    knob_wrist_target_for_pad_contact(
                        eef=eef, pad_center=pad_center, knob_center=desired_reference,
                    )
                    if pad_center is not None else desired_reference
                )
                error = np.asarray(controller.world_to_origin_frame(target), dtype=float) - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
                distance = float(np.linalg.norm(target - eef))
            distance = float(np.linalg.norm(target - eef))
            current_joint = float(raw.sim.data.get_joint_qpos(joint_id))
            goal_reached = knob_turn_should_stop(
                initial_joint=initial_joint,
                current_joint=current_joint,
                required_delta=required_delta,
                contact_observed=contact_observed,
                exact_contact=exact_contact,
            )
            control_stage = knob_turn_control_stage(
                exact_contact=exact_contact,
                contact_observed=contact_observed,
                goal_reached=goal_reached,
            )
            if control_stage == "stop":
                self.trace.append({
                    "step": len(self.trace) + 1, "stage": "turn_complete",
                    "joint_id": joint_id, "initial_joint": initial_joint,
                    "current_joint": current_joint,
                    "joint_delta": abs(current_joint - initial_joint),
                    "required_joint_delta": required_delta,
                    "exact_body_contact_before_step": exact_contact,
                })
                try:
                    cookware = raw.objects["cookware"]
                    cookware_id = raw.sim.model.body_name2id(cookware.root_body)
                    retreat = knob_post_turn_retreat_vector(
                        knob_center=knob_center,
                        cookware_center=np.asarray(raw.sim.data.body_xpos[cookware_id], dtype=float),
                    )
                except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                    retreat = np.array((-.12, .08, .14), dtype=float)
                retreat_target = eef + retreat
                for retreat_step in range(16):
                    current_eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
                    retreat_error = (
                        np.asarray(controller.world_to_origin_frame(retreat_target), dtype=float)
                        - np.asarray(controller.world_to_origin_frame(current_eef), dtype=float)
                    )
                    retreat_action = np.zeros_like(low, dtype=np.float32)
                    retreat_action[:3] = np.clip(retreat_error / .05, -.7, .7)
                    retreat_action[layout.base_mode] = -1.0
                    if retreat_action.size > layout.gripper:
                        retreat_action[layout.gripper] = -1.0
                    self.trace.append({
                        "step": len(self.trace) + 1, "stage": "post_turn_retreat",
                        "retreat_step": retreat_step + 1,
                        "eef_target_distance": float(np.linalg.norm(retreat_target - current_eef)),
                        "action": np.asarray(retreat_action, dtype=float).round(6).tolist(),
                    })
                    yield np.clip(retreat_action, low, high)
                return
            stage = control_stage
            if control_stage == "rotate":
                # Tangential wrist rotation starts only after the selected
                # knob's exact geom is in a gripper contact pair.
                action[:3] = np.clip(error / 0.05, -0.8, 0.8)
                action[3:6] = (0.0, 0.0, 0.8)
                if action.size > layout.gripper:
                    action[layout.gripper] = 0.8
                # The front-left stove control is physically adjacent to the
                # cookware.  At the beginning of a real turn, make a short
                # low-speed base retreat along the knob-to-cookware clearance
                # direction.  The arm continues to servo to the live contact
                # point; neither cookware state nor collision filters change.
                if selection_id == "VLA82-007" and avoidance_pulses < 12:
                    try:
                        cookware = raw.objects["cookware"]
                        cookware_id = raw.sim.model.body_name2id(cookware.root_body)
                        cookware_center = np.asarray(raw.sim.data.body_xpos[cookware_id], dtype=float)
                        direction = knob_cookware_avoidance_vector(
                            knob_center=knob_center, cookware_center=cookware_center,
                        )
                        _, base_rotation = base_controller.get_base_pose()
                        local_step = np.asarray(base_rotation, dtype=float)[:2, :2].T @ (direction * .006)
                        action[layout.base[0]:layout.base[0] + 2] = np.clip(local_step / .08, -.25, .25)
                        action[layout.base_mode] = 1.0
                        avoidance_pulses += 1
                    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                        pass
            else:
                action[:3] = np.clip(error / 0.05, -0.8, 0.8)
                if action.size > layout.gripper:
                    # Closing while still translating the final centimetres
                    # creates the physical gripper--knob contact; rotation is
                    # nevertheless withheld until that exact contact is seen.
                    action[layout.gripper] = 0.8 if distance < 0.09 else -1.0
                if distance < 0.09 and not contact_observed:
                    # A short seating-orientation action keeps translating
                    # along the target normal while closing. The following
                    # simulator snapshot must show exact body contact before
                    # the persistent turn segment below is permitted.
                    action[3:6] = (0.0, 0.0, 0.8)
                    stage = "seat_contact"
            self.trace.append({
                "step": len(self.trace) + 1, "stage": stage,
                "eef_target_distance": distance, "exact_body_contact_before_step": exact_contact,
                "action": np.asarray(action, dtype=float).round(6).tolist(),
            })
            yield np.clip(action, low, high)


def drawer_close_stage(*, exact_handle_contact: bool, joint_position: float, closed_threshold: float = .08) -> str:
    """Select a physical drawer-close state without mutating the slide joint."""
    if abs(float(joint_position)) <= abs(float(closed_threshold)):
        return "settle_closed"
    return "push_close" if bool(exact_handle_contact) else "approach_handle"


def fixture_close_active_stage(
    *, exact_handle_contact: bool, contact_seen: bool,
    joint_position: float, selection_id: str = "", completion_allowed: bool = True,
) -> str:
    """Choose a contact-gated fixture-close state from live joint evidence."""
    if not bool(completion_allowed) and abs(float(joint_position)) <= .08:
        return "push_close" if bool(exact_handle_contact) else "approach_handle"
    # The compact toaster-oven door can elastically separate from the Panda
    # finger after the first valid panel contact.  Retain its physical closing
    # direction through that short separation; MuJoCo collision remains the
    # sole mechanism that can move the door and the joint still must reach
    # the normal closed threshold.
    if (
        str(selection_id) == "VLA82-034"
        and bool(contact_seen)
        and abs(float(joint_position)) > .08
    ):
        return "push_close"
    return drawer_close_stage(
        exact_handle_contact=exact_handle_contact,
        joint_position=joint_position,
    )


def fixture_close_push_distance(*, joint_type: int) -> float:
    """Use a contact-preserving tangent step for hinges and a long slide push."""
    return .01 if int(joint_type) == 3 else .24


def fixture_close_base_follow_local(
    *, joint_type: int, exact_contact: bool, joint_position: float,
    push_direction_world: Sequence[float], base_rotation: Sequence[Sequence[float]],
) -> np.ndarray | None:
    """Follow a long slide only after the arm reaches its useful extension."""
    if int(joint_type) != 2 or not bool(exact_contact) or float(joint_position) > .25:
        return None
    direction = np.asarray(push_direction_world, dtype=float)[:2]
    rotation = np.asarray(base_rotation, dtype=float)[:2, :2]
    local = rotation.T @ direction
    local /= max(float(np.linalg.norm(local)), 1e-9)
    return local * .18


def fixture_approach_base_local(
    *, pad_handle_distance: float, handle_delta_world: Sequence[float],
    base_rotation: Sequence[Sequence[float]], threshold: float = .14,
) -> np.ndarray | None:
    """Bring a mobile manipulator into arm range before handle contact."""
    if float(pad_handle_distance) <= float(threshold):
        return None
    planar = np.asarray(handle_delta_world, dtype=float)[:2]
    norm = float(np.linalg.norm(planar))
    if norm <= 1e-9:
        return None
    rotation = np.asarray(base_rotation, dtype=float)[:2, :2]
    local = rotation.T @ (planar / norm)
    return local * .16


def fixture_handle_stage_base_follow(
    *, stage: str, bypass_required: bool, target_distance: float,
    target_delta_world: Sequence[float], base_rotation: Sequence[Sequence[float]],
) -> np.ndarray | None:
    """Use the mobile base while an audited handle corridor is beyond arm reach."""
    if bool(bypass_required) or str(stage) not in {
        "safe_approach_above_handle", "approach_handle",
    }:
        return None
    return fixture_approach_base_local(
        pad_handle_distance=target_distance,
        handle_delta_world=target_delta_world,
        base_rotation=base_rotation,
    )


def fixture_close_target(
    *, stage: str, handle_world: Sequence[float], eef_world: Sequence[float],
    push_direction: Sequence[float], push_distance: float,
    contact_offset: Sequence[float] | None = None, joint_type: int,
) -> np.ndarray:
    """Preserve the live wrist-to-contact offset while pushing a fixture.

    The Panda EEF site is several centimetres behind its finger contact point.
    Once contact exists, targeting ``handle + tangent`` incorrectly tries to
    collapse that physical offset and can cancel the useful tangent command.
    """
    if int(joint_type) == 2:
        distance = float(push_distance) if str(stage) == "push_close" else 0.0
        return (
            np.asarray(handle_world, dtype=float)
            + np.asarray(push_direction, dtype=float) * distance
        )
    if str(stage) != "push_close":
        offset = (
            np.zeros(3, dtype=float) if contact_offset is None
            # Closing the fingers shifts the collision pads slightly.  A
            # bounded 10% reseating bias restores contact without tunnelling
            # through the fixture surface.
            else .9 * np.asarray(contact_offset, dtype=float)
        )
        return np.asarray(handle_world, dtype=float) + offset
    return (
        np.asarray(eef_world, dtype=float)
        + np.asarray(push_direction, dtype=float) * float(push_distance)
    )


def fixture_close_push_direction(
    *, joint_type: int, joint_position: float,
    joint_axis_world: Sequence[float], joint_anchor_world: Sequence[float],
    handle_world: Sequence[float],
) -> np.ndarray:
    """Return the world-space handle motion that drives a joint toward zero."""
    axis = np.asarray(joint_axis_world, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        raise ValueError("fixture close joint axis is degenerate")
    axis = axis / axis_norm
    sign_to_zero = -1.0 if float(joint_position) >= 0.0 else 1.0
    if int(joint_type) == 2:  # MuJoCo slide joint
        direction = sign_to_zero * axis
    elif int(joint_type) == 3:  # MuJoCo hinge joint
        lever = np.asarray(handle_world, dtype=float) - np.asarray(joint_anchor_world, dtype=float)
        # The physical point velocity for increasing q is axis x lever.
        # Multiplying by the sign toward zero closes either joint convention.
        direction = sign_to_zero * np.cross(axis, lever)
    else:
        raise ValueError(f"unsupported fixture close joint type: {joint_type}")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise ValueError("fixture close handle has no usable motion direction")
    return direction / norm


def fixture_open_pull_direction(
    *, joint_type: int, joint_position: float,
    joint_axis_world: Sequence[float], joint_anchor_world: Sequence[float],
    handle_world: Sequence[float],
) -> np.ndarray:
    """Return the tangent that increases the physical hinge/slide coordinate."""
    return -fixture_close_push_direction(
        joint_type=joint_type,
        # The closing helper is sign-dependent.  A non-negative virtual
        # position makes it return the negative-q tangent, whose inverse is
        # the positive-q opening direction regardless of the live start.
        joint_position=abs(float(joint_position)),
        joint_axis_world=joint_axis_world,
        joint_anchor_world=joint_anchor_world,
        handle_world=handle_world,
    )


def fixture_open_presentation_complete(*, contact_seen: bool, joint_position: float) -> bool:
    """Keep opening until the door has a clearly visible full-open angle."""
    return bool(contact_seen) and float(joint_position) >= .80


def fixture_open_contact_hold_target(
    *, handle_world: Sequence[float], contact_offset: Sequence[float],
    pull_direction: Sequence[float], pull_distance: float = .04,
) -> np.ndarray:
    """Track a moving handle while retaining the measured physical contact pose."""
    direction = np.asarray(pull_direction, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return (
        np.asarray(handle_world, dtype=float)
        + np.asarray(contact_offset, dtype=float)
        + direction * float(pull_distance)
    )


def fixture_close_requires_free_edge_bypass(
    joint_id: str, *, initial_joint_position: float | None = None,
) -> bool:
    """Route around a microwave panel only when it starts widely open."""
    if "microjoint" not in str(joint_id).lower():
        return False
    # A normal partially-open door presents its handle directly to the arm;
    # the wide-open free-edge route is unnecessary and otherwise consumes the
    # whole close-action budget before any physical push occurs.
    return initial_joint_position is None or abs(float(initial_joint_position)) > .70


def fixture_hinge_free_edge_waypoints(
    *, panel_center: Sequence[float], panel_rotation: Sequence[float],
    panel_half_size: Sequence[float], hinge_anchor: Sequence[float],
    eef_world: Sequence[float], push_direction: Sequence[float],
    edge_clearance: float = .09, side_clearance: float = .06,
) -> dict[str, np.ndarray]:
    """Return collision-clearing waypoints around a hinged panel's free edge."""
    center = np.asarray(panel_center, dtype=float)
    rotation = np.asarray(panel_rotation, dtype=float).reshape(3, 3)
    size = np.asarray(panel_half_size, dtype=float)
    hinge = np.asarray(hinge_anchor, dtype=float)
    eef = np.asarray(eef_world, dtype=float)
    push = np.asarray(push_direction, dtype=float)
    push_norm = float(np.linalg.norm(push))
    if push_norm <= 1e-9 or np.any(size <= 1e-9):
        raise ValueError("microwave bypass geometry is degenerate")
    push /= push_norm

    normal_index = int(np.argmin(size))
    in_plane = tuple(index for index in range(3) if index != normal_index)
    edge_index = max(
        in_plane,
        key=lambda index: float(size[index] * np.linalg.norm(rotation[:2, index])),
    )
    edge_axis = rotation[:, edge_index]
    candidates = (
        center + edge_axis * size[edge_index],
        center - edge_axis * size[edge_index],
    )
    free_edge = max(
        candidates,
        key=lambda point: float(np.linalg.norm(point - hinge)),
    )
    outward_edge = free_edge - center
    outward_edge /= max(float(np.linalg.norm(outward_edge)), 1e-9)
    outside = free_edge + outward_edge * float(edge_clearance)
    current_sign = 1.0 if float(np.dot(eef - center, push)) >= 0.0 else -1.0
    return {
        "free_edge": free_edge.copy(),
        "outside_current_side": outside + current_sign * push * float(side_clearance),
        "outside_closing_side": outside - push * float(side_clearance),
        "contact_prepoint": center - push * float(side_clearance),
    }


def fixture_close_bypass_stage(
    *, joint_closed: bool, free_edge_distance: float,
    cross_plane_distance: float, precontact_distance: float,
    exact_contact: bool, contact_seen_after_cross: bool = False,
    tolerance: float = .025,
) -> str:
    """Advance a hinged-door bypass only after each waypoint is reached."""
    if bool(joint_closed):
        return "settle_closed"
    if float(free_edge_distance) > float(tolerance):
        return "bypass_to_free_edge"
    if float(cross_plane_distance) > float(tolerance):
        return "bypass_cross_plane"
    if bool(exact_contact):
        return "push_close"
    if bool(contact_seen_after_cross):
        return "seat_contact"
    if float(precontact_distance) > float(tolerance):
        return "bypass_return_to_panel"
    return "seat_contact"


def fixture_close_filtered_contact_normal(
    previous: Sequence[float] | None, current: Sequence[float],
) -> np.ndarray:
    """Hemisphere-align and smooth a live gripper-to-fixture contact normal."""
    current_vector = np.asarray(current, dtype=float)
    current_norm = float(np.linalg.norm(current_vector))
    if current_norm <= 1e-9:
        raise ValueError("fixture close contact normal is degenerate")
    current_vector /= current_norm
    if previous is None:
        return current_vector
    previous_vector = np.asarray(previous, dtype=float)
    previous_norm = float(np.linalg.norm(previous_vector))
    if previous_norm <= 1e-9:
        raise ValueError("fixture close previous normal is degenerate")
    previous_vector /= previous_norm
    if float(np.dot(previous_vector, current_vector)) < 0.0:
        current_vector = -current_vector
    filtered = .65 * previous_vector + .35 * current_vector
    filtered_norm = float(np.linalg.norm(filtered))
    if filtered_norm <= 1e-9:
        raise ValueError("fixture close filtered normal is degenerate")
    return filtered / filtered_norm


def fixture_close_contact_preserving_direction(
    *, tangent: Sequence[float], contact_normal: Sequence[float],
    minimum_inward: float = .08, max_normal_bias: float = .45,
) -> tuple[np.ndarray, float]:
    """Blend a closing tangent with only the normal bias needed for contact."""
    tangent_vector = np.asarray(tangent, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent_vector))
    if tangent_norm <= 1e-9:
        raise ValueError("fixture close tangent is degenerate")
    tangent_vector /= tangent_norm
    normal_vector = np.asarray(contact_normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal_vector))
    if normal_norm <= 1e-9:
        raise ValueError("fixture close contact normal is degenerate")
    normal_vector /= normal_norm
    inward = float(np.dot(tangent_vector, normal_vector))
    bias = float(np.clip(
        float(minimum_inward) - inward,
        0.0,
        float(max_normal_bias),
    ))
    direction = tangent_vector + bias * normal_vector
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return direction, bias


def fixture_close_deepest_contact_normal(
    details: Sequence[Mapping[str, Any]],
) -> np.ndarray | None:
    """Return the normalized gripper-to-object normal of the deepest contact."""
    if not details:
        return None
    deepest = min(details, key=lambda detail: float(detail["distance"]))
    normal = np.asarray(deepest["normal_gripper_to_object"], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("fixture close deepest contact normal is degenerate")
    return normal / norm


def fixture_close_contact_reseat_target(
    *, contact_reference: Sequence[float], last_contact_normal: Sequence[float],
    distance: float = .006,
) -> np.ndarray:
    """Move a lost contact only a short step back toward the fixture."""
    normal = np.asarray(last_contact_normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("fixture close reseat normal is degenerate")
    return (
        np.asarray(contact_reference, dtype=float)
        + normal / norm * float(distance)
    )


def fixture_close_panel_geom_name(
    *, candidate_names: Sequence[str], geom_types: Mapping[str, int],
    geom_sizes: Mapping[str, Sequence[float]],
) -> str:
    """Select the thinnest broad non-handle box bound to a fixture joint."""
    candidates: list[tuple[float, float, str]] = []
    for raw_name in candidate_names:
        name = str(raw_name)
        if "handle" in name.lower() or int(geom_types[name]) != 6:
            continue
        dimensions = sorted(float(value) for value in geom_sizes[name])
        if len(dimensions) != 3 or dimensions[0] <= 1e-9:
            continue
        candidates.append((dimensions[0], -(dimensions[1] * dimensions[2]), name))
    if not candidates:
        raise ValueError("fixture close has no broad panel box geometry")
    return min(candidates)[2]


def fixture_close_upper_panel_waypoints(
    *, panel_center: Sequence[float], panel_rotation: Sequence[float],
    panel_half_size: Sequence[float], hinge_anchor: Sequence[float],
    handle_world: Sequence[float],
    push_direction: Sequence[float], outside_closing_side: Sequence[float],
    vertical_fraction: float = .45, precontact_clearance: float = .06,
    min_handle_clearance: float = .07, top_clearance: float = .08,
    horizontal_free_edge_fraction: float = .65,
    min_edge_clearance: float = .075, max_path_distance: float = .20,
) -> dict[str, np.ndarray]:
    """Return a collision-clearing route to the upper broad door panel."""
    center = np.asarray(panel_center, dtype=float)
    rotation = np.asarray(panel_rotation, dtype=float).reshape(3, 3)
    size = np.asarray(panel_half_size, dtype=float)
    handle = np.asarray(handle_world, dtype=float)
    hinge = np.asarray(hinge_anchor, dtype=float)
    push = np.asarray(push_direction, dtype=float)
    push_norm = float(np.linalg.norm(push))
    if push_norm <= 1e-9 or np.any(size <= 1e-9):
        raise ValueError("fixture close upper panel geometry is degenerate")
    push /= push_norm
    normal_index = int(np.argmin(size))
    in_plane = tuple(index for index in range(3) if index != normal_index)
    vertical_index = max(in_plane, key=lambda index: abs(float(rotation[2, index])))
    horizontal_index = next(index for index in in_plane if index != vertical_index)
    vertical_axis = np.asarray(rotation[:, vertical_index], dtype=float)
    if vertical_axis[2] < 0.0:
        vertical_axis = -vertical_axis
    half_height = float(size[vertical_index])
    half_width = float(size[horizontal_index])
    horizontal_axis = np.asarray(rotation[:, horizontal_index], dtype=float)
    edge_candidates = (
        center + horizontal_axis * half_width,
        center - horizontal_axis * half_width,
    )
    free_edge = max(
        edge_candidates,
        key=lambda point: float(np.linalg.norm(point - hinge)),
    )
    free_edge_direction = free_edge - center
    free_edge_direction /= max(float(np.linalg.norm(free_edge_direction)), 1e-9)
    horizontal_offset = (
        free_edge_direction * float(horizontal_free_edge_fraction) * half_width
    )
    edge_clearance = (1.0 - float(horizontal_free_edge_fraction)) * half_width
    if edge_clearance < float(min_edge_clearance):
        raise ValueError("fixture close upper panel edge clearance is unsafe")
    vertical_offset = float(vertical_fraction) * half_height
    projected_extent = float(np.sum(np.abs(rotation.T @ push) * size))
    face = (
        center + horizontal_offset + vertical_axis * vertical_offset
        - push * projected_extent
    )
    handle_clearance = float(np.dot(face - handle, vertical_axis))
    remaining_top_clearance = half_height - vertical_offset
    if handle_clearance < float(min_handle_clearance):
        raise ValueError("fixture close upper panel handle clearance is unsafe")
    if remaining_top_clearance < float(top_clearance):
        raise ValueError("fixture close upper panel top clearance is unsafe")
    precontact = face - push * float(precontact_clearance)
    outside = np.asarray(outside_closing_side, dtype=float)
    raised = outside + vertical_axis * float(np.dot(precontact - outside, vertical_axis))
    if float(np.linalg.norm(precontact - raised)) > float(max_path_distance):
        raise ValueError("fixture close upper panel path distance is unreachable")
    return {
        "raise": raised,
        "precontact": precontact,
        "face": face,
        "vertical_axis": vertical_axis,
        "free_edge_direction": free_edge_direction,
        "horizontal_offset": horizontal_offset,
        "edge_clearance": float(edge_clearance),
    }


def fixture_close_upper_panel_stage(
    *, joint_closed: bool, raised: bool, traversed: bool,
    exact_panel_contact: bool, panel_contact_seen: bool,
) -> str:
    """Gate microwave closing on an ordered route and broad-panel contact."""
    if bool(joint_closed):
        return "settle_closed"
    if not bool(raised):
        return "raise_above_handle"
    if not bool(traversed):
        return "traverse_to_upper_panel"
    if bool(exact_panel_contact):
        return "push_close"
    return "reseat_upper_panel" if bool(panel_contact_seen) else "seat_upper_panel"


def fixture_raise_base_assist_local(
    *, stage: str, stage_frames: int, distance_history: Sequence[float],
    target_delta_world: Sequence[float], base_rotation: Sequence[float],
    cumulative_displacement: float, collision: bool, locked: bool,
    min_stage_frames: int = 60, history_size: int = 40,
    plateau_improvement: float = .003, target_tolerance: float = .025,
    max_displacement: float = .05, command_magnitude: float = .08,
) -> np.ndarray | None:
    """Return a bounded local base command only for a stalled safe raise."""
    history = tuple(float(value) for value in distance_history)
    if str(stage) != "raise_above_handle" or int(stage_frames) < int(min_stage_frames):
        return None
    if len(history) < int(history_size) or history[-1] <= float(target_tolerance):
        return None
    improvement = history[-int(history_size)] - history[-1]
    if improvement >= float(plateau_improvement):
        return None
    if bool(collision) or bool(locked):
        return None
    if float(cumulative_displacement) >= float(max_displacement):
        return None
    planar = np.asarray(target_delta_world, dtype=float)[:2]
    planar_norm = float(np.linalg.norm(planar))
    if planar_norm <= 1e-9:
        return None
    rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3)
    local = rotation[:2, :2].T @ (planar / planar_norm)
    local_norm = float(np.linalg.norm(local))
    if local_norm <= 1e-9:
        return None
    return local / local_norm * float(command_magnitude)


def fixture_close_contact_geometries(contract: Any, joint_id: str) -> tuple[str, ...]:
    """Return every selected movable-body geom bound to the close joint."""
    return tuple(
        str(name) for name, joint in getattr(contract, "fixture_contact_joints", {}).items()
        if str(joint) == str(joint_id)
    )


HANDLE_HELD_DOOR_CLOSE_SELECTIONS = frozenset({
    "VLA82-008", "VLA82-028", "VLA82-029", "VLA82-031", "VLA82-033", "VLA82-034",
})


def fixture_close_requires_two_pad_handle_grasp(selection_id: str) -> bool:
    """Require a physical two-pad hold for every articulated door-close task.

    A door-panel push can close a hinge but does not meet the documented
    operation: the gripper must first take the named door handle.  Drawers
    remain outside this policy because their specified motion is a slide, not
    a door-close action.
    """
    return str(selection_id) in HANDLE_HELD_DOOR_CLOSE_SELECTIONS


def fixture_close_handle_grasp_observed(selection_id: str, controller_trace: Sequence[Mapping[str, Any]]) -> bool:
    """Require a two-pad handle hold during the physical closing segment when audited."""
    if not fixture_close_requires_two_pad_handle_grasp(selection_id):
        return True
    return any(
        str(frame.get("stage", "")) == "push_close"
        and (
            bool(frame.get("exact_two_pad_handle_contact", False))
            or bool(frame.get("exact_two_finger_handle_contact", False))
        )
        for frame in controller_trace
    )


def fixture_close_gripper_command(
    *, handle_grasp_required: bool, bypass_required: bool,
    contact_seen: bool, pad_handle_distance: float, pregrasp_reached: bool = True,
    orientation_ready: bool = True,
) -> float:
    """Close early enough to envelop an audited handle before the hand body reaches it."""
    if bool(handle_grasp_required):
        return .8 if (
            bool(orientation_ready)
            and bool(pregrasp_reached)
            and float(pad_handle_distance) <= .24
        ) else -1.0
    return .8 if bool(bypass_required) or bool(contact_seen) else -1.0


def fixture_close_exact_contact_geometries(
    *, labelled_handle_geoms: Sequence[str], joint_bound_geoms: Sequence[str],
    allow_panel_contact: bool = False,
) -> tuple[str, ...]:
    """Select valid physical contacts for closing an articulated fixture.

    Opening requires a handle pull, but closing may legitimately push any
    collision geometry rigidly attached to the selected moving door or drawer.
    The explicit flag keeps the stricter handle-only behavior available to
    callers that need it.
    """
    handles = tuple(str(name) for name in labelled_handle_geoms)
    if bool(allow_panel_contact):
        return tuple(str(name) for name in joint_bound_geoms)
    return handles or tuple(str(name) for name in joint_bound_geoms)


def fixture_close_wrist_target_for_pad_contact(
    *, eef: Sequence[float], pad_center: Sequence[float],
    desired_pad_center: Sequence[float],
) -> np.ndarray:
    """Convert a desired handle-contact pad point into the controller wrist target."""
    return (
        np.asarray(eef, dtype=float)
        + np.asarray(desired_pad_center, dtype=float)
        - np.asarray(pad_center, dtype=float)
    )


def fixture_handle_roll_command(
    *, opening_axis_world: Sequence[float], handle_axis_world: Sequence[float],
    tool_axis_world: Sequence[float], base_rotation: Sequence[float],
    tolerance: float = .18, max_command: float = .6,
) -> tuple[np.ndarray, float]:
    """Roll the hand until its finger-opening axis crosses the handle bar.

    OSC rotation commands are expressed in the robot base frame.  Selecting
    the nearer of the two equivalent perpendicular directions prevents an
    unnecessary half turn, while the signed angle preserves closed-loop
    convergence from either side of the handle.
    """
    tool = np.asarray(tool_axis_world, dtype=float)
    tool_norm = float(np.linalg.norm(tool))
    if tool_norm <= 1e-9:
        return np.zeros(3, dtype=float), 0.0
    tool /= tool_norm
    opening = np.asarray(opening_axis_world, dtype=float)
    opening -= tool * float(np.dot(opening, tool))
    opening_norm = float(np.linalg.norm(opening))
    handle = np.asarray(handle_axis_world, dtype=float)
    desired = np.cross(tool, handle)
    desired_norm = float(np.linalg.norm(desired))
    if opening_norm <= 1e-9 or desired_norm <= 1e-9:
        return np.zeros(3, dtype=float), 0.0
    opening /= opening_norm
    desired /= desired_norm
    if float(np.dot(opening, desired)) < 0.0:
        desired = -desired
    signed_angle = float(np.arctan2(
        np.dot(tool, np.cross(opening, desired)),
        np.clip(np.dot(opening, desired), -1.0, 1.0),
    ))
    error = abs(signed_angle)
    if error <= float(tolerance):
        return np.zeros(3, dtype=float), error
    rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3)
    local_tool = rotation.T @ tool
    magnitude = min(error / .5, float(max_command))
    return local_tool * np.sign(signed_angle) * magnitude, error


def fixture_handle_tool_axis_from_eef_rotation(
    eef_rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return the Panda hand's stable outward approach axis in world frame."""
    rotation = np.asarray(eef_rotation, dtype=float).reshape(3, 3)
    axis = -rotation[:, 2]
    return axis / max(float(np.linalg.norm(axis)), 1e-9)


def fixture_handle_tool_alignment_command(
    *, tool_axis_world: Sequence[float], outward_normal_world: Sequence[float],
    base_rotation: Sequence[float], tolerance: float = .18,
    max_command: float = .6,
) -> tuple[np.ndarray, float]:
    """Pitch/yaw the finger travel axis toward the fixture handle face."""
    tool = np.asarray(tool_axis_world, dtype=float)
    outward = np.asarray(outward_normal_world, dtype=float)
    tool_norm = float(np.linalg.norm(tool))
    outward_norm = float(np.linalg.norm(outward))
    if tool_norm <= 1e-9 or outward_norm <= 1e-9:
        return np.zeros(3, dtype=float), 0.0
    tool /= tool_norm
    desired = -outward / outward_norm
    cross = np.cross(tool, desired)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(tool, desired), -1.0, 1.0))
    error = float(np.arctan2(sine, cosine))
    if error <= float(tolerance):
        return np.zeros(3, dtype=float), error
    if sine <= 1e-9:
        # A deterministic perpendicular is needed only for the 180-degree case.
        trial = np.array((1.0, 0.0, 0.0), dtype=float)
        if abs(float(np.dot(tool, trial))) > .9:
            trial = np.array((0.0, 1.0, 0.0), dtype=float)
        axis_world = np.cross(tool, trial)
        axis_world /= max(float(np.linalg.norm(axis_world)), 1e-9)
    else:
        axis_world = cross / sine
    rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3)
    axis_local = rotation.T @ axis_world
    magnitude = min(error / .5, float(max_command))
    return axis_local * magnitude, error


def fixture_handle_outward_normal(
    *, joint_axis_world: Sequence[float], handle_world: Sequence[float],
    joint_anchor_world: Sequence[float], eef_world: Sequence[float],
) -> np.ndarray:
    """Recover a hinged fixture face normal from its live joint geometry."""
    axis = np.asarray(joint_axis_world, dtype=float)
    lever = np.asarray(handle_world, dtype=float) - np.asarray(joint_anchor_world, dtype=float)
    normal = np.cross(axis, lever)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("fixture handle and hinge do not define a face normal")
    normal /= norm
    if float(np.dot(normal, np.asarray(eef_world, dtype=float) - np.asarray(handle_world, dtype=float))) < 0.0:
        normal = -normal
    return normal


def fixture_handle_grasp_surface_normal(
    *, handle_rotation: Sequence[float], handle_half_size: Sequence[float],
    handle_world: Sequence[float], eef_world: Sequence[float],
) -> np.ndarray:
    """Choose the handle cross-section direction that faces the approaching wrist."""
    rotation = np.asarray(handle_rotation, dtype=float).reshape(3, 3)
    sizes = np.asarray(handle_half_size, dtype=float)
    long_axis_index = int(np.argmax(sizes))
    candidates = tuple(
        rotation[:, index] for index in range(3) if index != long_axis_index
    )
    # Refrigerator/cabinet handles are mounted on a vertical door face.
    # Prefer the horizontal cross-section direction so an elevated wrist does
    # not mistake the bar's vertical clearance axis for the approach normal.
    normal = min(candidates, key=lambda axis: abs(float(axis[2]))).copy()
    if float(np.dot(normal, np.asarray(eef_world, dtype=float) - np.asarray(handle_world, dtype=float))) < 0.0:
        normal = -normal
    return normal


def fixture_handle_pregrasp_point(
    *, handle_world: Sequence[float], outward_normal: Sequence[float], clearance: float = .10,
) -> np.ndarray:
    """Place the pad centre on the clear exterior side before entering a handle grasp."""
    normal = np.asarray(outward_normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    return np.asarray(handle_world, dtype=float) + normal * float(clearance)


def fixture_handle_safe_approach_point(
    *, pregrasp_target: Sequence[float], eef_world: Sequence[float], clearance: float = .14,
) -> np.ndarray:
    """Keep the open hand above a door before it enters the grasp corridor."""
    target = np.asarray(pregrasp_target, dtype=float).copy()
    target[2] = max(float(target[2] + clearance), float(np.asarray(eef_world, dtype=float)[2]))
    return target


def fixture_handle_safe_lift_point(
    *, eef_world: Sequence[float], pregrasp_target: Sequence[float],
    clearance: float = .18,
) -> np.ndarray:
    """Raise straight up before the open hand crosses a door plane."""
    eef = np.asarray(eef_world, dtype=float)
    target = eef.copy()
    target[2] = max(float(eef[2] + clearance), float(np.asarray(pregrasp_target, dtype=float)[2] + .24))
    return target


def fixture_handle_safe_lift_target_for_selection(
    selection_id: str, *, eef_world: Sequence[float],
    pregrasp_target: Sequence[float], outward_normal: Sequence[float],
) -> np.ndarray:
    """Retreat VLA82-031 clear of the open door sweep before raising the wrist."""
    target = fixture_handle_safe_lift_point(
        eef_world=eef_world, pregrasp_target=pregrasp_target,
    )
    if str(selection_id) == "VLA82-031":
        normal = np.asarray(outward_normal, dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        target[:2] = np.asarray(pregrasp_target, dtype=float)[:2] + normal[:2] * .05
    return target


def fixture_handle_pregrasp_reached(
    distance: float, *, tolerance: float = .06,
) -> bool:
    """Accept the exterior grasp corridor once an open Panda hand fits it."""
    return float(distance) <= float(tolerance)


def fixture_handle_waypoint_state(
    *, previously_reached: bool, distance: float, tolerance: float,
) -> bool:
    """Latch a one-way collision-clearance waypoint for the rest of an episode."""
    return bool(previously_reached) or float(distance) <= float(tolerance)


def fixture_handle_pregrasp_state(
    *, previously_reached: bool, distance: float,
) -> bool:
    """Apply an entry/exit hysteresis to a physical handle grasp corridor."""
    tolerance = .11 if bool(previously_reached) else .07
    return float(distance) <= float(tolerance)


def fixture_handle_pregrasp_rotation_command(
    stage: str, rotation_command: Sequence[float], *, selection_id: str = "",
) -> np.ndarray:
    """Keep the wrist straight during the collision-free handle approach."""
    if str(selection_id) == "VLA82-031" and str(stage) == "safe_approach_above_handle":
        return np.asarray(rotation_command, dtype=float)
    if str(stage) in {
        "safe_lift_above_door", "safe_approach_above_handle", "pregrasp_handle",
    }:
        return np.zeros(3, dtype=float)
    return np.asarray(rotation_command, dtype=float)


def fixture_close_handle_grasp_direction(
    selection_id: str, nominal_direction: Sequence[float],
) -> np.ndarray:
    """Apply the audited hinge sign for a handle-held refrigerator close."""
    direction = np.asarray(nominal_direction, dtype=float)
    # VLA82-028's authored left-door joint increases toward the exterior while
    # the generic convention assumes the opposite sign.  The correction is
    # exercised only after the controller has a real two-finger handle grasp.
    return -direction if str(selection_id) == "VLA82-028" else direction


def fixture_close_selection_push_direction(
    selection_id: str, direction: Sequence[float],
) -> np.ndarray:
    """Apply a measured hinge-sign correction for the microwave close task."""
    result = np.asarray(direction, dtype=float)
    # VLA82-029's microwave hinge coordinate is opposite to the generic
    # appliance convention. The previous tangential command visibly opened
    # the partially-open door while in true fixture contact.
    return -result if str(selection_id) == "VLA82-029" else result


def fixture_close_latched_handle_contact(
    *, grasp_latched: bool, exact_handle_contact: bool,
) -> bool:
    """Keep a verified grasp moving only while the hand remains on the handle."""
    return bool(grasp_latched) and bool(exact_handle_contact)


def fixture_close_step_budget(selection_id: str, *, default_steps: int) -> int:
    """Reserve enough physical steps for the audited refrigerator's full hinge sweep."""
    return max(int(default_steps), 1600) if str(selection_id) == "VLA82-028" else int(default_steps)


def fixture_close_effective_push_distance(selection_id: str, *, joint_type: int) -> float:
    """Use a continuous but bounded tangential handle pull for the audited fridge."""
    if str(selection_id) == "VLA82-028" and int(joint_type) == 3:
        return .04
    return fixture_close_push_distance(joint_type=joint_type)


def fixture_close_contact_target_name(
    *, labelled_geoms: Sequence[str], joint_bound_geoms: Sequence[str],
    geom_positions: Mapping[str, Sequence[float]], eef_world: Sequence[float],
    contactable_geoms: Sequence[str] = (), prefer_nearest_joint_bound: bool = False,
    geom_half_sizes: Mapping[str, Sequence[float]] | None = None,
    prefer_grasp_bar: bool = False,
) -> str:
    """Choose a real joint-bound contact geom even when an asset lacks a handle label."""
    candidates = (
        tuple(joint_bound_geoms)
        if bool(prefer_nearest_joint_bound)
        else tuple(labelled_geoms) or tuple(joint_bound_geoms)
    )
    if not candidates:
        raise ValueError("fixture close has no joint-bound contact geometry")
    contactable = set(str(name) for name in contactable_geoms)
    physical_candidates = tuple(name for name in candidates if str(name) in contactable)
    if physical_candidates:
        candidates = physical_candidates
    if not bool(prefer_nearest_joint_bound):
        central_bars = tuple(
            name for name in candidates
            if str(name).lower().endswith("handle_main")
        )
        if central_bars:
            candidates = central_bars
    if bool(prefer_grasp_bar) and geom_half_sizes:
        # Assets often model a usable bar as one long collision geom plus
        # many tiny decorative/end-cap geoms.  The bar is the only candidate
        # that can receive both parallel-jaw pads reliably.
        return str(max(
            candidates,
            key=lambda name: float(np.max(np.asarray(
                geom_half_sizes.get(str(name), (0., 0., 0.)), dtype=float,
            ))),
        ))
    eef = np.asarray(eef_world, dtype=float)
    return str(min(
        candidates,
        key=lambda name: float(np.linalg.norm(np.asarray(geom_positions[str(name)], dtype=float) - eef)),
    ))


def rack_front_contact_point(
    *, center: Sequence[float], rotation: Sequence[float],
    half_size: Sequence[float], push_direction: Sequence[float],
) -> np.ndarray:
    """Return the OBB face centre that the hand must push from.

    The point lies on the face opposite the desired rack motion.  Using the
    live box orientation makes this work in every sampled kitchen yaw without
    guessing that the appliance always faces a global axis.
    """
    origin = np.asarray(center, dtype=float)
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    size = np.asarray(half_size, dtype=float)
    direction = np.asarray(push_direction, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    projected_extent = float(np.sum(np.abs(matrix.T @ direction) * size))
    return origin - direction * projected_extent


def rack_wrist_target_for_pad_contact(
    *, eef: Sequence[float], pad_center: Sequence[float],
    desired_pad_center: Sequence[float],
) -> np.ndarray:
    """Convert a desired rack-contact pad point to the controller wrist target."""
    return (
        np.asarray(eef, dtype=float)
        + np.asarray(desired_pad_center, dtype=float)
        - np.asarray(pad_center, dtype=float)
    )


def rack_seat_contact_point(
    *, face: Sequence[float], push_direction: Sequence[float],
) -> np.ndarray:
    """Return a pad point just outside the rack face before an inward push.

    The rack face is the collision boundary.  Seating on its interior side
    asks the controller to cross that boundary before any contact has been
    recorded, causing a preapproach/seat oscillation instead of a push.
    """
    direction = np.asarray(push_direction, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return np.asarray(face, dtype=float) - direction * .012


def rack_contact_stage(
    *, joint_closed: bool, exact_contact: bool,
    pad_precontact_distance: float,
    seat_latched: bool = False,
    push_latched: bool = False,
) -> str:
    """Advance by the physical pad location, not the remote wrist site."""
    if bool(joint_closed):
        return "settle_closed"
    if bool(exact_contact):
        return "push_close"
    if bool(push_latched):
        return "push_close"
    if bool(seat_latched):
        return "seat_contact"
    return "preapproach" if float(pad_precontact_distance) > .018 else "seat_contact"


class RackPushPrimitive:
    """Contact-gated, action-only controller for fully sliding a rack inward."""

    MAX_STEPS = 320

    def __init__(self, contract: Any):
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    def actions(self, environment: Any) -> Iterable[np.ndarray]:
        raw = _raw_environment(environment)
        try:
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            base_controller = robot.composite_controller.part_controllers["base"]
            eef_site = robot.eef_site_id["right"]
            joint_id = str(self.contract.fixture_joint_ids["push"])
            joint_index = raw.sim.model.joint_name2id(joint_id)
            if int(raw.sim.model.jnt_type[joint_index]) != 2:
                raise ValueError("rack push requires a slide joint")
            contact_geoms = fixture_close_contact_geometries(self.contract, joint_id)
            # Prefer the broad, thin horizontal rack plate. Its front OBB face
            # is a repeatable physical contact target for the two Panda pads.
            box_ids = [
                raw.sim.model.geom_name2id(name) for name in contact_geoms
                if int(raw.sim.model.geom_type[raw.sim.model.geom_name2id(name)]) == 6
            ]
            plate_id = min(
                box_ids,
                key=lambda geom_id: (
                    float(raw.sim.model.geom_size[geom_id][2]),
                    -float(np.prod(raw.sim.model.geom_size[geom_id][:2])),
                ),
            )
        except (AttributeError, KeyError, IndexError, StopIteration, TypeError, ValueError):
            yield from PrimitiveExpert("push", 0).actions(environment)
            return

        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        joint_range = np.asarray(raw.sim.model.jnt_range[joint_index], dtype=float)
        closed_threshold = float(np.min(np.abs(joint_range))) + .05 * float(np.ptp(joint_range))
        push_direction = fixture_close_push_direction(
            joint_type=2,
            joint_position=float(raw.sim.data.get_joint_qpos(joint_id)),
            joint_axis_world=np.asarray(raw.sim.data.xaxis[joint_index], dtype=float),
            joint_anchor_world=np.asarray(raw.sim.data.xanchor[joint_index], dtype=float),
            handle_world=np.asarray(raw.sim.data.geom_xpos[plate_id], dtype=float),
        )
        contact_seen = False
        seat_latched = False
        push_latched = False
        for step in range(1, self.MAX_STEPS + 1):
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            center = np.asarray(raw.sim.data.geom_xpos[plate_id], dtype=float)
            rotation = np.asarray(raw.sim.data.geom_xmat[plate_id], dtype=float).reshape(3, 3)
            size = np.asarray(raw.sim.model.geom_size[plate_id], dtype=float)
            face = rack_front_contact_point(
                center=center, rotation=rotation, half_size=size,
                push_direction=push_direction,
            )
            joint_position = float(raw.sim.data.get_joint_qpos(joint_id))
            exact_contact = _selected_geom_has_gripper_contact(raw, contact_geoms)
            contact_seen = contact_seen or exact_contact
            precontact = face - push_direction * .055
            pad_center = selected_gripper_pad_center(raw)
            stage = rack_contact_stage(
                joint_closed=joint_position <= closed_threshold,
                exact_contact=exact_contact,
                pad_precontact_distance=float(np.linalg.norm(
                    (pad_center if pad_center is not None else eef) - precontact
                )),
                seat_latched=seat_latched,
                push_latched=push_latched,
            )
            seat_latched = seat_latched or stage == "seat_contact"
            push_latched = push_latched or exact_contact
            if stage == "settle_closed":
                target = eef
            elif stage == "push_close":
                target = face + push_direction * (max(joint_position, 0.) + .035)
            elif stage == "preapproach":
                target = precontact
            else:
                target = rack_seat_contact_point(
                    face=face, push_direction=push_direction,
                )
            if pad_center is not None and stage != "settle_closed":
                target = rack_wrist_target_for_pad_contact(
                    eef=eef, pad_center=pad_center, desired_pad_center=target,
                )
            error = (
                np.asarray(controller.world_to_origin_frame(target), dtype=float)
                - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
            )
            action = np.zeros_like(low, dtype=np.float32)
            action[:3] = np.clip(error / .05, -.65, .65)
            action[layout.gripper] = -1.0
            action[layout.base_mode] = -1.0
            self.trace.append({
                "step": step, "stage": stage, "joint_id": joint_id,
                "joint_position": joint_position, "closed_threshold": closed_threshold,
                "exact_rack_contact": exact_contact, "contact_seen": contact_seen,
                "eef_face_distance": float(np.linalg.norm(eef - face)),
                "pad_face_distance": None if pad_center is None else float(np.linalg.norm(pad_center - face)),
                "pad_center_world": None if pad_center is None else np.asarray(pad_center, dtype=float).round(6).tolist(),
                "rack_face_world": np.asarray(face, dtype=float).round(6).tolist(),
                "precontact_world": np.asarray(precontact, dtype=float).round(6).tolist(),
                "target_world": np.asarray(target, dtype=float).round(6).tolist(),
                "push_direction_world": np.asarray(push_direction, dtype=float).round(6).tolist(),
                "controller_error": np.asarray(error, dtype=float).round(6).tolist(),
                "action": np.asarray(action, dtype=float).round(6).tolist(),
            })
            yield np.clip(action, low, high)
            if stage == "settle_closed":
                return


class FixtureOpenPrimitive:
    """Contact-gated, action-only opening controller for a hinged fixture."""

    MAX_STEPS = 420

    def __init__(self, contract: Any):
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    def actions(self, environment: Any) -> Iterable[np.ndarray]:
        raw = _raw_environment(environment)
        selection_id = str(getattr(getattr(environment, "request", None), "selection_id", ""))
        try:
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            eef_site = robot.eef_site_id["right"]
            joint_id = str(self.contract.fixture_joint_ids["open"])
            joint_index = raw.sim.model.joint_name2id(joint_id)
            contact_geoms = fixture_close_contact_geometries(self.contract, joint_id)
            handle_name = fixture_close_contact_target_name(
                labelled_geoms=tuple(
                    name for name in getattr(self.contract, "object_geom_names", {}).get("obj", ())
                    if "handle" in str(name).lower()
                ),
                joint_bound_geoms=contact_geoms,
                contactable_geoms=tuple(
                    name for name in contact_geoms
                    if int(raw.sim.model.geom_contype[raw.sim.model.geom_name2id(name)])
                    or int(raw.sim.model.geom_conaffinity[raw.sim.model.geom_name2id(name)])
                ),
                geom_positions={
                    name: np.asarray(raw.sim.data.geom_xpos[raw.sim.model.geom_name2id(name)], dtype=float)
                    for name in contact_geoms
                },
                eef_world=np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float),
            )
            handle_id = raw.sim.model.geom_name2id(handle_name)
            joint_type = int(raw.sim.model.jnt_type[joint_index])
        except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as error:
            self.trace.append({"stage": "fallback", "setup_error": f"{type(error).__name__}: {error}"})
            yield from PrimitiveExpert("open", 0).actions(environment)
            return
        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        contact_seen = False
        last_contact_normal: np.ndarray | None = None
        for step in range(1, self.MAX_STEPS + 1):
            handle = np.asarray(raw.sim.data.geom_xpos[handle_id], dtype=float)
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            joint_position = float(raw.sim.data.get_joint_qpos(joint_id))
            contact_details = selected_gripper_contact_details(raw, contact_geoms)
            exact_contact = _selected_geom_has_gripper_contact(raw, contact_geoms)
            contact_seen = contact_seen or exact_contact
            if exact_contact:
                normal = fixture_close_deepest_contact_normal(contact_details)
                if normal is not None:
                    last_contact_normal = normal
            if fixture_open_presentation_complete(
                contact_seen=contact_seen, joint_position=joint_position,
            ):
                self.trace.append({
                    "step": step, "stage": "open_complete", "joint_id": joint_id,
                    "joint_position": joint_position, "exact_contact": exact_contact,
                })
                return
            direction = fixture_open_pull_direction(
                joint_type=joint_type, joint_position=joint_position,
                joint_axis_world=np.asarray(raw.sim.data.xaxis[joint_index], dtype=float),
                joint_anchor_world=np.asarray(raw.sim.data.xanchor[joint_index], dtype=float),
                handle_world=handle,
            )
            if exact_contact:
                # Maintain a small inward component while pulling tangentially
                # so a real surface contact does not immediately detach.
                inward = last_contact_normal if last_contact_normal is not None else np.zeros(3)
                target = eef + direction * .035 + inward * .012
                stage = "pull_open"
            else:
                target = handle
                stage = "approach_handle"
            error = (
                np.asarray(controller.world_to_origin_frame(target), dtype=float)
                - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
            )
            distance = float(np.linalg.norm(target - eef))
            action = np.zeros_like(low, dtype=np.float32)
            action[:3] = np.clip(error / .05, -.7, .7)
            action[layout.gripper] = .8 if (exact_contact or distance < .08) else -1.0
            action[layout.base_mode] = -1.0
            self.trace.append({
                "step": step, "stage": stage, "joint_id": joint_id,
                "joint_position": joint_position, "exact_contact": exact_contact,
                "contact_seen": contact_seen, "target_distance": distance,
                "action": np.asarray(action, dtype=float).round(6).tolist(),
            })
            yield np.clip(action, low, high)
        self.trace.append({"stage": "open_step_cap_exhausted", "joint_id": joint_id, "contact_seen": contact_seen})


class DrawerClosePrimitive:
    """Contact-gated, action-only controller for a source-labelled fixture close."""

    MAX_STEPS = 700

    def __init__(self, contract: Any):
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    def actions(self, environment: Any) -> Iterable[np.ndarray]:
        raw = _raw_environment(environment)
        selection_id = str(getattr(getattr(environment, "request", None), "selection_id", ""))
        try:
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            base_controller = robot.composite_controller.part_controllers["base"]
            eef_site = robot.eef_site_id["right"]
            joint_id = str(self.contract.fixture_joint_ids["close"])
            joint_index = raw.sim.model.joint_name2id(joint_id)
            selected_geoms = tuple(getattr(self.contract, "object_geom_names", {}).get("obj", ()))
            handle_geoms = tuple(name for name in selected_geoms if "handle" in str(name).lower())
            if not handle_geoms:
                handle_geoms = tuple(
                    name for name in getattr(self.contract, "fixture_contact_joints", {})
                    if "handle" in str(name).lower()
                )
            contact_geoms = fixture_close_contact_geometries(self.contract, joint_id)
            if not contact_geoms:
                contact_geoms = handle_geoms
            handle_grasp_required = fixture_close_requires_two_pad_handle_grasp(selection_id)
            exact_contact_geoms = fixture_close_exact_contact_geometries(
                labelled_handle_geoms=handle_geoms,
                joint_bound_geoms=contact_geoms,
                # VLA82-028 is assessed as a handle-held refrigerator close;
                # door-panel pushes are therefore never a valid close trigger.
                allow_panel_contact=not handle_grasp_required,
            )
            handle_name = fixture_close_contact_target_name(
                labelled_geoms=handle_geoms,
                joint_bound_geoms=contact_geoms,
                contactable_geoms=tuple(
                    str(name) for name in contact_geoms
                    if int(raw.sim.model.geom_contype[raw.sim.model.geom_name2id(name)]) != 0
                    or int(raw.sim.model.geom_conaffinity[raw.sim.model.geom_name2id(name)]) != 0
                ),
                geom_positions={
                    str(name): np.asarray(raw.sim.data.geom_xpos[raw.sim.model.geom_name2id(name)], dtype=float)
                    for name in contact_geoms
                },
                geom_half_sizes={
                    str(name): np.asarray(raw.sim.model.geom_size[raw.sim.model.geom_name2id(name)], dtype=float)
                    for name in contact_geoms
                },
                eef_world=np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float),
                prefer_nearest_joint_bound="microjoint" in joint_id.lower(),
                prefer_grasp_bar=handle_grasp_required,
            )
            handle_id = raw.sim.model.geom_name2id(handle_name)
            joint_type = int(raw.sim.model.jnt_type[joint_index])
        except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as error:
            self.trace.append({
                "stage": "fallback",
                "setup_error": f"{type(error).__name__}: {error}",
            })
            yield from PrimitiveExpert("close", 0).actions(environment)
            return
        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        contact_seen = False
        contact_offset: np.ndarray | None = None
        orientation_ready = not bool(handle_geoms)
        pregrasp_reached = not handle_grasp_required
        safe_lift_reached = not handle_grasp_required
        safe_approach_reached = not handle_grasp_required
        safe_approach_height: float | None = None
        handle_grasp_latched = False
        bypass_required = fixture_close_requires_free_edge_bypass(
            joint_id,
            initial_joint_position=float(raw.sim.data.get_joint_qpos(joint_id)),
        )
        bypass_points: dict[str, np.ndarray] | None = None
        bypass_free_edge_complete = False
        bypass_cross_plane_complete = False
        bypass_return_complete = False
        bypass_contact_seen_after_cross = False
        panel_name: str | None = None
        panel_id: int | None = None
        panel_contact_geoms: tuple[str, ...] = ()
        upper_panel_points: dict[str, np.ndarray] | None = None
        upper_raise_complete = False
        upper_traverse_complete = False
        upper_panel_contact_seen = False
        filtered_contact_normal: np.ndarray | None = None
        continuous_contact_steps = 0
        contact_loss_reseat_steps = 0
        raise_stage_frames = 0
        raise_distance_history: list[float] = []
        base_assist_start_world: np.ndarray | None = None
        base_assist_displacement = 0.0
        base_assist_locked = False
        if bypass_required:
            initial_handle = np.asarray(raw.sim.data.geom_xpos[handle_id], dtype=float)
            initial_eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            initial_push_direction = fixture_close_push_direction(
                joint_type=joint_type,
                joint_position=float(raw.sim.data.get_joint_qpos(joint_id)),
                joint_axis_world=np.asarray(raw.sim.data.xaxis[joint_index], dtype=float),
                joint_anchor_world=np.asarray(raw.sim.data.xanchor[joint_index], dtype=float),
                handle_world=initial_handle,
            )
            bypass_points = fixture_hinge_free_edge_waypoints(
                panel_center=initial_handle,
                panel_rotation=np.asarray(raw.sim.data.geom_xmat[handle_id], dtype=float),
                panel_half_size=np.asarray(raw.sim.model.geom_size[handle_id], dtype=float),
                hinge_anchor=np.asarray(raw.sim.data.xanchor[joint_index], dtype=float),
                eef_world=initial_eef,
                push_direction=initial_push_direction,
            )
            panel_name = fixture_close_panel_geom_name(
                candidate_names=contact_geoms,
                geom_types={
                    str(name): int(raw.sim.model.geom_type[raw.sim.model.geom_name2id(name)])
                    for name in contact_geoms
                },
                geom_sizes={
                    str(name): np.asarray(
                        raw.sim.model.geom_size[raw.sim.model.geom_name2id(name)], dtype=float,
                    )
                    for name in contact_geoms
                },
            )
            panel_id = raw.sim.model.geom_name2id(panel_name)
            panel_contact_geoms = tuple(
                str(name) for name in contact_geoms
                if str(name) == panel_name
                or (
                    str(name).lower().endswith("door_main")
                    and "handle" not in str(name).lower()
                )
            )
            semantic_handle_name = next(
                (
                    str(name) for name in handle_geoms
                    if str(name).lower().endswith("door_handle_main")
                ),
                str(handle_name),
            )
            semantic_handle_id = raw.sim.model.geom_name2id(semantic_handle_name)
            upper_panel_points = fixture_close_upper_panel_waypoints(
                panel_center=np.asarray(raw.sim.data.geom_xpos[panel_id], dtype=float),
                panel_rotation=np.asarray(raw.sim.data.geom_xmat[panel_id], dtype=float),
                panel_half_size=np.asarray(raw.sim.model.geom_size[panel_id], dtype=float),
                hinge_anchor=np.asarray(raw.sim.data.xanchor[joint_index], dtype=float),
                handle_world=np.asarray(raw.sim.data.geom_xpos[semantic_handle_id], dtype=float),
                push_direction=initial_push_direction,
                outside_closing_side=bypass_points["outside_closing_side"],
            )
        for step in range(1, fixture_close_step_budget(
            selection_id, default_steps=self.MAX_STEPS,
        ) + 1):
            handle = np.asarray(raw.sim.data.geom_xpos[handle_id], dtype=float)
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            pad_center = selected_gripper_pad_center(raw)
            contact_reference = pad_center if pad_center is not None else eef
            named_pad_positions = {
                str(raw.sim.model.geom_id2name(index)): np.asarray(raw.sim.data.geom_xpos[index], dtype=float)
                for index in range(int(raw.sim.model.ngeom))
                if is_gripper_pad_geometry_name(str(raw.sim.model.geom_id2name(index) or ""))
            }
            joint_position = float(raw.sim.data.get_joint_qpos(joint_id))
            contact_details = selected_gripper_contact_details(raw, contact_geoms)
            panel_contact_details = (
                selected_gripper_contact_details(raw, panel_contact_geoms)
                if bypass_required else contact_details
            )
            raw_contact_normal = fixture_close_deepest_contact_normal(panel_contact_details)
            exact_contact = _selected_geom_has_gripper_contact(raw, exact_contact_geoms)
            two_pad_handle_contact = selected_geom_has_two_finger_pad_contacts(
                raw, exact_contact_geoms,
            )
            two_finger_handle_contact = _selected_geom_has_two_finger_contacts(
                raw, exact_contact_geoms,
            )
            handle_grasp_latched = handle_grasp_latched or two_finger_handle_contact
            exact_panel_contact = (
                _selected_geom_has_gripper_contact(raw, panel_contact_geoms)
                if bypass_required else exact_contact
            )
            control_exact_contact = (
                fixture_close_latched_handle_contact(
                    grasp_latched=handle_grasp_latched,
                    exact_handle_contact=exact_contact,
                )
                if handle_grasp_required
                else (exact_panel_contact if bypass_required else exact_contact)
            )
            contact_seen = contact_seen or control_exact_contact
            if control_exact_contact and contact_offset is None:
                contact_offset = contact_reference - handle
            stage = fixture_close_active_stage(
                exact_handle_contact=control_exact_contact,
                contact_seen=contact_seen,
                joint_position=joint_position,
                selection_id="" if handle_grasp_required else selection_id,
                completion_allowed=(not handle_grasp_required or handle_grasp_latched),
            )
            handle_rotation = np.asarray(raw.sim.data.geom_xmat[handle_id], dtype=float).reshape(3, 3)
            handle_size = np.asarray(raw.sim.model.geom_size[handle_id], dtype=float)
            pregrasp_target: np.ndarray | None = None
            safe_lift_target: np.ndarray | None = None
            safe_approach_target: np.ndarray | None = None
            pregrasp_distance: float | None = None
            safe_lift_distance: float | None = None
            safe_approach_distance: float | None = None
            if handle_grasp_required:
                surface_normal = fixture_handle_grasp_surface_normal(
                    handle_rotation=handle_rotation,
                    handle_half_size=handle_size,
                    handle_world=handle,
                    eef_world=eef,
                )
                pregrasp_target = fixture_handle_pregrasp_point(
                    handle_world=handle, outward_normal=surface_normal,
                )
                if safe_approach_height is None:
                    safe_approach_height = float(fixture_handle_safe_lift_target_for_selection(
                        selection_id,
                        eef_world=eef, pregrasp_target=pregrasp_target,
                        outward_normal=surface_normal,
                    )[2])
                safe_lift_target = fixture_handle_safe_lift_target_for_selection(
                    selection_id,
                    eef_world=eef, pregrasp_target=pregrasp_target,
                    outward_normal=surface_normal,
                )
                safe_lift_target[2] = safe_approach_height
                safe_lift_distance = float(np.linalg.norm(
                    contact_reference - safe_lift_target,
                ))
                safe_lift_reached = fixture_handle_waypoint_state(
                    previously_reached=safe_lift_reached,
                    distance=safe_lift_distance,
                    tolerance=.055,
                )
                safe_approach_target = fixture_handle_safe_approach_point(
                    pregrasp_target=pregrasp_target,
                    eef_world=(0., 0., safe_approach_height),
                )
                safe_approach_distance = float(np.linalg.norm(
                    contact_reference - safe_approach_target,
                ))
                safe_approach_reached = fixture_handle_waypoint_state(
                    previously_reached=safe_approach_reached,
                    distance=safe_approach_distance,
                    tolerance=.07,
                )
                pregrasp_distance = float(np.linalg.norm(contact_reference - pregrasp_target))
                pregrasp_reached = fixture_handle_pregrasp_state(
                    previously_reached=pregrasp_reached,
                    distance=pregrasp_distance,
                )
            rotation_command = np.zeros(3, dtype=float)
            orientation_error = 0.0
            tool_alignment_error = 0.0
            base_position, base_rotation = base_controller.get_base_pose()
            base_position = np.asarray(base_position, dtype=float)
            if not orientation_ready and pad_center is not None and len(named_pad_positions) >= 2:
                pad_points = tuple(named_pad_positions.values())
                opening_axis = pad_points[-1] - pad_points[0]
                tool_axis = fixture_handle_tool_axis_from_eef_rotation(
                    np.asarray(raw.sim.data.site_xmat[eef_site], dtype=float),
                )
                long_axis_index = int(np.argmax(handle_size))
                handle_axis = handle_rotation[:, long_axis_index]
                if joint_type == 3 and not handle_grasp_required:
                    outward = fixture_handle_outward_normal(
                        joint_axis_world=np.asarray(raw.sim.data.xaxis[joint_index], dtype=float),
                        handle_world=handle,
                        joint_anchor_world=np.asarray(raw.sim.data.xanchor[joint_index], dtype=float),
                        eef_world=eef,
                    )
                else:
                    outward = fixture_handle_grasp_surface_normal(
                        handle_rotation=handle_rotation,
                        handle_half_size=handle_size,
                        handle_world=handle,
                        eef_world=eef,
                    )
                rotation_command, tool_alignment_error = fixture_handle_tool_alignment_command(
                    tool_axis_world=tool_axis,
                    outward_normal_world=outward, base_rotation=base_rotation,
                )
                if tool_alignment_error > .18:
                    orientation_error = tool_alignment_error
                    stage = "orient_tool"
                else:
                    rotation_command, orientation_error = fixture_handle_roll_command(
                        opening_axis_world=opening_axis,
                        handle_axis_world=handle_axis,
                        tool_axis_world=tool_axis,
                        base_rotation=base_rotation,
                    )
                    orientation_ready = orientation_error <= .18
                if not orientation_ready and stage != "orient_tool":
                    stage = "orient_handle"
            if handle_grasp_required:
                if not safe_lift_reached:
                    stage = "safe_lift_above_door"
                elif not safe_approach_reached:
                    stage = "safe_approach_above_handle"
                elif orientation_ready and not pregrasp_reached:
                    # Descend only after the parallel jaws are aligned in an
                    # exterior, obstacle-free corridor above the door.
                    stage = "pregrasp_handle"
            push_direction = fixture_close_push_direction(
                joint_type=joint_type,
                joint_position=joint_position,
                joint_axis_world=np.asarray(raw.sim.data.xaxis[joint_index], dtype=float),
                joint_anchor_world=np.asarray(raw.sim.data.xanchor[joint_index], dtype=float),
                handle_world=handle,
            )
            if handle_grasp_required:
                push_direction = fixture_close_handle_grasp_direction(
                    selection_id, push_direction,
                )
            push_direction = fixture_close_selection_push_direction(
                selection_id, push_direction,
            )
            raise_target_distance: float | None = None
            if (
                bypass_required and orientation_ready
                and bypass_points is not None and upper_panel_points is not None
            ):
                free_edge_distance = float(np.linalg.norm(
                    contact_reference - bypass_points["outside_current_side"]
                ))
                if free_edge_distance <= .025:
                    bypass_free_edge_complete = True
                cross_plane_distance = float(np.linalg.norm(
                    contact_reference - bypass_points["outside_closing_side"]
                ))
                if bypass_free_edge_complete and cross_plane_distance <= .025:
                    bypass_cross_plane_complete = True
                if not bypass_free_edge_complete:
                    stage = "bypass_to_free_edge"
                elif not bypass_cross_plane_complete:
                    stage = "bypass_cross_plane"
                else:
                    raise_distance = float(np.linalg.norm(
                        contact_reference - upper_panel_points["raise"]
                    ))
                    raise_target_distance = raise_distance
                    if raise_distance <= .025:
                        upper_raise_complete = True
                    traverse_distance = float(np.linalg.norm(
                        contact_reference - upper_panel_points["precontact"]
                    ))
                    if upper_raise_complete and traverse_distance <= .025:
                        upper_traverse_complete = True
                    upper_panel_contact_seen = (
                        upper_panel_contact_seen or exact_panel_contact
                    )
                    bypass_contact_seen_after_cross = upper_panel_contact_seen
                    stage = fixture_close_upper_panel_stage(
                        joint_closed=abs(joint_position) <= .08,
                        raised=upper_raise_complete,
                        traversed=upper_traverse_complete,
                        exact_panel_contact=exact_panel_contact,
                        panel_contact_seen=upper_panel_contact_seen,
                    )
            contact_preserving_direction = push_direction
            normal_bias = 0.0
            if bypass_required and bypass_cross_plane_complete:
                if exact_panel_contact and raw_contact_normal is not None:
                    filtered_contact_normal = fixture_close_filtered_contact_normal(
                        filtered_contact_normal, raw_contact_normal,
                    )
                    continuous_contact_steps += 1
                    contact_loss_reseat_steps = 0
                    contact_preserving_direction, normal_bias = (
                        fixture_close_contact_preserving_direction(
                            tangent=push_direction,
                            contact_normal=filtered_contact_normal,
                        )
                    )
                elif filtered_contact_normal is not None and bypass_contact_seen_after_cross:
                    continuous_contact_steps = 0
                    contact_loss_reseat_steps += 1
                    if contact_loss_reseat_steps > 20:
                        filtered_contact_normal = None
            if stage in {"orient_tool", "orient_handle"}:
                target = eef
            elif stage == "safe_lift_above_door" and safe_lift_target is not None:
                target = safe_lift_target
            elif stage == "safe_approach_above_handle" and safe_approach_target is not None:
                target = safe_approach_target
            elif stage == "pregrasp_handle" and pregrasp_target is not None:
                target = pregrasp_target
            elif stage == "bypass_to_free_edge" and bypass_points is not None:
                target = bypass_points["outside_current_side"]
            elif stage == "bypass_cross_plane" and bypass_points is not None:
                target = bypass_points["outside_closing_side"]
            elif stage == "raise_above_handle" and upper_panel_points is not None:
                target = upper_panel_points["raise"]
            elif stage == "traverse_to_upper_panel" and upper_panel_points is not None:
                target = upper_panel_points["precontact"]
            elif stage == "seat_upper_panel" and upper_panel_points is not None:
                target = upper_panel_points["face"]
            elif (
                stage == "reseat_upper_panel"
                and bypass_required
                and filtered_contact_normal is not None
                and 0 < contact_loss_reseat_steps <= 20
            ):
                target = fixture_close_contact_reseat_target(
                    contact_reference=contact_reference,
                    last_contact_normal=filtered_contact_normal,
                )
            elif stage == "reseat_upper_panel" and upper_panel_points is not None:
                target = upper_panel_points["face"]
            else:
                target = fixture_close_target(
                    stage=stage,
                    handle_world=handle,
                    eef_world=contact_reference,
                    push_direction=(
                        contact_preserving_direction
                        if stage == "push_close" and bypass_required
                        else push_direction
                    ),
                    push_distance=fixture_close_effective_push_distance(
                        selection_id, joint_type=joint_type,
                    ),
                    contact_offset=contact_offset,
                    joint_type=joint_type,
                )
            if pad_center is not None and stage not in {"orient_tool", "orient_handle"}:
                target = fixture_close_wrist_target_for_pad_contact(
                    eef=eef, pad_center=pad_center, desired_pad_center=target,
                )
            error = np.asarray(controller.world_to_origin_frame(target), dtype=float) - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
            action = np.zeros_like(low, dtype=np.float32)
            action[:3] = np.clip(error / .05, -.65, .65)
            action[3:6] = fixture_handle_pregrasp_rotation_command(
                stage, rotation_command, selection_id=selection_id,
            )
            action[layout.gripper] = fixture_close_gripper_command(
                handle_grasp_required=handle_grasp_required,
                bypass_required=bypass_required,
                contact_seen=contact_seen,
                pad_handle_distance=float(np.linalg.norm(contact_reference - handle)),
                pregrasp_reached=pregrasp_reached,
                orientation_ready=orientation_ready,
            )
            material_base_collision = False
            try:
                for contact_index in range(int(raw.sim.data.ncon)):
                    base_contact = raw.sim.data.contact[contact_index]
                    base_contact_names = (
                        str(raw.sim.model.geom_id2name(base_contact.geom1) or "").lower(),
                        str(raw.sim.model.geom_id2name(base_contact.geom2) or "").lower(),
                    )
                    if is_material_mobile_base_fixture_contact(
                        base_contact_names, distance=float(base_contact.dist),
                    ):
                        material_base_collision = True
                        break
            except (AttributeError, IndexError, TypeError):
                material_base_collision = False
            if material_base_collision:
                base_assist_locked = True
            if base_assist_start_world is not None:
                base_assist_displacement = float(np.linalg.norm(
                    base_position[:2] - base_assist_start_world[:2]
                ))
            if stage == "raise_above_handle" and raise_target_distance is not None:
                raise_stage_frames += 1
                raise_distance_history.append(float(raise_target_distance))
                raise_distance_history = raise_distance_history[-40:]
            base_assist_command = fixture_raise_base_assist_local(
                stage=stage,
                stage_frames=raise_stage_frames,
                distance_history=raise_distance_history,
                target_delta_world=(
                    np.zeros(3, dtype=float)
                    if upper_panel_points is None
                    else upper_panel_points["raise"] - contact_reference
                ),
                base_rotation=base_rotation,
                cumulative_displacement=base_assist_displacement,
                collision=material_base_collision,
                locked=base_assist_locked,
            ) if bypass_required else None
            if base_assist_command is not None and base_assist_start_world is None:
                base_assist_start_world = base_position.copy()
            base_follow = fixture_close_base_follow_local(
                joint_type=joint_type, exact_contact=exact_contact,
                joint_position=joint_position, push_direction_world=push_direction,
                base_rotation=base_rotation,
            )
            if (
                base_follow is None
                and stage == "safe_approach_above_handle"
                and safe_approach_target is not None
                and safe_approach_distance is not None
            ):
                base_follow = fixture_handle_stage_base_follow(
                    stage=stage,
                    bypass_required=bypass_required,
                    target_distance=safe_approach_distance,
                    target_delta_world=safe_approach_target - contact_reference,
                    base_rotation=base_rotation,
                )
            if base_follow is None and stage == "approach_handle" and not bypass_required:
                base_follow = fixture_approach_base_local(
                    pad_handle_distance=float(np.linalg.norm(contact_reference - handle)),
                    handle_delta_world=handle - contact_reference,
                    base_rotation=base_rotation,
                )
            if base_assist_command is not None:
                base_follow = base_assist_command
            if base_follow is None:
                action[layout.base_mode] = -1.0
            else:
                action[layout.base[0]:layout.base[0] + 2] = base_follow
                action[layout.base_mode] = 1.0
            self.trace.append({
                "step": step, "stage": stage, "joint_id": joint_id,
                "handle_geom": str(handle_name),
                "handle_rotation": np.asarray(raw.sim.data.geom_xmat[handle_id], dtype=float).reshape(3, 3).round(6).tolist(),
                "handle_half_size": np.asarray(raw.sim.model.geom_size[handle_id], dtype=float).round(6).tolist(),
                "eef_rotation": np.asarray(raw.sim.data.site_xmat[eef_site], dtype=float).reshape(3, 3).round(6).tolist(),
                "named_pad_positions": {
                    name: position.round(6).tolist() for name, position in named_pad_positions.items()
                },
                "handle_orientation_error_rad": orientation_error,
                "handle_tool_alignment_error_rad": tool_alignment_error,
                "handle_orientation_ready": orientation_ready,
                "bypass_required": bypass_required,
                "bypass_free_edge_world": None if bypass_points is None else np.asarray(
                    bypass_points["free_edge"], dtype=float,
                ).round(6).tolist(),
                "bypass_outside_current_side_world": None if bypass_points is None else np.asarray(
                    bypass_points["outside_current_side"], dtype=float,
                ).round(6).tolist(),
                "bypass_outside_closing_side_world": None if bypass_points is None else np.asarray(
                    bypass_points["outside_closing_side"], dtype=float,
                ).round(6).tolist(),
                "bypass_contact_prepoint_world": None if bypass_points is None else np.asarray(
                    bypass_points["contact_prepoint"], dtype=float,
                ).round(6).tolist(),
                "bypass_free_edge_complete": bypass_free_edge_complete,
                "bypass_cross_plane_complete": bypass_cross_plane_complete,
                "bypass_return_complete": bypass_return_complete,
                "bypass_contact_seen_after_cross": bypass_contact_seen_after_cross,
                "upper_panel_geom": panel_name,
                "upper_panel_half_size": None if panel_id is None else np.asarray(
                    raw.sim.model.geom_size[panel_id], dtype=float,
                ).round(6).tolist(),
                "upper_panel_raise_world": None if upper_panel_points is None else np.asarray(
                    upper_panel_points["raise"], dtype=float,
                ).round(6).tolist(),
                "upper_panel_precontact_world": None if upper_panel_points is None else np.asarray(
                    upper_panel_points["precontact"], dtype=float,
                ).round(6).tolist(),
                "upper_panel_face_world": None if upper_panel_points is None else np.asarray(
                    upper_panel_points["face"], dtype=float,
                ).round(6).tolist(),
                "upper_panel_free_edge_direction": None if upper_panel_points is None else np.asarray(
                    upper_panel_points["free_edge_direction"], dtype=float,
                ).round(6).tolist(),
                "upper_panel_horizontal_offset": None if upper_panel_points is None else np.asarray(
                    upper_panel_points["horizontal_offset"], dtype=float,
                ).round(6).tolist(),
                "upper_panel_edge_clearance": (
                    None if upper_panel_points is None
                    else float(upper_panel_points["edge_clearance"])
                ),
                "upper_raise_complete": upper_raise_complete,
                "upper_traverse_complete": upper_traverse_complete,
                "upper_panel_contact_seen": upper_panel_contact_seen,
                "handle_grasp_required": handle_grasp_required,
                "handle_grasp_latched": handle_grasp_latched,
                "exact_two_pad_handle_contact": two_pad_handle_contact,
                "exact_two_finger_handle_contact": two_finger_handle_contact,
                "handle_pregrasp_target_world": None if pregrasp_target is None else pregrasp_target.round(6).tolist(),
                "handle_safe_lift_target_world": None if safe_lift_target is None else safe_lift_target.round(6).tolist(),
                "handle_safe_approach_target_world": None if safe_approach_target is None else safe_approach_target.round(6).tolist(),
                "pregrasp_distance": pregrasp_distance,
                "pregrasp_reached": pregrasp_reached,
                "safe_lift_distance": safe_lift_distance,
                "safe_lift_reached": safe_lift_reached,
                "safe_approach_distance": safe_approach_distance,
                "safe_approach_reached": safe_approach_reached,
                "exact_panel_contact": exact_panel_contact,
                "raise_stage_frames": raise_stage_frames,
                "raise_target_distance": raise_target_distance,
                "raise_distance_window_start": (
                    None if not raise_distance_history else raise_distance_history[0]
                ),
                "raise_distance_window_end": (
                    None if not raise_distance_history else raise_distance_history[-1]
                ),
                "raise_distance_window_improvement": (
                    None if len(raise_distance_history) < 40
                    else raise_distance_history[0] - raise_distance_history[-1]
                ),
                "base_assist_eligible": base_assist_command is not None,
                "base_assist_command_local": (
                    None if base_assist_command is None
                    else np.asarray(base_assist_command, dtype=float).round(6).tolist()
                ),
                "base_assist_start_world": (
                    None if base_assist_start_world is None
                    else np.asarray(base_assist_start_world, dtype=float).round(6).tolist()
                ),
                "base_position_world": base_position.round(6).tolist(),
                "base_assist_displacement": base_assist_displacement,
                "base_assist_cap_reached": base_assist_displacement >= .05,
                "material_base_fixture_collision": material_base_collision,
                "base_assist_locked": base_assist_locked,
                "push_direction_world": np.asarray(push_direction, dtype=float).round(6).tolist(),
                "raw_contact_normal_world": None if raw_contact_normal is None else np.asarray(
                    raw_contact_normal, dtype=float,
                ).round(6).tolist(),
                "filtered_contact_normal_world": None if filtered_contact_normal is None else np.asarray(
                    filtered_contact_normal, dtype=float,
                ).round(6).tolist(),
                "normal_bias": normal_bias,
                "contact_preserving_direction_world": np.asarray(
                    contact_preserving_direction, dtype=float,
                ).round(6).tolist(),
                "continuous_contact_steps": continuous_contact_steps,
                "contact_loss_reseat_steps": contact_loss_reseat_steps,
                "fixture_gripper_contacts": contact_details,
                "panel_gripper_contacts": panel_contact_details,
                "joint_position": joint_position, "exact_handle_contact": exact_contact,
                "eef_handle_distance": float(np.linalg.norm(eef - handle)),
                "pad_handle_distance": float(np.linalg.norm(contact_reference - handle)),
                "action": np.asarray(action, dtype=float).round(6).tolist(),
            })
            yield np.clip(action, low, high)
            if stage == "settle_closed":
                return


def cleaning_descent_transition(*, distance: float, exact_gripper_contact: bool) -> str:
    """Stop descending once the physical hand first reaches the cleaning tool."""
    return "close" if bool(exact_gripper_contact) or float(distance) <= .055 else "descend"


def cleaning_grasp_target(selection_id: str, object_center: Sequence[float]) -> np.ndarray:
    """Use the upper physical handle of the tall VLA82-003 dish brush."""
    target = np.asarray(object_center, dtype=float).copy()
    if str(selection_id) == "VLA82-003":
        target[2] += .065
    return target


class CleaningPrimitive:
    """Live closed-loop grasp, surface-coverage, and source-return expert.

    Privileged reads are limited to the object/EFF positions and RoboCasa's
    public grasp query for generating training labels.  Every movement still
    goes through the normal PandaOmron action vector yielded to ``env.step``.
    """

    MAX_STEPS = 360
    APPROACH_HEIGHT = 0.15
    LIFT_HEIGHT = 0.12
    CLEAN_TRANSIT_HEIGHT = 0.06
    CLOSE_STEPS = 32
    SECURE_STEPS = 8
    RELEASE_STEPS = 28
    FLAT_TOOL_ORIENT_STEPS = 27
    REACH_STEPS = 24
    YAW_ALIGN_STEPS = 60

    def __init__(self, phases: Sequence[str], contract: Any):
        self.phases = tuple(phases)
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    @staticmethod
    def _distance(controller: Any, current: np.ndarray, target: np.ndarray) -> float:
        return float(np.linalg.norm(
            np.asarray(controller.world_to_origin_frame(target), dtype=float)
            - np.asarray(controller.world_to_origin_frame(current), dtype=float)
        ))

    @staticmethod
    def _bounded_point(point: np.ndarray, target: Any, margin: float = 0.06) -> np.ndarray:
        lower = np.asarray(target.min_corner, dtype=float)
        upper = np.asarray(target.max_corner, dtype=float)
        # The source support can be broad, but selected cleaning points must
        # be safely inside its live collision geometry rather than a fixture
        # name guessed from the mapping table.
        low_xy = lower[:2] + margin
        high_xy = upper[:2] - margin
        result = np.asarray(point, dtype=float).copy()
        result[:2] = np.minimum(np.maximum(result[:2], low_xy), high_xy)
        return result

    def actions(self, environment: Any) -> Iterable[tuple[str, np.ndarray]]:
        raw = _raw_environment(environment)
        try:
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            eef_site = robot.eef_site_id["right"]
            body_id = raw.obj_body_id["obj"]
            target_id = next(iter(getattr(self.contract, "target_geometries", {})))
            target = self.contract.target_geometries[target_id]
            object_geoms = tuple(getattr(self.contract, "object_geom_names", {}).get("obj", ()))
            object_half_height = max(float(raw.sim.model.geom_size[raw.sim.model.geom_name2id(name)][2]) for name in object_geoms)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            for index, phase in enumerate(self.phases):
                for action in PrimitiveExpert(phase, index).actions(environment):
                    yield phase, action
            return
        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        clean_phase = "wipe" if "wipe" in self.phases else "scrub"
        selection_id = str(getattr(getattr(environment, "request", None), "selection_id", ""))
        # A trial vertical-pinch rotation was physically unreachable above the
        # counter. The source-flat sponge is instead yaw-aligned by the
        # pre-reset Task-2 placement sampler; keep normal controller posture.
        state = "reach_base"
        state_steps = 0
        wipe_index = 0
        source_position = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float).copy()
        grasp_offset: np.ndarray | None = None
        align_best = float("inf")
        align_stagnant = 0
        wipe_started = False
        wipe_regrasp_attempts = 0
        contact_eef_z: float | None = None

        def target_contact_force() -> float:
            force = 0.0
            try:
                for index in range(int(raw.sim.data.ncon)):
                    contact = raw.sim.data.contact[index]
                    names = (raw.sim.model.geom_id2name(contact.geom1) or "", raw.sim.model.geom_id2name(contact.geom2) or "")
                    if set(object_geoms).intersection(names) and any(str(target.fixture_id) in name for name in names):
                        if int(contact.efc_address) >= 0:
                            force = max(force, abs(float(raw.sim.data.efc_force[int(contact.efc_address)])))
            except (AttributeError, TypeError, IndexError):
                pass
            return force

        def action_for(target_world: np.ndarray, *, closed: bool, translation_limit: float | None = None, rotation: np.ndarray | None = None) -> np.ndarray:
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            error = np.asarray(controller.world_to_origin_frame(target_world), dtype=float) - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
            action = np.zeros_like(low, dtype=np.float32)
            limit = float(translation_limit) if translation_limit is not None else (0.8 if closed else 1.0)
            action[:3] = np.clip(error / 0.05, -limit, limit)
            if rotation is not None:
                action[3:6] = np.asarray(rotation, dtype=np.float32)
            if action.size > layout.gripper:
                # Flat source tools need the calibrated full Panda gripper
                # command. A partial close can momentarily touch both pads
                # yet lose the tool as soon as the support load transfers.
                action[layout.gripper] = 1.0 if closed else -1.0
            action[layout.base_mode] = -1.0
            return np.clip(action, low, high)

        def gripper_physics() -> dict[str, float]:
            """Read contact load and finger spacing for the training trace."""
            try:
                model, data = raw.sim.model, raw.sim.data
                fingers = [
                    np.asarray(data.geom_xpos[index], dtype=float)
                    for index in range(int(model.ngeom))
                    if "finger" in (model.geom_id2name(index) or "").lower()
                    and "collision" in (model.geom_id2name(index) or "").lower()
                    and "pad" not in (model.geom_id2name(index) or "").lower()
                ]
                separation = float(np.linalg.norm(fingers[0] - fingers[1])) if len(fingers) >= 2 else 0.0
                normal = 0.0
                selected = set(object_geoms)
                for index in range(int(data.ncon)):
                    contact = data.contact[index]
                    names = {model.geom_id2name(contact.geom1) or "", model.geom_id2name(contact.geom2) or ""}
                    if selected & names and any("finger" in name.lower() for name in names):
                        # RoboSuite's binding exposes the resolved contact
                        # constraint force directly; native ``mujoco`` calls
                        # against this wrapper otherwise report a false zero.
                        address = int(getattr(contact, "efc_address", -1))
                        if address >= 0:
                            normal = max(normal, abs(float(data.efc_force[address])))
                return {"finger_separation": separation, "normal_force": normal}
            except (AttributeError, TypeError, ValueError, IndexError):
                return {"finger_separation": 0.0, "normal_force": 0.0}

        def base_reach_action() -> tuple[np.ndarray, float]:
            base = raw.robots[0].composite_controller.part_controllers["base"]
            base_pos, base_rot = base.get_base_pose()
            delta = obj[:2] - np.asarray(base_pos, dtype=float)[:2]
            distance = float(np.linalg.norm(delta))
            action = np.zeros_like(low, dtype=np.float32)
            # Native sponge has a low tabletop grasp pose; move the mobile
            # base into Panda's comfortable 25--30 cm arm workspace before
            # freezing base mode and beginning the arm-only oracle sequence.
            if distance > 0.28:
                world_step = delta / max(distance, 1e-6) * min(distance - .28, .12)
                local_step = np.asarray(base_rot, dtype=float)[:2, :2].T @ world_step
                action[layout.base[0]:layout.base[0] + 2] = np.clip(local_step / .08, -.5, .5)
                action[layout.base_mode] = 1.0
            else:
                action[layout.base_mode] = -1.0
            return action, distance

        for step in range(1, self.MAX_STEPS + 1):
            error: float | None = None
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
            grasp_target = cleaning_grasp_target(selection_id, obj)
            surface_z = float(source_position[2] - object_half_height)
            bottom_gap = float(obj[2] - object_half_height - surface_z)
            grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            object_contact = _selected_geom_has_gripper_contact(raw, object_geoms)
            target_surface_contact = any(
                contact.object_id == "obj" and contact.target_id == target_id
                for contact in snapshot_from_environment(environment, step, contract=self.contract).contact_evidence
            )
            label = "return" if state.startswith("return") or state in {"release", "retreat", "settle"} else (clean_phase if wipe_started else "grasp")
            if state == "reach_base":
                action, base_distance = base_reach_action()
                state_steps += 1
                if base_distance <= .31 or state_steps >= self.REACH_STEPS:
                    state, state_steps = "align_yaw", 0
            elif state == "align_yaw":
                # OBB narrow horizontal axis and finger opening axis are read
                # from live MuJoCo geometry; only Z wrist rotation is allowed.
                geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                obb = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                size = np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                narrow = obb[:2, int(np.argmin(size[:2]))]
                fingers = [np.asarray(raw.sim.data.geom_xpos[index], dtype=float) for index in range(int(raw.sim.model.ngeom)) if "finger" in (raw.sim.model.geom_id2name(index) or "").lower() and "collision" in (raw.sim.model.geom_id2name(index) or "").lower() and "pad" not in (raw.sim.model.geom_id2name(index) or "").lower()]
                opening = fingers[1][:2] - fingers[0][:2] if len(fingers) >= 2 else np.array((0., 1.))
                opening /= max(float(np.linalg.norm(opening)), 1e-6)
                narrow /= max(float(np.linalg.norm(narrow)), 1e-6)
                def axis_error(axis: np.ndarray) -> float:
                    return float(np.arctan2(opening[0] * axis[1] - opening[1] * axis[0], float(np.dot(opening, axis))))
                direct, inverse = axis_error(narrow), axis_error(-narrow)
                error = direct if abs(direct) <= abs(inverse) else inverse
                magnitude = abs(error)
                if align_best - magnitude < np.deg2rad(1.0):
                    align_stagnant += 1
                else:
                    align_best, align_stagnant = magnitude, 0
                action = action_for(eef, closed=False, rotation=np.array((0., 0., np.clip(error / .35, -.5, .5))))
                state_steps += 1
                if abs(error) < np.deg2rad(5.0) or state_steps >= self.YAW_ALIGN_STEPS or align_stagnant >= 5:
                    state, state_steps = "approach", 0
            elif state == "orient":
                action = action_for(eef, closed=False, rotation=np.array((1.0, 0.0, 0.0)))
                state_steps += 1
                if state_steps >= self.FLAT_TOOL_ORIENT_STEPS:
                    state, state_steps = "approach", 0
            elif state == "approach":
                desired = grasp_target + np.array((0.0, 0.0, self.APPROACH_HEIGHT))
                action = action_for(desired, closed=False)
                if self._distance(controller, eef, desired) <= 0.035:
                    state, state_steps = "descend", 0
            elif state == "descend":
                desired = grasp_target + np.array((0.0, 0.0, 0.025))
                action = action_for(desired, closed=False)
                # The PandaOmron wrist reference sits several centimetres
                # above a flat sponge.  Its physical fingers reach the object
                # before that reference reaches the object's centre.
                if cleaning_descent_transition(
                    distance=self._distance(controller, eef, desired),
                    exact_gripper_contact=object_contact,
                ) == "close":
                    state, state_steps = "close", 0
            elif state == "close":
                # Keep servoing to the object's observed body centre while
                # closing. Holding the wrist at the pre-contact reference
                # leaves PandaOmron's finger pads above tall, narrow tools.
                # Reuse the calibrated initial-grasp contact centre exactly.
                # The prior lowered target left recovery 3.4 cm from the tool
                # and never satisfied raw._check_grasp.
                action = action_for(grasp_target, closed=True)
                state_steps += 1
                if object_contact:
                    # Stop the downward servo at the *first* exact finger
                    # contact. Waiting for the following grasp flag while
                    # continuing towards the object centre squeezes a flat
                    # sponge into its support and destroys the emerging pinch.
                    grasp_offset = eef - obj
                    state, state_steps = "secure", 0
                elif state_steps >= self.CLOSE_STEPS:
                    # A new attempt still uses observed object state; a failed
                    # grasp is never converted into a successful label.
                    state, state_steps = "approach", 0
            elif state == "secure":
                action = action_for(eef, closed=True, translation_limit=0.10)
                if grasped:
                    state_steps += 1
                else:
                    state_steps += 1
                    if state_steps >= 10:
                        state, state_steps = "close", 0
                # Two physical grasp observations establish the pinch without
                # allowing another downward contact-seeking movement.
                if grasped and state_steps >= self.SECURE_STEPS:
                    state, state_steps = "lift", 0
            elif state == "lift":
                desired = source_position + np.array((0.0, 0.0, self.LIFT_HEIGHT)) + (grasp_offset if grasp_offset is not None else np.zeros(3))
                # Transfer load off the counter promptly while the verified
                # pinch is still present, then revert to a gentle lift.
                action = action_for(desired, closed=True, translation_limit=0.70 if state_steps < 5 else 0.20)
                state_steps += 1
                if not grasped:
                    state, state_steps = "close", 0
                elif obj[2] >= source_position[2] + 0.045:
                    wipe_started = True
                    state, state_steps = "clean_approach", 0
            else:
                if wipe_started and should_recover_wipe_grasp(
                    grasped=grasped,
                    exact_gripper_contact=object_contact,
                    attempts=wipe_regrasp_attempts,
                    wipe_complete=state in {"return_lift", "return_descend", "release", "retreat", "settle"},
                ):
                    # Recovery remains inside the wipe episode: it returns to
                    # the observed object via public actions, verifies the
                    # held-tool condition again, and only then permits any
                    # cleaning-cell credit.
                    wipe_regrasp_attempts += 1
                    grasp_offset = None
                    state, state_steps = "approach", 0
                    action = action_for(obj + np.array((0.0, 0.0, self.APPROACH_HEIGHT)), closed=False)
                elif grasp_offset is None:
                    state, state_steps = "approach", 0
                    action = action_for(obj + np.array((0.0, 0.0, self.APPROACH_HEIGHT)), closed=False)
                else:
                    points = cleaning_coverage_points(source_position, target)
                    if state == "clean_approach":
                        # The verified pinch is already 4.5 cm above its
                        # source before this transition.  A modest 6 cm
                        # transport clearance avoids over-extending the
                        # compliant sponge during lateral motion.
                        # Preserve the actually achieved lift during lateral
                        # travel; asking the compliant tool to climb again is
                        # what previously unloaded the pinch.  Descending is
                        # a separate, contact-guarded phase below.
                        transport_clearance = max(self.CLEAN_TRANSIT_HEIGHT, float(obj[2] - source_position[2]))
                        desired = points[0] + grasp_offset + np.array((0.0, 0.0, transport_clearance))
                        # Match OpenGripperPickPlaceOracle's validated
                        # closed-transit generator: controller-frame error / 
                        # 5cm, clipped at 0.20 with gripper command +1.
                        action = oracle_horizontal_action(controller, eef, desired, low, high, layout=layout)
                        if self._distance(controller, eef, desired) <= 0.04:
                            state, state_steps = "clean_descend", 0
                    elif state == "clean_descend":
                        # Move the current wrist reference down in 2.5-mm
                        # increments; this is a compliant contact seek, not
                        # a COM-target chase through the counter.
                        desired = eef + np.array((0.0, 0.0, -min(0.0025, max(bottom_gap, 0.0))))
                        action = action_for(desired, closed=True, translation_limit=compliant_descend_limit(bottom_gap))
                        action[2] = compliant_descend_command(bottom_gap)
                        at_first_surface_point = float(np.linalg.norm(obj[:2] - points[0][:2])) <= 0.02
                        # Do not chase the COM through the counter.  Exact
                        # declared object/surface contact is the stopping
                        # condition, with a distance fallback only to avoid
                        # an unbounded controller loop on a missed surface.
                        if target_surface_contact and at_first_surface_point and bottom_gap <= 0.003:
                            contact_eef_z = float(eef[2])
                            state, state_steps = "clean_contact", 0
                    elif state == "clean_contact":
                        desired = cleaning_contact_target(
                            points[wipe_index], grasp_offset,
                            contact_eef_z=contact_eef_z if contact_eef_z is not None else float(eef[2]),
                            target_contact=target_surface_contact,
                        )
                        # Lock the proven contact height.  Wiping changes
                        # only the calibrated horizontal axis; if a contact
                        # briefly opens, permit at most a sub-millimetre
                        # downward re-seat, never a COM chase.
                        action = oracle_horizontal_action(controller, eef, desired, low, high, layout=layout)
                        if not target_surface_contact:
                            action[2] = np.clip(
                                (float(desired[2]) - float(eef[2])) / .05, -.02, 0.0,
                            )
                        state_steps += 1
                        at_intended_cell = float(np.linalg.norm(obj[:2] - points[wipe_index][:2])) <= 0.015
                        if target_surface_contact and grasped and object_contact and at_intended_cell and state_steps >= 2:
                            wipe_index += 1
                            state_steps = 0
                            if wipe_index >= len(points):
                                state = "return_lift"
                    elif state == "return_lift":
                        desired = source_position + grasp_offset + np.array((0.0, 0.0, self.LIFT_HEIGHT))
                        action = action_for(desired, closed=True)
                        if self._distance(controller, eef, desired) <= 0.04:
                            state, state_steps = "return_descend", 0
                    elif state == "return_descend":
                        desired = source_position + grasp_offset
                        action = action_for(desired, closed=True)
                        if self._distance(controller, eef, desired) <= 0.025:
                            state, state_steps = "release", 0
                    elif state == "release":
                        action = action_for(eef, closed=False)
                        state_steps += 1
                        if state_steps >= self.RELEASE_STEPS:
                            state, state_steps = "retreat", 0
                    elif state == "retreat":
                        desired = source_position + grasp_offset + np.array((0.0, 0.0, self.APPROACH_HEIGHT))
                        action = action_for(desired, closed=False)
                        if self._distance(controller, eef, desired) <= 0.035:
                            state, state_steps = "settle", 0
                    elif state == "settle":
                        action = action_for(eef, closed=False)
                        state_steps += 1
                        if state_steps >= 8:
                            return
                    else:
                        return
            self.trace.append({
                "step": step, "state": state, "phase": label, "grasped": grasped,
                "exact_gripper_object_contact": object_contact,
                "exact_object_surface_contact": target_surface_contact,
                "object_bottom_gap": bottom_gap,
                "target_contact_normal_force": target_contact_force(),
                "object_z": float(obj[2]), "eef_z": float(eef[2]),
                "opening_axis_alignment_deg": float(np.degrees(abs(error))) if error is not None else None,
                **gripper_physics(),
                "eef_world": eef.round(6).tolist(), "object_world": obj.round(6).tolist(),
                "action": np.asarray(action, dtype=float).round(6).tolist(),
            })
            yield label, action


def pick_place_base_reach_threshold(selection_id: str) -> float:
    """Return the source-validated arm workspace threshold for the selected scene."""
    thresholds = {
        "VLA82-012": .45,
        "VLA82-015": .46,
        "VLA82-016": .65,
        "VLA82-020": .515,
        "VLA82-021": .35,
        "VLA82-037": .39,
        "VLA82-042": .65,
        "VLA82-044": .66,
        "VLA82-046": .65,
        "VLA82-048": .65,
        "VLA82-051": .65,
        "VLA82-052": .65,
        "VLA82-053": .50,
        "VLA82-054": .41,
        "VLA82-055": .65,
        "VLA82-056": .52,
        "VLA82-057": .65,
        "VLA82-058": .65,
    }
    return thresholds.get(str(selection_id), .31)


def pick_place_base_reach_step_budget(selection_id: str) -> int:
    """Give the wide gripper enough collision-checked time to approach safely."""
    selection = str(selection_id)
    if selection == "VLA82-021":
        return 220
    return 120 if selection == "VLA82-020" else PickPlaceExpert.BASE_REACH_STEPS


def pick_place_secure_step_requirement(selection_id: str) -> int:
    """Minimum consecutive native grasp frames before the first lift pulse."""
    return 1 if str(selection_id) in {"VLA82-016", "VLA82-021", "VLA82-046", "VLA82-048"} else PickPlaceExpert.SECURE_STEPS


def pick_place_secure_transition(
    selection_id: str, *, raw_grasp: bool, secure_steps: int,
    two_finger_contact: bool = False,
) -> str:
    """Choose the next state after a closed-gripper hold observation."""
    hold_valid = bool(raw_grasp) or (
        str(selection_id) in {"VLA82-050", "VLA82-056"}
        and bool(two_finger_contact)
    )
    if hold_valid and int(secure_steps) >= pick_place_secure_step_requirement(selection_id):
        return "lift"
    if not hold_valid and int(secure_steps) >= 10:
        return "close"
    return "secure"


def pick_place_initial_lift_limit(selection_id: str) -> float:
    """Use a gentle first lift for a multi-finger grasp still resting on a counter."""
    selected = str(selection_id)
    if selected == "VLA82-051":
        return .12
    return .15 if selected == "VLA82-021" else .25 if selected == "VLA82-004" else .70


def pick_place_lift_limit(selection_id: str, lift_steps: int) -> float:
    selected = str(selection_id)
    if selected in {"VLA82-046", "VLA82-051"}:
        return .15
    if selected == "VLA82-021":
        return .15 if int(lift_steps) < 4 else .25
    if selected == "VLA82-004":
        return .25
    return .45


def pick_place_gripper_hold_command(selection_id: str) -> float:
    """Hold an integrated multi-finger closure instead of over-closing it."""
    return 0.0 if str(selection_id) == "VLA82-021" else 1.0


def pick_place_secure_hold_target(
    selection_id: str,
    *,
    eef: np.ndarray,
    obj: np.ndarray,
    grasp_offset: np.ndarray | None,
) -> np.ndarray:
    """Counter post-contact controller drift for thin top-grasped objects."""
    if str(selection_id) == "VLA82-056" and grasp_offset is not None:
        return np.asarray(obj, dtype=float) + np.asarray(grasp_offset, dtype=float)
    return np.asarray(eef, dtype=float).copy()


def pick_place_bottle_aperture_command(
    selection_id: str, *, aperture: float, previous_aperture: float | None,
    lower: float = .058, upper: float = .062, velocity_tolerance: float = 1e-4,
    grasped: bool = False,
) -> float:
    """Brake Panda's accumulated close action inside the bottle grasp band."""
    selected = str(selection_id)
    if selected not in {"VLA82-004", "VLA82-017"}:
        return 1.0
    if selected == "VLA82-004":
        if bool(grasped):
            return 0.0
        lower, upper = .029, .032
    current = float(aperture)
    if current > float(upper):
        return 1.0
    if current < float(lower):
        return -1.0
    if previous_aperture is None:
        return 0.0
    velocity = current - float(previous_aperture)
    if velocity < -float(velocity_tolerance):
        return -1.0
    if velocity > float(velocity_tolerance):
        return 1.0
    return 0.0


def pick_place_base_reach_state(*, distance: float, steps: int, maximum_steps: int, threshold: float = .31) -> str:
    """Advance only after observed base-to-source reach, never a timer alone."""
    if float(distance) <= float(threshold):
        return "approach"
    if int(steps) >= int(maximum_steps):
        return "failed"
    return "reach_base"


def pick_place_initial_state(selection_id: str) -> str:
    """Use a front-entry grasp after VLA004's collision-free reset alignment."""
    selected = str(selection_id)
    if selected in {"VLA82-004", "VLA82-017"}:
        return "cabinet_front_stage"
    if selected in {"VLA82-005", "VLA82-016", "VLA82-046", "VLA82-048", "VLA82-050", "VLA82-053", "VLA82-055", "VLA82-056"}:
        return "align_yaw"
    return "reach_base"


def pick_place_initial_state_for_request(request: SceneRequest | Any) -> str:
    """Choose cabinet front-entry from authoritative fixture semantics."""
    if str(getattr(request, "source_fixture", "")) == "drawer":
        return (
            "drawer_side_orient"
            if pick_place_grasp_strategy(str(getattr(request, "selection_id", ""))) == "drawer_front_side"
            else "approach"
        )
    if uses_cabinet_source_alignment(request) and str(getattr(request, "selection_id", "")) != "VLA82-021":
        return "cabinet_front_stage"
    return pick_place_initial_state(str(getattr(request, "selection_id", "")))


def pick_place_controller_profile_id(request: SceneRequest | Any) -> str:
    """Reuse calibrated motion profiles for equivalent source/target semantics."""
    if is_counter_to_drawer_request(request):
        return "VLA82-005"
    if uses_cabinet_source_alignment(request) and str(getattr(request, "selection_id", "")) != "VLA82-021":
        return "VLA82-004"
    return str(getattr(request, "selection_id", ""))


def shortest_axis_alignment_error(opening_axis: Sequence[float], object_axis: Sequence[float]) -> float:
    """Signed smallest XY angle between an undirected object axis and finger opening."""
    opening = np.asarray(opening_axis, dtype=float)[:2]
    target = np.asarray(object_axis, dtype=float)[:2]
    opening /= max(float(np.linalg.norm(opening)), 1e-9)
    target /= max(float(np.linalg.norm(target)), 1e-9)

    def signed(axis: np.ndarray) -> float:
        return float(np.arctan2(opening[0] * axis[1] - opening[1] * axis[0], float(np.dot(opening, axis))))

    direct, inverse = signed(target), signed(-target)
    return direct if abs(direct) <= abs(inverse) else inverse


def pick_place_descent_reached(*, distance: float, any_gripper_contact: bool) -> bool:
    """Stop an open-gripper descent at first physical gripper contact."""
    return bool(any_gripper_contact) or float(distance) <= .055


def pick_place_descent_transition(
    *, distance: float, any_gripper_contact: bool, selection_id: str = "",
) -> str:
    """A first shell contact ends the open-hand descent, but is not a grasp."""
    if str(selection_id) in {"VLA82-016", "VLA82-046", "VLA82-048"}:
        # The open finger shells touch the broad top face before their inner
        # pads reach the thin folder edges. Keep descending through that
        # expected symmetric contact until the pad midpoint is inside the
        # grasp band; otherwise closing merely presses on the cover.
        return "close" if float(distance) <= .006 else "descend"
    return "close" if bool(any_gripper_contact) or float(distance) <= .055 else "descend"


def pick_place_existing_grasp_transition(
    *, state: str, raw_grasp: bool, two_pad_contact: bool,
) -> str:
    """Resume secure handling when a retry already owns a native grasp."""
    current = str(state)
    if current in {"approach", "descend", "close"} and bool(raw_grasp) and bool(two_pad_contact):
        return "secure"
    return current


def pick_place_close_world_target(
    selection_id: str, *, eef: Sequence[float], obj: Sequence[float], any_gripper_contact: bool,
) -> np.ndarray:
    """Do not drive the palm through a contacted object while closing."""
    if str(selection_id) in {"VLA82-004", "VLA82-005"} and bool(any_gripper_contact):
        return np.asarray(eef, dtype=float)
    return np.asarray(obj, dtype=float)


def should_lock_vla004_wrist_for_close(
    selection_id: str, *, state: str, any_gripper_contact: bool,
) -> bool:
    """Hold the spray-bottle wrist still after its first close-stage contact."""
    return (
        str(selection_id) == "VLA82-004"
        and str(state) == "close"
        and bool(any_gripper_contact)
    )


def pick_place_transit_reached(*, eef: Sequence[float], desired: Sequence[float], tolerance: float = .035) -> bool:
    """A horizontal transport ends on XY; contact-gated lowering owns Z."""
    return float(np.linalg.norm(np.asarray(eef, dtype=float)[:2] - np.asarray(desired, dtype=float)[:2])) <= float(tolerance)


def pick_place_transit_ready_for_selection(
    selection_id: str, *, eef: Sequence[float], desired: Sequence[float],
    object_center: Sequence[float], release: Sequence[float],
) -> bool:
    """Centre the toothbrush above the narrow cup before vertical insertion."""
    if not pick_place_transit_reached(eef=eef, desired=desired):
        return False
    if str(selection_id) == "VLA82-055" and float(np.asarray(eef, dtype=float)[2]) < float(np.asarray(desired, dtype=float)[2]) - .06:
        return False
    if str(selection_id) in {"VLA82-055", "VLA82-058"}:
        object_xy_tolerance = .015 if str(selection_id) == "VLA82-055" else .01
        return float(np.linalg.norm(
            np.asarray(object_center, dtype=float)[:2]
            - np.asarray(release, dtype=float)[:2]
        )) <= object_xy_tolerance
    return True


def cabinet_target_transport_waypoint(
    *, selection_id: str, object_center: Sequence[float], release: Sequence[float],
    target_low: Sequence[float], target_high: Sequence[float],
    robot_base: Sequence[float], lift_height: float, cleared: bool, raised: bool,
) -> tuple[np.ndarray, bool, bool, bool]:
    """Route a counter object around a cabinet shelf instead of through it."""
    release_point = np.asarray(release, dtype=float)
    if str(selection_id) != "VLA82-056":
        return release_point + np.array((0., 0., float(lift_height))), bool(cleared), bool(raised), True
    obj = np.asarray(object_center, dtype=float)
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    base = np.asarray(robot_base, dtype=float)
    spans = high[:2] - low[:2]
    front_axis = int(np.argmin(spans))
    centre = (low[:2] + high[:2]) / 2.0
    if abs(float(base[front_axis]) - float(centre[front_axis])) <= 1e-9:
        raise ValueError("cabinet target and robot base cannot share the front-axis coordinate")
    outside_axis = (
        float(high[front_axis]) + .06
        if float(base[front_axis]) > float(centre[front_axis])
        else float(low[front_axis]) - .06
    )
    high_z = float(release_point[2] + float(lift_height))

    cleared_now = bool(cleared) or abs(float(obj[front_axis]) - outside_axis) <= .025
    if not cleared_now:
        waypoint = obj.copy()
        waypoint[front_axis] = outside_axis
        return waypoint, False, False, False
    raised_now = bool(raised) or abs(float(obj[2]) - high_z) <= .035
    if not raised_now:
        waypoint = obj.copy()
        waypoint[front_axis] = outside_axis
        waypoint[2] = high_z
        return waypoint, True, False, False
    return np.array((release_point[0], release_point[1], high_z)), True, True, True


def cabinet_target_eef_waypoint(
    *, object_waypoint: Sequence[float], eef: Sequence[float],
    held_offset: Sequence[float], raising: bool,
) -> np.ndarray:
    """Prevent stale grasp offsets from turning a cabinet raise into a diagonal."""
    target = np.asarray(object_waypoint, dtype=float) + np.asarray(held_offset, dtype=float)
    if bool(raising):
        target[:2] = np.asarray(eef, dtype=float)[:2]
    return target


def pick_place_counter_upright_stage_required(selection_id: str) -> bool:
    """Require a tall bottle to be upright before it descends onto a counter."""
    return str(selection_id) == "VLA82-004"


def drawer_tip_first_insertion_enabled(selection_id: str) -> bool:
    """Use a tip-first placement for a long tool grasped near its outer end."""
    return str(selection_id) == "VLA82-005"


def pick_place_post_transit_state(selection_id: str, *, front_insert: bool = False) -> str:
    """Route selection-specific placement preparation after free-space transit."""
    if bool(front_insert):
        return "transit_front"
    if drawer_tip_first_insertion_enabled(selection_id):
        return "tilt_drawer_entry"
    if str(selection_id) == "VLA82-055":
        return "level_for_shelf"
    return "upright_for_counter" if pick_place_counter_upright_stage_required(selection_id) else "lower"


def shelf_leveling_transition(
    *, hold_valid: bool, alignment_error_deg: float, yaw_error_deg: float,
    xy_distance: float, steps: int,
    tolerance_deg: float = 4.0, maximum_steps: int = 80,
) -> str:
    """Gate shelf descent on a physically held, level package pose."""
    if not bool(hold_valid):
        return "close"
    if (
        float(alignment_error_deg) <= float(tolerance_deg)
        and float(yaw_error_deg) <= float(tolerance_deg)
    ):
        return "lower" if float(xy_distance) <= .015 else "transit"
    return "failed" if int(steps) >= int(maximum_steps) else "level_for_shelf"


def shelf_leveling_position_target(
    *, release: Sequence[float], eef: Sequence[float],
    object_center: Sequence[float], lift_height: float,
) -> np.ndarray:
    """Hold position while orientation is corrected; transit owns translation."""
    return np.asarray(eef, dtype=float).copy()


def drawer_tip_tilt_ready(
    *, eef: Sequence[float], object_center: Sequence[float], minimum_offset: float = .065,
) -> bool:
    """Require the grasped end to be safely above the tool centre before descent."""
    return (
        float(np.asarray(eef, dtype=float)[2])
        - float(np.asarray(object_center, dtype=float)[2])
    ) >= float(minimum_offset)


def drawer_tip_release_ready(
    *, inner_bottom_contact: bool, inside_xy: bool, fixture_collision: bool,
) -> bool:
    """Release only after the inserted tip has real internal support."""
    return bool(inner_bottom_contact) and bool(inside_xy) and not bool(fixture_collision)


def drawer_tip_advance_target(
    *, object_center: Sequence[float], target_low: Sequence[float],
    target_high: Sequence[float], robot_base: Sequence[float], distance: float = .12,
) -> np.ndarray:
    """Slide a tilted tool past the front panel without lowering its grasp end."""
    result = np.asarray(object_center, dtype=float).copy()
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    base = np.asarray(robot_base, dtype=float)
    depth_axis = int(np.argmax(high[:2] - low[:2]))
    lateral_axis = 1 - depth_axis
    center = float((low[depth_axis] + high[depth_axis]) / 2.0)
    inward_sign = np.sign(center - float(base[depth_axis])) or 1.0
    result[lateral_axis] = float((low[lateral_axis] + high[lateral_axis]) / 2.0)
    result[depth_axis] += float(inward_sign) * float(distance)
    result[depth_axis] = float(np.clip(result[depth_axis], low[depth_axis], high[depth_axis]))
    return result


def drawer_tip_contact_height(release_height: float, *, preload: float = .008) -> float:
    """Command a small compliant overlap so MuJoCo reports real floor support."""
    return float(release_height) - float(preload)


def pick_place_post_upright_state(selection_id: str) -> str:
    """Create reachable wrist clearance before lowering a tall upright bottle."""
    return "counter_lower_base_reposition" if pick_place_counter_upright_stage_required(selection_id) else "lower"


def pick_place_counter_upright_base_reposition_distance(selection_id: str) -> float:
    """Return the minimal collision-cleared base trim used for upright counter placement."""
    return .08 if pick_place_counter_upright_stage_required(selection_id) else 0.0


def pick_place_counter_upright_base_reposition_direction(selection_id: str) -> float:
    """Open arm down-reach by backing the mobile base away from the counter."""
    return 1.0 if pick_place_counter_upright_stage_required(selection_id) else 0.0


def pick_place_post_counter_base_reposition_state(selection_id: str) -> str:
    """Return to a world-space counter alignment after the mobile-base trim."""
    return "upright_recenter_for_counter" if pick_place_counter_upright_stage_required(selection_id) else "lower"


def upright_axis_rotation_command(
    *, object_axis: Sequence[float], base_rotation: Sequence[Sequence[float]],
    tolerance: float = np.deg2rad(12.), maximum: float = .30,
) -> tuple[np.ndarray, float]:
    """Return a bounded OSC rotation that aligns a held object's long axis to world Z.

    The bottle is axially symmetric, so either direction along its longitudinal
    axis is an upright placement.  The cross-product axis gives the shortest
    right-handed correction; conversion to the mobile-base frame keeps the
    public OSC action valid at every sampled kitchen yaw.
    """
    axis = np.asarray(object_axis, dtype=float)[:3]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    desired = np.array((0., 0., 1.), dtype=float)
    if float(np.dot(axis, desired)) < 0.0:
        desired = -desired
    cosine = float(np.clip(np.dot(axis, desired), -1.0, 1.0))
    error = float(np.arccos(cosine))
    if error <= float(tolerance):
        return np.zeros(3, dtype=float), error
    world_axis = np.cross(axis, desired)
    world_axis /= max(float(np.linalg.norm(world_axis)), 1e-9)
    local_axis = np.asarray(base_rotation, dtype=float).reshape(3, 3).T @ world_axis
    magnitude = min(error / .35, float(maximum))
    return local_axis * magnitude, error


def pick_place_counter_margin(selection_id: str) -> float:
    """Keep the wide dish-brush head and tracking error over a solid slab."""
    return .065 if str(selection_id) == "VLA82-042" else .015


def drawer_transit_ready(
    *, xy_reached: bool, vertical_half_extent: float | None,
    depth_alignment_error: float | None, joint5: float,
    vertical_threshold: float = .03, angular_threshold: float = np.deg2rad(25.),
) -> bool:
    """Gate drawer descent on live position, shape pose, long axis, and wrist margin."""
    return bool(
        xy_reached
        and vertical_half_extent is not None
        and float(vertical_half_extent) <= float(vertical_threshold)
        and depth_alignment_error is not None
        and abs(float(depth_alignment_error)) <= float(angular_threshold)
        and not drawer_wrist_relief_required(joint5=float(joint5))
    )


def drawer_horizontal_rotation_command(
    *, long_axis: Sequence[float], target_low: Sequence[float], target_high: Sequence[float],
    magnitude: float = .25, angular_tolerance: float = np.deg2rad(2.),
) -> np.ndarray:
    """Level a long tool about the axis orthogonal to the drawer depth."""
    axis = np.asarray(long_axis, dtype=float)[:3]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    if abs(float(axis[2])) <= float(np.sin(angular_tolerance)):
        return np.zeros(3, dtype=float)
    depth_axis = int(np.argmax(
        np.asarray(target_high, dtype=float)[:2] - np.asarray(target_low, dtype=float)[:2]
    ))
    depth_component = float(axis[depth_axis])
    depth_sign = 1.0 if abs(depth_component) < .2 else float(np.sign(depth_component))
    vertical_sign = np.sign(float(axis[2])) or 1.0
    command = np.zeros(3, dtype=float)
    if depth_axis == 1:
        command[0] = -vertical_sign * depth_sign * float(magnitude)
    else:
        command[1] = vertical_sign * depth_sign * float(magnitude)
    return command


def drawer_orientation_pose_ready(
    *, vertical_half_extent: float | None, threshold: float = .031,
) -> bool:
    """Accept a physically horizontal drawer object with numerical clearance."""
    return vertical_half_extent is not None and float(vertical_half_extent) <= float(threshold)


def drawer_release_orientation_ready(selection_id: str, *, vertical_half_extent: float | None) -> bool:
    """Keep the long whisk within the measured drawer-height envelope before release."""
    if str(selection_id) == "VLA82-005":
        return drawer_orientation_pose_ready(
            vertical_half_extent=vertical_half_extent,
            threshold=.055,
        )
    return True


def drawer_orientation_control_magnitude(selection_id: str) -> float:
    """Give long cylindrical tools enough authority to escape pose oscillation."""
    selected = str(selection_id)
    return .8 if selected in {"VLA82-006", "VLA82-038", "VLA82-041"} else .25


def drawer_leveling_rotation_latch(
    previous: Sequence[float] | None, candidate: Sequence[float],
) -> np.ndarray:
    """Keep a confirmed leveling direction while a long object crosses vertical."""
    return np.asarray(candidate if previous is None else previous, dtype=float)


def update_drawer_horizontal_latch(
    *, previous: bool, vertical_half_extent: float | None,
) -> bool:
    """Preserve a verified horizontal pose throughout XY transport."""
    return bool(previous) or drawer_orientation_pose_ready(
        vertical_half_extent=vertical_half_extent,
    )


def drawer_lower_rotation_command(
    *, long_axis: Sequence[float], target_low: Sequence[float],
    target_high: Sequence[float], magnitude: float = .25,
) -> np.ndarray:
    """Correct lower-stage pitch along the shortest live long-axis route."""
    return drawer_horizontal_rotation_command(
        long_axis=long_axis, target_low=target_low, target_high=target_high,
        magnitude=magnitude,
    )


def drawer_transport_level_rotation_command(
    *, long_axis: Sequence[float], target_low: Sequence[float],
    target_high: Sequence[float], magnitude: float = .2,
) -> np.ndarray:
    """Level a transported object using its live axis, not a fixed wrist turn."""
    return drawer_horizontal_rotation_command(
        long_axis=long_axis,
        target_low=target_low,
        target_high=target_high,
        magnitude=magnitude,
    )


def drawer_gravity_release_pose_ready(
    *, pose_correction: bool, yaw_correction: bool, joint_correction: bool,
) -> bool:
    """Joint margin must not veto an otherwise safe physical release."""
    del joint_correction
    return not (bool(pose_correction) or bool(yaw_correction))


def drawer_lower_base_recenter_required(
    *, object_center: Sequence[float], release: Sequence[float], threshold: float = .08,
) -> bool:
    """Use the mobile base once low-height arm-only XY correction saturates."""
    return float(np.linalg.norm(
        np.asarray(release, dtype=float)[:2] - np.asarray(object_center, dtype=float)[:2]
    )) > float(threshold)


def drawer_lower_base_recenter_enabled(selection_id: str) -> bool:
    """Keep the base fixed for the rolling pin's millimetre-clearance descent."""
    return str(selection_id) not in {"VLA82-006", "VLA82-038"}


def drawer_locked_descent_enabled(selection_id: str) -> bool:
    """Prevent wrist/base feedback oscillation after a thin tool reaches the slot."""
    return str(selection_id) in {"VLA82-005", "VLA82-038", "VLA82-041"}


def drawer_locked_descent_keeps_pose_correction(selection_id: str) -> bool:
    """Keep only the whisk leveling loop while locking its base and drawer yaw."""
    return str(selection_id) == "VLA82-005"


def drawer_lost_hold_transition(
    selection_id: str, *, target_contact: bool, inside_xy: bool,
) -> str:
    """Do not re-grasp a pizza cutter already supported inside the drawer."""
    if str(selection_id) == "VLA82-038" and bool(target_contact) and bool(inside_xy):
        return "settle"
    return "close"


def pick_place_lost_hold_transition(
    *, drawer_pick_place: bool, selection_id: str,
    target_contact: bool, inside_xy: bool, release_pose_ready: bool = False,
) -> str:
    """Finish a supported placement instead of attempting an in-place re-grasp.

    Once an object is resting inside the requested XY target, opening the
    gripper is the physically controlled completion action.  Re-entering the
    grasp state there only pinches and displaces an already placed object.
    """
    if bool(drawer_pick_place):
        return drawer_lost_hold_transition(
            selection_id, target_contact=target_contact, inside_xy=inside_xy,
        )
    if bool(inside_xy) and (bool(target_contact) or bool(release_pose_ready)):
        return "release"
    return "close"


def drawer_lower_base_recenter_local_direction(
    *, object_center: Sequence[float], release: Sequence[float],
    base_rotation: Sequence[Sequence[float]], magnitude: float = .3,
) -> np.ndarray:
    """Translate the held object toward drawer center through the mobile base."""
    world = np.asarray(release, dtype=float)[:2] - np.asarray(object_center, dtype=float)[:2]
    local = np.asarray(base_rotation, dtype=float)[:2, :2].T @ world
    local /= max(float(np.max(np.abs(local))), 1e-9)
    return np.clip(local * float(magnitude), -float(magnitude), float(magnitude)).astype(np.float32)


def pick_place_base_transport_required(
    *, source_fixture: str, target_fixture: str,
    horizontal_distance: float, threshold: float = .45,
) -> bool:
    """Use the mobile base for a long cabinet-to-counter carry.

    Cabinet extraction is deliberately arm-controlled.  Once the object has
    cleared the door, however, a remote counter can be well outside Panda's
    stationary workspace.  Continuing arm-only motion there eventually loses
    an otherwise verified grasp, so hand the long XY segment to PandaOmron's
    public mobile-base controller.
    """
    return bool(
        str(source_fixture) == "cabinet"
        and str(target_fixture) == "counter"
        and float(horizontal_distance) > float(threshold)
    )


def pick_place_base_transport_threshold(selection_id: str) -> float:
    """Start mobile-base carrying sooner for the fragile cabinet spray bottle."""
    return .20 if str(selection_id) == "VLA82-004" else .45


def pick_place_base_transport_required_for_selection(
    selection_id: str, *, source_fixture: str, target_fixture: str,
    horizontal_distance: float,
) -> bool:
    """Apply the calibrated mobile-base carry to the spray-bottle scene labels."""
    if str(selection_id) == "VLA82-004":
        return False
    return pick_place_base_transport_required(
        source_fixture=source_fixture,
        target_fixture=target_fixture,
        horizontal_distance=horizontal_distance,
        threshold=pick_place_base_transport_threshold(selection_id),
    )


def pick_place_base_transport_latched(
    selection_id: str, *, horizontal_distance: float,
    initial_base_x: float | None, base_x: float,
) -> bool:
    """Keep the spray's base route active until its lateral clearance is undone."""
    return bool(
        str(selection_id) == "VLA82-004"
        and initial_base_x is not None
        and float(horizontal_distance) > .035
        and float(base_x) > float(initial_base_x) + .12
    )


def pick_place_base_transport_magnitude(selection_id: str) -> float:
    """Use a lower mobile-base acceleration while carrying the narrow spray bottle."""
    return .20 if str(selection_id) == "VLA82-004" else .60


def pick_place_base_transport_local_direction(
    *, object_center: Sequence[float], release: Sequence[float],
    base_rotation: Sequence[Sequence[float]], magnitude: float = .6,
) -> np.ndarray:
    """Return a bounded local-base command carrying the object to release XY."""
    world = np.asarray(release, dtype=float)[:2] - np.asarray(object_center, dtype=float)[:2]
    local = np.asarray(base_rotation, dtype=float)[:2, :2].T @ world
    local /= max(float(np.max(np.abs(local))), 1e-9)
    return np.clip(local * float(magnitude), -float(magnitude), float(magnitude)).astype(np.float32)


def solid_counter_release_point(
    *, patches: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    source: Sequence[float], robot_base: Sequence[float],
    object_half_size_xy: Sequence[float], margin: float = .015,
) -> np.ndarray:
    """Choose a reachable point inside one real counter collision slab.

    Sink counters are assembled from several slabs around a cut-out.  The
    center of their union AABB can lie in the empty sink, so placement must be
    projected into an individual slab with enough footprint for the object.
    """
    source_xy = np.asarray(source, dtype=float)[:2]
    base_xy = np.asarray(robot_base, dtype=float)[:2]
    inset = np.asarray(object_half_size_xy, dtype=float)[:2] + float(margin)
    candidates: list[tuple[float, str, np.ndarray]] = []
    for name, low_value, high_value in patches:
        low = np.asarray(low_value, dtype=float)
        high = np.asarray(high_value, dtype=float)
        inner_low = low[:2] + inset
        inner_high = high[:2] - inset
        if np.any(inner_low > inner_high):
            continue
        point_xy = np.clip(source_xy, inner_low, inner_high)
        # Prefer the shortest held-object motion.  Base distance is only a
        # tie-breaker between equally close slabs on opposite sides of a sink.
        score = float(np.linalg.norm(point_xy - source_xy)) + .01 * float(np.linalg.norm(point_xy - base_xy))
        candidates.append((score, str(name), np.array((point_xy[0], point_xy[1], high[2]), dtype=float)))
    if not candidates:
        raise ValueError("counter target has no collision slab large enough for the object footprint")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def counter_collision_support_top_at_point(
    *, patches: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    point_xy: Sequence[float], fallback: float,
) -> float:
    """Return the real collision-slab top directly beneath a release point."""
    point = np.asarray(point_xy, dtype=float)[:2]
    tops = [
        float(np.asarray(high, dtype=float)[2])
        for _, low, high in patches
        if np.all(point >= np.asarray(low, dtype=float)[:2] - 1e-6)
        and np.all(point <= np.asarray(high, dtype=float)[:2] + 1e-6)
    ]
    return max(tops) if tops else float(fallback)


def pick_place_required_lift_height(
    *, source_fixture: str, source_z: float, support_top: float,
    object_half_height: float, nominal: float = .045, clearance: float = .03,
) -> float:
    """Require recessed or elevated-target objects to clear obstacles before XY transit."""
    if str(source_fixture) != "sink":
        return float(nominal)
    clear_height = float(support_top) + float(object_half_height) + float(clearance) - float(source_z)
    return max(float(nominal), clear_height)


def pick_place_lift_proof_height(selection_id: str, required_height: float) -> float:
    """Begin cabinet extraction after a short verified lift below the shelf."""
    selection = str(selection_id)
    if selection == "VLA82-018":
        # The drink carton clears the upper shelf by only a few millimetres;
        # prove it has left the support, then extract horizontally.
        return .01
    if selection == "VLA82-058":
        # The toothbrush bottom must clear the cup rim before horizontal
        # centring; otherwise it catches the near wall and stalls there.
        return .12
    if selection == "VLA82-055":
        # Keep the gripper and full box above the rack's back wall until the
        # horizontal transit has reached the open shelf volume.
        return .12
    return .05 if selection == "VLA82-004" else float(required_height)


def pick_place_cabinet_transport_stage(
    *, base_x: float, initial_base_x: float, object_y: float, release_y: float,
    door_clearance: float = .35, return_clearance: float = .12,
    arm_finish_y_tolerance: float = .35,
) -> str:
    """Route a held cabinet object around the low stack door in four stages."""
    if abs(float(object_y) - float(release_y)) > float(arm_finish_y_tolerance):
        if float(base_x) < float(initial_base_x) + float(door_clearance):
            return "clear_x"
        return "translate_y"
    if float(base_x) > float(initial_base_x) + float(return_clearance):
        return "return_x"
    return "arm_finish"


def pick_place_cabinet_arm_finish_y_tolerance(selection_id: str) -> float:
    """Reserve arm-only finishing for the final short segment of the spray carry."""
    return .05 if str(selection_id) == "VLA82-004" else .35


def drawer_transit_control_stage(
    *, vertical_half_extent: float | None, depth_alignment_error: float | None,
    xy_reached: bool, joint5: float, depth_alignment_latched: bool = False,
    horizontal_latched: bool = False, vertical_threshold: float = .03,
) -> str:
    """Serialize held-tool corrections before allowing drawer descent.

    A live alignment acquired before XY transport is latched so small inertial
    yaw excursions cannot alternate alignment and translation forever.  Once
    the object reaches the drawer, the live angle is checked again before
    wrist relief and descent.
    """
    horizontal = (
        vertical_half_extent is not None
        and float(vertical_half_extent) <= float(vertical_threshold)
    )
    if not horizontal and (bool(xy_reached) or not bool(horizontal_latched)):
        return "level"
    depth_aligned = (
        depth_alignment_error is not None
        and abs(float(depth_alignment_error)) <= np.deg2rad(25.)
    )
    if not depth_aligned and (bool(xy_reached) or not bool(depth_alignment_latched)):
        return "align_depth"
    if not bool(xy_reached):
        return "translate"
    if drawer_wrist_relief_required(joint5=float(joint5)):
        return "relieve_wrist"
    return "ready"


def drawer_wrist_relief_feedback(
    *, joint5: float, baseline_joint5: float, attempts: int,
    direction: float, already_flipped: bool, switch_after: int = 6,
    minimum_improvement: float = .01,
) -> tuple[float, bool]:
    """Reverse a wrist-relief command once when live joint feedback stalls."""
    if (
        not bool(already_flipped)
        and int(attempts) >= int(switch_after)
        and float(joint5) - float(baseline_joint5) < float(minimum_improvement)
    ):
        return -float(direction), True
    return float(direction), bool(already_flipped)


def drawer_lower_pose_correction_required(
    *, vertical_half_extent: float | None, horizontal_latched: bool,
    acquire_threshold: float = .03, release_threshold: float = .055,
) -> bool:
    """Apply hysteresis to the held object's horizontal pose while lowering."""
    if vertical_half_extent is None:
        return True
    threshold = float(release_threshold) if bool(horizontal_latched) else float(acquire_threshold)
    return float(vertical_half_extent) > threshold


def drawer_lower_pose_release_threshold(selection_id: str) -> float:
    """Avoid the controller's measured near-horizontal no-command dead zone."""
    return .04 if str(selection_id) == "VLA82-006" else .055


def drawer_lower_yaw_tolerance(selection_id: str) -> float:
    """Use the measured exposed-slot angular clearance for the rolling pin."""
    selected = str(selection_id)
    degrees = 5.0 if selected == "VLA82-006" else 27.0 if selected == "VLA82-038" else 25.0
    return float(np.deg2rad(degrees))


def drawer_transit_xy_tolerance(selection_id: str) -> float:
    """Require narrow-slot XY convergence before VLA82-006 begins descent."""
    return .012 if str(selection_id) == "VLA82-006" else .035


def drawer_transit_xy_reached(selection_id: str, *, distance: float) -> bool:
    """Use one selection-specific XY threshold throughout the transit state."""
    return float(distance) <= drawer_transit_xy_tolerance(selection_id)


def drawer_vertical_transport_threshold(selection_id: str) -> float:
    """Keep the foil box outside the level-controller deadband at the opening."""
    return .036 if str(selection_id) == "VLA82-045" else .03


def drawer_torso_reserve_required(
    *, selection_id: str, torso_qpos: float, reserve_target: float = .15,
) -> bool:
    """Reserve vertical torso travel for a long tool's constrained descent."""
    return str(selection_id) == "VLA82-006" and float(torso_qpos) < float(reserve_target)


def drawer_lower_torso_command(
    *, selection_id: str, joint5: float, torso_qpos: float,
    reserve_floor: float = .02,
) -> float:
    """Use reserved torso travel when wrist redundancy nears its lower limit."""
    if (
        str(selection_id) == "VLA82-006"
        and drawer_wrist_relief_required(joint5=float(joint5))
        and float(torso_qpos) > float(reserve_floor)
    ):
        return -.5
    return 0.0


def drawer_vertical_descent_torso_command(
    *, object_z: float, release_z: float, torso_qpos: float,
    height_threshold: float = .06, reserve_floor: float = .02,
) -> float:
    """Spend reserved torso travel after arm-only drawer descent saturates."""
    if (
        float(object_z) - float(release_z) > float(height_threshold)
        and float(torso_qpos) > float(reserve_floor)
    ):
        return -.5
    return 0.0


def open_drawer_release_point(
    *, target_low: Sequence[float], target_high: Sequence[float],
    robot_base: Sequence[float], margin: float = .14,
) -> np.ndarray:
    """Choose the open, base-facing part of a drawer volume for vertical entry."""
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    result = (low + high) / 2.0
    horizontal_axis = int(np.argmax(high[:2] - low[:2]))
    candidates = (low[horizontal_axis] + float(margin), high[horizontal_axis] - float(margin))
    base_coordinate = float(np.asarray(robot_base, dtype=float)[horizontal_axis])
    result[horizontal_axis] = min(candidates, key=lambda value: abs(float(value) - base_coordinate))
    return result


def inside_release_height(
    *, target_low_z: float, geom_size: Sequence[float],
    geom_rotation: Sequence[Sequence[float]], clearance: float = .005,
) -> float:
    """Place a rotated object on an inside target's physical lower support."""
    world_half_extent_z = rotated_geom_vertical_half_extent(
        geom_size=geom_size, geom_rotation=geom_rotation,
    )
    return float(target_low_z) + world_half_extent_z + float(clearance)


def cabinet_to_counter_release_point(
    *, target_low: Sequence[float], target_high: Sequence[float],
    source: Sequence[float], robot_base: Sequence[float], margin: float = .08,
    front_offset: float = .20, lateral_offset: float = .22,
) -> np.ndarray:
    """Choose a nearby solid worktop patch instead of a merged counter's centre.

    RoboCasa's counter fixture bounds can span a sink cut-out.  The cabinet
    source Y coordinate identifies the adjacent worktop section; X is biased
    from the robot base toward the counter so the released object remains in
    the arm's normal workspace.
    """
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    source_point = np.asarray(source, dtype=float)
    base = np.asarray(robot_base, dtype=float)
    point = (low + high) / 2.0
    point[0] = np.clip(float(base[0]) - float(front_offset), low[0] + margin, high[0] - margin)
    point[1] = np.clip(float(source_point[1]) + float(lateral_offset), low[1] + margin, high[1] - margin)
    point[2] = high[2]
    return point


def pick_place_counter_release_adjustment(selection_id: str, *, release: Sequence[float]) -> np.ndarray:
    """Return the reachable counter point; failed object-specific offsets stay out of production."""
    return np.asarray(release, dtype=float).copy()


def rotated_geom_vertical_half_extent(
    *, geom_size: Sequence[float], geom_rotation: Sequence[Sequence[float]],
) -> float:
    """Exact world-Z half extent of a rotated box collision geometry."""
    size = np.asarray(geom_size, dtype=float)
    rotation = np.asarray(geom_rotation, dtype=float).reshape(3, 3)
    return float((np.abs(rotation) @ size)[2])


def flat_support_normal_alignment_degrees(geom_rotation: Sequence[Sequence[float]]) -> float:
    """Measure the unsigned tilt between a box's local top normal and world Z."""
    rotation = np.asarray(geom_rotation, dtype=float).reshape(3, 3)
    normal = rotation[:, 2]
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    cosine = float(np.clip(abs(np.dot(normal, np.array((0., 0., 1.)))), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def flat_support_signed_yaw_error_radians(geom_rotation: Sequence[Sequence[float]]) -> float:
    """Return local-X yaw from the shelf X axis, modulo the box's 180° symmetry."""
    rotation = np.asarray(geom_rotation, dtype=float).reshape(3, 3)
    long_axis = rotation[:, 0]
    yaw = float(np.arctan2(long_axis[1], long_axis[0]))
    return float((yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)


def flat_support_yaw_alignment_degrees(geom_rotation: Sequence[Sequence[float]]) -> float:
    """Measure unsigned long-edge yaw from the shelf front edge."""
    return float(np.degrees(abs(flat_support_signed_yaw_error_radians(geom_rotation))))


def shelf_package_rotation_command(
    *, object_rotation: Sequence[Sequence[float]],
    base_rotation: Sequence[Sequence[float]], maximum: float = .40,
) -> tuple[np.ndarray, float, float]:
    """Simultaneously level a box and align its long edge with the shelf."""
    rotation = np.asarray(object_rotation, dtype=float).reshape(3, 3)
    base = np.asarray(base_rotation, dtype=float).reshape(3, 3)
    level_command, level_error = upright_axis_rotation_command(
        object_axis=rotation[:, 2], base_rotation=base,
        tolerance=np.deg2rad(2.), maximum=float(maximum),
    )
    yaw_error = flat_support_signed_yaw_error_radians(rotation)
    yaw_world = np.array((0., 0., -np.sign(yaw_error)), dtype=float)
    yaw_local = base.T @ yaw_world
    yaw_magnitude = 0.0 if abs(yaw_error) <= np.deg2rad(2.) else min(abs(yaw_error) / .35, float(maximum))
    command = level_command + yaw_local * yaw_magnitude
    norm = float(np.linalg.norm(command))
    if norm > float(maximum):
        command *= float(maximum) / norm
    return command, float(np.degrees(level_error)), float(np.degrees(abs(yaw_error)))


def pick_place_physical_geom_names(
    geom_names: Sequence[str], *, contypes: Sequence[int], conaffinities: Sequence[int],
) -> tuple[str, ...]:
    """Keep contact-enabled object geoms out of appearance-only geometry."""
    return tuple(
        str(name)
        for name, contype, conaffinity in zip(geom_names, contypes, conaffinities)
        if int(contype) or int(conaffinity)
    )


def pick_place_counter_release_height(
    *, support_top: float, geom_size: Sequence[float],
    geom_rotation: Sequence[Sequence[float]], clearance: float,
) -> float:
    """Return the physical counter-contact height for the held orientation."""
    return float(support_top) + rotated_geom_vertical_half_extent(
        geom_size=geom_size, geom_rotation=geom_rotation,
    ) + float(clearance)


def drawer_drop_ready(
    *, object_center: Sequence[float], target_low: Sequence[float], target_high: Sequence[float],
    horizontal_margin: float = .03, maximum_drop: float = 0.0,
) -> bool:
    """Permit a real gravity release only over the interior drawer aperture."""
    obj = np.asarray(object_center, dtype=float)
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    inside_xy = bool(np.all(obj[:2] >= low[:2] + float(horizontal_margin)) and np.all(obj[:2] <= high[:2] - float(horizontal_margin)))
    return inside_xy and float(obj[2]) <= float(high[2]) + float(maximum_drop)


def update_stable_release_counter(
    *, previous: Sequence[float] | None, current: Sequence[float],
    target_contact: bool, count: int, tolerance: float = .001,
) -> int:
    """Count consecutive target-contact frames with <=1 mm object motion."""
    if not bool(target_contact) or previous is None:
        return 0
    displacement = float(np.linalg.norm(np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)))
    return int(count) + 1 if displacement <= float(tolerance) else 0


def pick_place_recorded_stability_ready(count: int, required_pairs: int = 3) -> bool:
    """Wait long enough for all stable frame pairs to be stored in the NPZ.

    The observation that increments ``count`` is evaluated before its next
    action is yielded.  One additional stable observation is therefore needed
    to persist ``required_pairs`` complete adjacent pairs in the episode.
    """
    return int(count) >= int(required_pairs) + 1


def pick_place_lower_limit(selection_id: str) -> float:
    """Use the collision-cleared drawer corridor without a needless slow descent."""
    selected = str(selection_id)
    if selected == "VLA82-020":
        return 1.0
    if selected == "VLA82-038":
        return .6
    return .45 if selected == "VLA82-004" else .8 if selected in {"VLA82-005", "VLA82-019", "VLA82-036", "VLA82-058"} else .22


def pick_place_transit_limit(selection_id: str) -> float:
    """Bound fast free-space carrying while preserving approach stability."""
    if str(selection_id) == "VLA82-056":
        return .35
    if str(selection_id) == "VLA82-004":
        return .30
    return .60 if str(selection_id) in {"VLA82-020", "VLA82-036"} else 1.0


def pick_place_uses_direct_transit(selection_id: str) -> bool:
    """Shorten fragile-object carry time without changing grasp force."""
    return str(selection_id) in {"VLA82-004", "VLA82-020", "VLA82-036"}


def pick_place_uses_height_controlled_transit(selection_id: str) -> bool:
    """Keep the held shelf package above the physical back wall during XY travel."""
    return str(selection_id) in {"VLA82-055", "VLA82-056"}


def pick_place_maximum_gravity_drop(selection_id: str) -> float:
    """Bound a real release fall after the complete object clears the drawer aperture."""
    limits = {"VLA82-005": .12, "VLA82-006": .03, "VLA82-041": .05}
    return float(limits.get(str(selection_id), 0.0))


def drawer_gravity_release_ceiling(
    *, floor_top_z: float, vertical_half_extent: float,
    clearance: float = .005, maximum_drop: float = .12,
) -> float:
    """Highest COM for a bounded fall onto the live drawer floor support."""
    return (
        float(floor_top_z) + float(vertical_half_extent)
        + float(clearance) + float(maximum_drop)
    )


def drawer_orientation_correction(
    vertical_half_extent: float | None, threshold: float = .03, direction: float = 1.0,
) -> float:
    """Keep a held long tool horizontal while descending into a shallow drawer."""
    return float(np.sign(direction)) * .6 if vertical_half_extent is not None and float(vertical_half_extent) > float(threshold) else 0.0


def drawer_depth_alignment_error(
    *, long_axis: Sequence[float], target_low: Sequence[float], target_high: Sequence[float],
) -> float:
    """Signed XY error that aligns a held object's long axis with drawer depth."""
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    depth_axis = np.zeros(2, dtype=float)
    depth_axis[int(np.argmax(high[:2] - low[:2]))] = 1.0
    return shortest_axis_alignment_error(np.asarray(long_axis, dtype=float)[:2], depth_axis)


def drawer_release_depth_coordinate(
    *, target_low: Sequence[float], target_high: Sequence[float], robot_base: Sequence[float],
    world_half_extents: Sequence[float], clearance: float = .015,
) -> tuple[int, float]:
    """Place the complete rotated object just inside the base-facing drawer edge."""
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    base = np.asarray(robot_base, dtype=float)
    half_extents = np.asarray(world_half_extents, dtype=float)
    axis = int(np.argmax(high[:2] - low[:2]))
    near_low = abs(float(base[axis]) - float(low[axis])) <= abs(float(base[axis]) - float(high[axis]))
    if near_low:
        coordinate = float(low[axis]) + float(half_extents[axis]) + float(clearance)
    else:
        coordinate = float(high[axis]) - float(half_extents[axis]) - float(clearance)
    return axis, coordinate


def drawer_release_front_bias(
    *, selection_id: str, coordinate: float, robot_coordinate: float,
    magnitude: float = .02, active: bool = True,
) -> float:
    """Center VLA82-006 in the measured narrow exposed drawer slot.

    Two centimetres toward the robot clears the fixed countertop edge while
    retaining physical clearance from the solid moving front panel.
    """
    selected = str(selection_id)
    if not bool(active) or selected not in {
        "VLA82-005", "VLA82-006", "VLA82-014", "VLA82-038", "VLA82-041",
    }:
        return float(coordinate)
    direction = np.sign(float(robot_coordinate) - float(coordinate)) or -1.0
    if selected in {"VLA82-005", "VLA82-014", "VLA82-038", "VLA82-041"}:
        # The horizontal whisk is grasped near its robot-facing end.  Move
        # long tools deeper into the open drawer so the palm remains above
        # the front panel while the object reaches the support floor.
        return float(coordinate) - float(direction) * .07
    return float(coordinate) + float(direction) * float(magnitude)


def drawer_post_release_retreat_target(
    *, eef: Sequence[float], object_center: Sequence[float],
    horizontal_distance: float = .08, lift: float = .06,
) -> np.ndarray:
    """Clear residual one-pad support after the grasp predicate opens."""
    result = np.asarray(eef, dtype=float).copy()
    away = result[:2] - np.asarray(object_center, dtype=float)[:2]
    away /= max(float(np.linalg.norm(away)), 1e-9)
    result[:2] += away * float(horizontal_distance)
    result[2] += float(lift)
    return result


def pick_place_release_retreat_target(
    selection_id: str, *, eef: Sequence[float], object_center: Sequence[float],
) -> np.ndarray:
    """Withdraw from a supported object without toppling a tall water bottle."""
    if str(selection_id) in {"VLA82-017", "VLA82-055"}:
        result = np.asarray(eef, dtype=float).copy()
        result[2] += .12
        return result
    return drawer_post_release_retreat_target(eef=eef, object_center=object_center)


def drawer_post_release_interior_retreat_target(
    *, eef: Sequence[float], object_center: Sequence[float],
    target_low: Sequence[float], target_high: Sequence[float],
    horizontal_distance: float = .08, lift: float = .06,
) -> np.ndarray:
    """Clear residual finger contact toward the drawer interior, never its lip."""
    result = np.asarray(eef, dtype=float).copy()
    center = (
        np.asarray(target_low, dtype=float)[:2]
        + np.asarray(target_high, dtype=float)[:2]
    ) / 2.0
    inward = center - np.asarray(object_center, dtype=float)[:2]
    inward /= max(float(np.linalg.norm(inward)), 1e-9)
    result[:2] += inward * float(horizontal_distance)
    result[2] += float(lift)
    return result


def pick_place_post_release_retreat_required(
    *, raw_grasp: bool, target_contact: bool, exact_gripper_contact: bool,
) -> bool:
    """Clear residual shell contact only after a controlled supported release."""
    return not bool(raw_grasp) and bool(target_contact) and bool(exact_gripper_contact)


def pick_place_latched_retreat_target(
    *, previous: Sequence[float] | None, eef: Sequence[float],
    object_center: Sequence[float], released_supported: bool, selection_id: str = "",
) -> np.ndarray | None:
    """Hold one retreat goal through intermittent shell-contact observations."""
    if previous is not None:
        return np.asarray(previous, dtype=float)
    if not bool(released_supported):
        return None
    return pick_place_release_retreat_target(
        selection_id, eef=eef, object_center=object_center,
    )


def pick_place_retreat_complete(
    selection_id: str, *, eef: Sequence[float], object_center: Sequence[float],
    retreat_target: Sequence[float] | None,
) -> bool:
    """Accept a visibly clear toothbrush wrist even if its servo goal is unreachable."""
    if retreat_target is None:
        return True
    if str(selection_id) in {"VLA82-051", "VLA82-058"}:
        minimum_clearance = .09 if str(selection_id) == "VLA82-051" else .10
        return float(np.linalg.norm(
            np.asarray(eef, dtype=float) - np.asarray(object_center, dtype=float)
        )) >= minimum_clearance
    return float(np.linalg.norm(
        np.asarray(eef, dtype=float) - np.asarray(retreat_target, dtype=float)
    )) <= .02


def pick_place_target_contact_release_ready(
    selection_id: str, *, target_contact: bool, inside_xy: bool,
    eef_distance: float, support_alignment_deg: float | None = None,
    support_yaw_alignment_deg: float | None = None,
) -> bool:
    """Require the toothbrush centre to cross the cup rim before opening."""
    if not bool(target_contact) or float(eef_distance) > .045:
        return False
    if (
        str(selection_id) == "VLA82-055"
        and (
            support_alignment_deg is None
            or float(support_alignment_deg) > 5.0
            or support_yaw_alignment_deg is None
            or float(support_yaw_alignment_deg) > 5.0
        )
    ):
        return False
    return bool(inside_xy) if str(selection_id) == "VLA82-058" else True


def update_release_clear_frames(
    *, raw_grasp: bool, target_contact: bool,
    exact_gripper_contact: bool, count: int,
) -> int:
    clear = not bool(raw_grasp) and bool(target_contact) and not bool(exact_gripper_contact)
    return int(count) + 1 if clear else 0


def release_clearance_ready(count: int, *, required_frames: int = 3) -> bool:
    return int(count) >= int(required_frames)


def drawer_front_insertion_targets(
    *, target_low: Sequence[float], target_high: Sequence[float],
    robot_base: Sequence[float], world_half_extents: Sequence[float],
    floor_top_z: float, front_clearance: float = .02,
    inside_clearance: float = .015, floor_clearance: float = .04,
) -> tuple[np.ndarray, np.ndarray]:
    """Build collision-derived staging and insertion centers for a long tool."""
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    base = np.asarray(robot_base, dtype=float)
    half = np.asarray(world_half_extents, dtype=float)
    axis = int(np.argmax(high[:2] - low[:2]))
    near_low = abs(float(base[axis]) - float(low[axis])) <= abs(float(base[axis]) - float(high[axis]))
    staging = (low + high) / 2.0
    inserted = staging.copy()
    if near_low:
        staging[axis] = low[axis] - half[axis] - float(front_clearance)
    else:
        staging[axis] = high[axis] + half[axis] + float(front_clearance)
    # The target bounds are the live inner-bottom footprint.  A center only
    # one object half-extent beyond the lip is geometrically "inside", but a
    # long tool can still rest on the external door / handle and fall out as
    # soon as the gripper opens.  Put its center over the middle of the real
    # support floor; ``inside_clearance`` remains part of the public helper
    # signature for backwards-compatible callers.
    _ = inside_clearance
    inserted[axis] = (low[axis] + high[axis]) / 2.0
    insertion_z = float(floor_top_z) + float(half[2]) + float(floor_clearance)
    staging[2] = insertion_z
    inserted[2] = insertion_z
    return staging, inserted


def drawer_pitch_rotation_direction(
    *, target_low: Sequence[float], target_high: Sequence[float], robot_base: Sequence[float],
) -> float:
    """Rotate the grasp point to the drawer-interior side of a held long tool."""
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    base = np.asarray(robot_base, dtype=float)
    axis = int(np.argmax(high[:2] - low[:2]))
    near_low = abs(float(base[axis]) - float(low[axis])) <= abs(float(base[axis]) - float(high[axis]))
    # PandaOmron's world-frame X wrist rotation moves its held upper end in
    # -Y for a positive command.  At the low-Y opening, use the inverse so
    # the gripper ends up inside the drawer and link7 clears the front panel.
    return -1.0 if near_low else 1.0


def drawer_transit_world_target(*, eef: Sequence[float], desired: Sequence[float]) -> np.ndarray:
    """Use a direct collision-monitored XY move above an already-open drawer."""
    result = np.asarray(desired, dtype=float).copy()
    result[2] = float(np.asarray(eef, dtype=float)[2])
    return result


def drawer_base_retreat_required(
    *, base: Sequence[float], release: Sequence[float], minimum_distance: float = .65,
) -> bool:
    """Keep the drawer target outside Panda's near-body workspace blind spot."""
    return float(np.linalg.norm(
        np.asarray(base, dtype=float)[:2] - np.asarray(release, dtype=float)[:2]
    )) < float(minimum_distance)


def drawer_base_retreat_distance(selection_id: str) -> float:
    """Choose the shortest collision-cleared base distance for the live tool."""
    # The VLA82-006 reset is already 30 cm from the exposed drawer aperture.
    # A forced 75 cm retreat drives its mobile base into the counter sidewall
    # before the arm can begin the otherwise verified transport.  Keep that
    # collision-free reset distance and use the arm/torso for the final reach.
    return .30 if str(selection_id) == "VLA82-006" else .65


def drawer_base_retreat_local_direction(
    *, base: Sequence[float], release: Sequence[float], base_rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return a unit local-base direction away from the drawer release point."""
    world = np.asarray(base, dtype=float)[:2] - np.asarray(release, dtype=float)[:2]
    world /= max(float(np.linalg.norm(world)), 1e-9)
    local = np.asarray(base_rotation, dtype=float)[:2, :2].T @ world
    local /= max(float(np.linalg.norm(local)), 1e-9)
    return local


def drawer_hold_valid(*, raw_grasp: bool, two_finger_contact: bool) -> bool:
    """Keep a drawer transfer active while both fingers physically oppose the object."""
    return bool(raw_grasp) or bool(two_finger_contact)


def pick_place_transport_hold_valid(
    *, selection_id: str, drawer_pick_place: bool, raw_grasp: bool,
    two_finger_contact: bool,
) -> bool:
    """Accept opposed shell contact for geometries whose pads can briefly unload."""
    shell_contact_allowed = bool(drawer_pick_place) or str(selection_id) in {
        "VLA82-050", "VLA82-053", "VLA82-056",
    }
    return bool(raw_grasp) or shell_contact_allowed and drawer_hold_valid(
        raw_grasp=raw_grasp, two_finger_contact=two_finger_contact,
    )


def update_transport_lost_grasp_frames(*, hold_valid: bool, count: int) -> int:
    """Debounce one-frame native grasp dropouts during an active carry."""
    return 0 if bool(hold_valid) else int(count) + 1


def transport_hold_recovery_required(count: int, *, required_frames: int = 3) -> bool:
    """Recover only after a sustained loss, not one solver-frame dropout."""
    return int(count) >= int(required_frames)


def cabinet_extract_transition(
    *, hold_valid: bool, lost_frames: int, extracted: float,
    threshold: float, next_state: str,
) -> str:
    """Keep a closed extraction through brief grasp-sensor dropouts."""
    if transport_hold_recovery_required(lost_frames):
        return "retry"
    if not bool(hold_valid):
        return "cabinet_extract"
    if float(extracted) >= float(threshold):
        return str(next_state)
    return "cabinet_extract"


def pick_place_strict_trace_errors(
    phases: Sequence[str], controller_trace: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Reject retry-and-drop placement traces even when the final pose is valid."""
    if not set(str(phase) for phase in phases) & {
        "place", "return", "deposit", "insert", "stack", "arrange",
    }:
        return ()
    trace = tuple(frame for frame in controller_trace if "state" in frame)
    states = tuple(str(frame.get("state", "")) for frame in trace)
    release_index = next((index for index, state in enumerate(states) if state == "release"), len(trace))
    grasp_runs = 0
    run_length = 0
    for frame in trace[:release_index]:
        if bool(frame.get("raw_grasp", False)):
            run_length += 1
        else:
            if run_length >= 3:
                grasp_runs += 1
            run_length = 0
    if run_length >= 3:
        grasp_runs += 1
    errors: list[str] = []
    if any(bool(frame.get("exact_gripper_target_contact", False)) for frame in trace):
        errors.append("pick_place:gripper_target_collision_before_release")
    release_alignment = next(
        (
            float(frame["support_normal_alignment_deg"])
            for frame in trace[release_index:]
            if frame.get("state") == "release"
            and frame.get("support_normal_alignment_deg") is not None
        ),
        None,
    )
    release_yaw_alignment = next(
        (
            float(frame["support_yaw_alignment_deg"])
            for frame in trace[release_index:]
            if frame.get("state") == "release"
            and frame.get("support_yaw_alignment_deg") is not None
        ),
        None,
    )
    if (
        (release_alignment is not None and release_alignment > 5.0)
        or (release_yaw_alignment is not None and release_yaw_alignment > 5.0)
    ):
        errors.append("pick_place:crooked_release_pose")
    final_alignment = next(
        (
            float(frame["support_normal_alignment_deg"])
            for frame in reversed(trace)
            if frame.get("support_normal_alignment_deg") is not None
        ),
        None,
    )
    if final_alignment is not None and final_alignment > 5.0:
        errors.append("pick_place:crooked_final_support_pose")
    final_yaw_alignment = next(
        (
            float(frame["support_yaw_alignment_deg"])
            for frame in reversed(trace)
            if frame.get("support_yaw_alignment_deg") is not None
        ),
        None,
    )
    if (
        final_yaw_alignment is not None
        and final_yaw_alignment > 5.0
        and "pick_place:crooked_final_support_pose" not in errors
    ):
        errors.append("pick_place:crooked_final_support_pose")
    if grasp_runs > 1:
        errors.append("pick_place:grasp_reacquired_after_sustained_loss")
    if "release" not in states:
        errors.append("pick_place:controlled_release_stage_not_observed")
    if "settle" not in states:
        errors.append("pick_place:settle_stage_not_observed")
    return tuple(errors)


def drawer_wrist_relief_required(
    *, joint5: float, lower_limit: float = -2.8973, margin: float = .18,
) -> bool:
    """Require roll redundancy before a constrained drawer descent."""
    return float(joint5) < float(lower_limit) + float(margin)


def pick_place_post_lift_state(selection_id: str) -> str:
    """Cabinet sources must clear the fixture before cross-scene transport."""
    selected = str(selection_id)
    if selected == "VLA82-004":
        return "cabinet_extract"
    if selected == "VLA82-005":
        return "orient_for_drawer"
    if selected == "VLA82-055":
        return "level_for_shelf"
    return "transit"


def pick_place_transit_preserves_shelf_orientation(selection_id: str) -> bool:
    """Keep a leveled shelf package aligned while translating it."""
    return str(selection_id) == "VLA82-055"


def pick_place_lift_torso_command(selection_id: str, *, grasped: bool) -> float:
    """Keep the torso fixed while a fragile cabinet pinch establishes lift clearance."""
    del selection_id, grasped
    return 0.0


def pick_place_lift_target(
    *, source: Sequence[float], grasp_source: Sequence[float],
    grasp_offset: Sequence[float], lift_height: float,
) -> np.ndarray:
    """Lift above the live grasp point without dragging back to the spawn XY."""
    source_array = np.asarray(source, dtype=float)
    grasp_array = np.asarray(grasp_source, dtype=float)
    offset = np.asarray(grasp_offset, dtype=float)
    return np.array((
        grasp_array[0] + offset[0],
        grasp_array[1] + offset[1],
        max(float(source_array[2]), float(grasp_array[2])) + offset[2] + float(lift_height),
    ), dtype=float)


def pick_place_extraction_lift_height(selection_id: str, *, nominal: float) -> float:
    """Keep a cabinet extraction above the lift proof but below its shelf."""
    selection = str(selection_id)
    if selection == "VLA82-018":
        return .01
    return .05 if selection == "VLA82-004" else float(nominal)


def cabinet_outward_from_base_rotation(base_rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the cabinet-outward world XY axis for a base facing the fixture."""
    rotation = np.asarray(base_rotation, dtype=float)
    if rotation.shape[0] < 2 or rotation.shape[1] < 1:
        raise ValueError("base rotation must expose its world forward axis")
    outward = -rotation[:2, 0]
    outward /= max(float(np.linalg.norm(outward)), 1e-9)
    return outward


def cabinet_outward_from_positions(
    *, object_xy: Sequence[float], robot_base_xy: Sequence[float],
) -> np.ndarray:
    """Return the live cabinet-outward direction from the object to the robot."""
    outward = np.asarray(robot_base_xy, dtype=float)[:2] - np.asarray(object_xy, dtype=float)[:2]
    norm = float(np.linalg.norm(outward))
    if norm <= 1e-9:
        raise ValueError("cabinet object and robot base cannot share an XY position")
    return outward / norm


def cabinet_extraction_target(
    *, source: Sequence[float], grasp_offset: Sequence[float], outward: Sequence[float],
    lift_height: float, extraction_distance: float = .25,
) -> np.ndarray:
    """Return the held-EFF target that reverses a cabinet front-entry grasp."""
    source_array = np.asarray(source, dtype=float)
    offset_array = np.asarray(grasp_offset, dtype=float)
    outward_array = np.asarray(outward, dtype=float)
    outward_array /= max(float(np.linalg.norm(outward_array)), 1e-9)
    desired = source_array + offset_array + np.array((0., 0., float(lift_height)))
    desired[:2] += outward_array[:2] * float(extraction_distance)
    return desired


def cabinet_extraction_distance(
    *, object_xy: Sequence[float], extraction_start_xy: Sequence[float],
    outward: Sequence[float],
) -> float:
    """Measure extraction from the live successful-grasp baseline."""
    direction = np.asarray(outward, dtype=float)[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return float(np.dot(
        np.asarray(object_xy, dtype=float)[:2] - np.asarray(extraction_start_xy, dtype=float)[:2],
        direction,
    ))


def cabinet_extraction_threshold(selection_id: str) -> float:
    """Use the observed collision-free exit distance for the tall dispenser."""
    return .218 if str(selection_id) == "VLA82-060" else .22


def cabinet_extraction_limit(
    selection_id: str, *, actual_selection_id: str | None = None,
) -> float:
    """Limit cabinet-exit acceleration for the narrow spray-bottle pinch."""
    selected = str(actual_selection_id or selection_id)
    if selected == "VLA82-018":
        return .12
    return .15 if selected == "VLA82-004" else .35


def cabinet_extraction_step_budget(
    selection_id: str, *, actual_selection_id: str | None = None,
) -> int:
    """Allow a low-acceleration cabinet exit to complete its clearance path."""
    selected = str(actual_selection_id or selection_id)
    if selected == "VLA82-017":
        return 96
    if str(selection_id) == "VLA82-004":
        return 160
    return PickPlaceExpert.CABINET_ALIGN_STEPS


def cabinet_extract_next_state(selection_id: str) -> str:
    """Let the narrow spray-bottle grasp settle before changing carry direction."""
    return "cabinet_settle" if str(selection_id) == "VLA82-004" else "transit"


def cabinet_settle_step_requirement(selection_id: str) -> int:
    """Return the brief post-extraction stabilization window for fragile grasps."""
    return 12 if str(selection_id) == "VLA82-004" else 0


def cabinet_bypass_y_target(
    *, door_y_mins: Sequence[float], door_y_maxs: Sequence[float], base_half_y: float,
    current_y: float, clearance: float = .02,
) -> float:
    """Nearest exterior y waypoint clearing every measured door by the base."""
    lows = tuple(float(value) for value in door_y_mins)
    highs = tuple(float(value) for value in door_y_maxs)
    if not lows or not highs or len(lows) != len(highs):
        raise ValueError("blocking-door lower and upper y bounds are required")
    margin = float(base_half_y) + float(clearance)
    intervals = sorted((low - margin, high + margin) for low, high in zip(lows, highs))
    merged: list[list[float]] = []
    for lower, upper in intervals:
        if not merged or lower > merged[-1][1]:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    current = float(current_y)
    containing = next(((lower, upper) for lower, upper in merged if lower <= current <= upper), None)
    if containing is None:
        # The base is already outside every expanded door span.  Returning its
        # current coordinate avoids a gratuitous detour through a remote aisle.
        return current
    southern, northern = containing
    return float(min((southern, northern), key=lambda value: abs(value - current)))


def cabinet_alignment_local_command(
    *, axis: str, eef_xy: Sequence[float], object_xy: Sequence[float], base_rotation: np.ndarray,
    threshold: float = .18,
) -> np.ndarray:
    """Return one measured local-base correction axis, or a stationary command.

    The PandaOmron base frame is rotated approximately 180 degrees in this
    cabinet scene.  This converts the *live* world-frame wrist/object error
    through the reported base rotation rather than assuming a global sign.  A
    single axis is issued at a time so a blocked cabinet-front path cannot hide
    which correction caused a collision.
    """
    axis_index = {"x": 0, "y": 1}.get(str(axis))
    if axis_index is None:
        raise ValueError(f"cabinet alignment axis must be 'x' or 'y', got {axis!r}")
    error = np.asarray(eef_xy, dtype=float)[:2] - np.asarray(object_xy, dtype=float)[:2]
    if abs(float(error[axis_index])) <= float(threshold):
        return np.zeros(2, dtype=np.float32)
    world_direction = np.zeros(2, dtype=float)
    world_direction[axis_index] = -np.sign(error[axis_index])
    local_direction = np.asarray(base_rotation, dtype=float)[:2, :2].T @ world_direction
    # The diagnostic probe used a half-scale pulse.  Normalise after the
    # transform to preserve exactly one commanded local translation component.
    local_direction /= max(float(np.max(np.abs(local_direction))), 1.0)
    return np.clip(local_direction * .5, -.5, .5).astype(np.float32)


def pick_place_torso_command(*, eef_z: float, target_z: float, torso_qpos: float, torso_upper: float = .18) -> float:
    """Raise only while the live wrist remains below its approach target."""
    return .5 if float(eef_z) < float(target_z) - .02 and float(torso_qpos) < float(torso_upper) else 0.0


def pick_place_approach_height(selection_id: str, *, default: float) -> float:
    """Keep a shallow top approach reachable for flat desktop objects."""
    return .07 if str(selection_id) == "VLA82-052" else float(default)


def pick_place_pregrasp_yaw_limit(selection_id: str) -> float:
    """Rotate the thin pizza cutter smoothly before its first translation."""
    return .25 if str(selection_id) == "VLA82-038" else .5


def pick_place_approach_limit(selection_id: str) -> float:
    """Avoid a full-scale XYZ step immediately after pizza-cutter yaw alignment."""
    return .35 if str(selection_id) == "VLA82-038" else .8


def pick_place_lower_torso_command(
    selection_id: str, *, eef_z: float, target_z: float, torso_qpos: float,
) -> float:
    """Raise the mobile torso while a held sink tool clears the low basin."""
    if str(selection_id) != "VLA82-042":
        return 0.0
    return pick_place_torso_command(
        eef_z=eef_z,
        target_z=target_z,
        torso_qpos=torso_qpos,
    )


def pick_place_transit_torso_command(
    selection_id: str, *, eef_z: float, target_z: float, torso_qpos: float,
) -> float:
    """Raise the mobile column while carrying soap to a high cabinet shelf."""
    if str(selection_id) != "VLA82-056":
        return 0.0
    if float(target_z) - float(eef_z) <= .04 or float(torso_qpos) >= .395:
        return 0.0
    return .5


def pick_place_step_budget(selection_id: str) -> int:
    """Allow a bounded extra descent only for the verified near-complete tool path."""
    selected = str(selection_id)
    if selected == "VLA82-041":
        return 1300
    if selected in {"VLA82-004", "VLA82-038", "VLA82-056"}:
        return 900
    if selected in {"VLA82-006", "VLA82-017"}:
        return 700
    return 600


def pick_place_release_clearance(selection_id: str) -> float:
    """Use contact height for a stable tall bottle; retain default drop margin."""
    if str(selection_id) in {"VLA82-016", "VLA82-046", "VLA82-048"}:
        # Both objects now have honest rectangular collision envelopes, so the
        # cover can be lowered directly to exact support contact.
        return 0.0
    return 0.0 if str(selection_id) in {
        "VLA82-004", "VLA82-017", "VLA82-042", "VLA82-050", "VLA82-053", "VLA82-055", "VLA82-058",
    } else .008


def pick_place_release_adjustment(
    selection_id: str, *, release: Sequence[float],
    target_low: Sequence[float], target_high: Sequence[float],
) -> np.ndarray:
    """Move a shelf placement away from both the back wall and front lip."""
    result = np.asarray(release, dtype=float).copy()
    if str(selection_id) == "VLA82-055":
        low = np.asarray(target_low, dtype=float)
        high = np.asarray(target_high, dtype=float)
        result[1] = float((low[1] + high[1]) / 2.0 - .04)
    return result


def pick_place_inside_support_top(
    selection_id: str, *, target_low_z: float, target_high_z: float,
) -> float:
    """Return the support plane rather than the rim for bounded insertion."""
    if str(selection_id) in {"VLA82-051", "VLA82-058"}:
        return float(target_low_z)
    return float(target_high_z)


def pick_place_refreshes_grasp_offset_before_lower(selection_id: str) -> bool:
    """Use the live pad-to-object offset for the toothbrush cup insertion."""
    return str(selection_id) == "VLA82-058"


def pick_place_released_target_transition(
    *, raw_grasp: bool, target_contact: bool, object_center: Sequence[float],
    target_low: Sequence[float], target_high: Sequence[float],
    spatial_relation: str = "inside", release_started: bool = True,
    exact_gripper_contact: bool = False, clearance_ready: bool = True,
) -> str | None:
    """Protect a physically released in-target object from recovery re-grasp."""
    if (
        not bool(release_started) or not bool(clearance_ready) or bool(raw_grasp)
        or bool(exact_gripper_contact) or not bool(target_contact)
    ):
        return None
    point = np.asarray(object_center, dtype=float)
    low = np.asarray(target_low, dtype=float)
    high = np.asarray(target_high, dtype=float)
    if str(spatial_relation) == "on":
        contained = bool(np.all(point[:2] >= low[:2]) and np.all(point[:2] <= high[:2]))
    else:
        contained = bool(np.all(point >= low) and np.all(point <= high))
    return "settle" if contained else None


class PickPlaceExpert:
    """Live, public-action grasp / counter-place controller for VLA82-004."""

    MAX_STEPS = 600
    APPROACH_HEIGHT = .15
    LIFT_HEIGHT = .14
    CLOSE_STEPS = 32
    SECURE_STEPS = 8
    RELEASE_STEPS = 20
    SETTLE_STEPS = 8
    BASE_REACH_STEPS = 96
    CABINET_ALIGN_STEPS = 64
    CABINET_ALIGN_THRESHOLD = .18
    BYPASS_SEGMENT_STEPS = 150
    BYPASS_AXIS_THRESHOLD = .035

    def __init__(self, phases: Sequence[str], contract: Any):
        self.phases = tuple(phases)
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    def actions(self, environment: Any) -> Iterable[tuple[str, np.ndarray]]:
        raw = _raw_environment(environment)
        try:
            robot = raw.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            base_controller = robot.composite_controller.part_controllers["base"]
            eef_site = robot.eef_site_id["right"]
            obj_model = raw.objects["obj"]
            body_name = getattr(obj_model, "root_body", "obj")
            body_id = raw.sim.model.body_name2id(body_name)
            target_id = next(iter(self.contract.target_geometries))
            target = self.contract.target_geometries[target_id]
            object_geoms = tuple(self.contract.object_geom_names["obj"])
            object_geom_ids = tuple(raw.sim.model.geom_name2id(name) for name in object_geoms)
            physical_object_geoms = pick_place_physical_geom_names(
                object_geoms,
                contypes=tuple(raw.sim.model.geom_contype[geom_id] for geom_id in object_geom_ids),
                conaffinities=tuple(raw.sim.model.geom_conaffinity[geom_id] for geom_id in object_geom_ids),
            )
            if not physical_object_geoms:
                physical_object_geoms = object_geoms
            target_object = raw.objects.get(str(target.fixture_id))
            target_prefix = str(getattr(target_object, "naming_prefix", ""))
            target_collision_geoms = tuple(
                name for index in range(int(raw.sim.model.ngeom))
                if (name := raw.sim.model.geom_id2name(index) or "").startswith(target_prefix)
                and "visual" not in name.lower()
                and (int(raw.sim.model.geom_contype[index]) or int(raw.sim.model.geom_conaffinity[index]))
            ) if target_prefix and str(target.fixture_id) == "bathroom_shelf" else ()
            half_height = max(
                float(raw.sim.model.geom_size[raw.sim.model.geom_name2id(name)][2])
                for name in physical_object_geoms
            )
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            for index, phase in enumerate(self.phases):
                for action in PrimitiveExpert(phase, index).actions(environment):
                    yield phase, action
            return
        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        source = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float).copy()
        target_low, target_high = np.asarray(target.min_corner, dtype=float), np.asarray(target.max_corner, dtype=float)
        request = getattr(environment, "request", None)
        selection_id = str(getattr(request, "selection_id", ""))
        if selection_id == "VLA82-051":
            half_height = max(
                rotated_geom_vertical_half_extent(
                    geom_size=np.asarray(
                        raw.sim.model.geom_size[raw.sim.model.geom_name2id(name)], dtype=float,
                    ),
                    geom_rotation=np.asarray(
                        raw.sim.data.geom_xmat[raw.sim.model.geom_name2id(name)], dtype=float,
                    ).reshape(3, 3),
                )
                for name in physical_object_geoms
            )
        drawer_pick_place = is_counter_to_drawer_request(request)
        # RoboCasa drawers have a solid front panel; placement must descend
        # through the exposed top opening rather than attempt to pass through
        # that collision geometry from the front.
        drawer_front_insert = False
        controller_profile_id = pick_place_controller_profile_id(request)
        drawer_floor_top: float | None = None
        drawer_floor_names: tuple[str, ...] = ()
        if drawer_pick_place:
            drawer_floor_names = tuple(
                name for name in self.contract.target_geom_names[target_id]
                if "inner_bottom" in name and "visual" not in name
            )
            if not drawer_floor_names:
                raise ValueError("counter-to-drawer target exposes no inner-bottom collision support")
            floor_highs: list[float] = []
            for name in drawer_floor_names:
                geom_id = raw.sim.model.geom_name2id(name)
                rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                extent = np.abs(rotation) @ np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                floor_highs.append(float(raw.sim.data.geom_xpos[geom_id][2]) + float(extent[2]))
            drawer_floor_top = max(floor_highs)
        release = (target_low + target_high) / 2.0
        release_support_top = pick_place_inside_support_top(
            selection_id,
            target_low_z=float(target_low[2]),
            target_high_z=float(target_high[2]),
        )
        if (
            str(getattr(request, "target_fixture", "")) == "cabinet"
            and str(getattr(request, "target_relation", "")) == "inside"
        ):
            # Cabinet capture supplies the shelf cavity: its lower Z bound is
            # the actual floor and its upper bound is the next shelf.  Placing
            # from the ceiling would release the object above the cabinet.
            release_support_top = float(target_low[2])
        if drawer_pick_place:
            robot_base, _ = base_controller.get_base_pose()
            release = open_drawer_release_point(
                target_low=target_low, target_high=target_high, robot_base=robot_base,
            )
        elif (
            str(getattr(request, "source_fixture", "")) == "sink"
            and str(getattr(request, "target_fixture", "")) == "counter"
        ):
            object_lows: list[np.ndarray] = []
            object_highs: list[np.ndarray] = []
            for name in object_geoms:
                geom_id = raw.sim.model.geom_name2id(name)
                rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                extent = np.abs(rotation) @ np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                center = np.asarray(raw.sim.data.geom_xpos[geom_id], dtype=float)
                object_lows.append(center - extent)
                object_highs.append(center + extent)
            object_low = np.min(np.stack(object_lows), axis=0)
            object_high = np.max(np.stack(object_highs), axis=0)
            object_half_xy = np.maximum(source[:2] - object_low[:2], object_high[:2] - source[:2])
            patches: list[tuple[str, np.ndarray, np.ndarray]] = []
            for name in self.contract.target_geom_names[target_id]:
                geom_id = raw.sim.model.geom_name2id(name)
                if "visual" in name.lower() or not (
                    int(raw.sim.model.geom_contype[geom_id]) or int(raw.sim.model.geom_conaffinity[geom_id])
                ):
                    continue
                rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                extent = np.abs(rotation) @ np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                center = np.asarray(raw.sim.data.geom_xpos[geom_id], dtype=float)
                patches.append((name, center - extent, center + extent))
            robot_base, _ = base_controller.get_base_pose()
            release = solid_counter_release_point(
                patches=patches,
                source=source,
                robot_base=robot_base,
                object_half_size_xy=object_half_xy,
                margin=pick_place_counter_margin(selection_id),
            )
            release_support_top = float(release[2])
        elif (
            str(getattr(request, "source_fixture", "")) == "cabinet"
            and str(getattr(request, "target_fixture", "")) == "counter"
        ):
            robot_base, _ = base_controller.get_base_pose()
            release = cabinet_to_counter_release_point(
                target_low=target_low,
                target_high=target_high,
                source=source,
                robot_base=robot_base,
            )
            release = pick_place_counter_release_adjustment(selection_id, release=release)
        release = pick_place_release_adjustment(
            selection_id, release=release, target_low=target_low, target_high=target_high,
        )
        if str(getattr(request, "target_fixture", "")) == "counter":
            support_patches: list[tuple[str, np.ndarray, np.ndarray]] = []
            for name in self.contract.target_geom_names[target_id]:
                geom_id = raw.sim.model.geom_name2id(name)
                if "visual" in name.lower() or not (
                    int(raw.sim.model.geom_contype[geom_id]) or int(raw.sim.model.geom_conaffinity[geom_id])
                ):
                    continue
                rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                extent = np.abs(rotation) @ np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                center = np.asarray(raw.sim.data.geom_xpos[geom_id], dtype=float)
                support_patches.append((name, center - extent, center + extent))
            release_support_top = counter_collision_support_top_at_point(
                patches=support_patches,
                point_xy=release[:2],
                fallback=release_support_top,
            )
        release[2] = (
            release_support_top
            + half_height
            + pick_place_release_clearance(selection_id)
        )
        required_lift_height = pick_place_required_lift_height(
            source_fixture=str(getattr(request, "source_fixture", "")),
            source_z=float(source[2]),
            support_top=release_support_top,
            object_half_height=half_height,
        )
        lift_command_height = max(float(self.LIFT_HEIGHT), required_lift_height + .025)
        state = pick_place_initial_state_for_request(request)
        if drawer_pick_place:
            state = pick_place_initial_state(controller_profile_id)
        state_steps = 0
        grasp_offset: np.ndarray | None = None
        grasp_source: np.ndarray | None = None
        bypass_route: tuple[float, float, float] | None = None
        previous_settle_obj: np.ndarray | None = None
        stable_release_steps = 0
        drawer_depth_alignment_latched = False
        drawer_horizontal_latched = False
        drawer_leveling_rotation: np.ndarray | None = None
        drawer_wrist_relief_direction = .4
        drawer_wrist_relief_baseline: float | None = None
        drawer_wrist_relief_steps = 0
        drawer_wrist_relief_flipped = False
        cabinet_transport_initial_base_x: float | None = None
        cabinet_target_clearance_latched = False
        cabinet_target_height_latched = False
        counter_lower_base_start_x: float | None = None
        cabinet_extract_start_xy: np.ndarray | None = None
        previous_gripper_aperture: float | None = None
        live_gripper_aperture = 0.0
        transport_lost_grasp_frames = 0
        post_release_retreat_target: np.ndarray | None = None
        drawer_tip_advance_world: np.ndarray | None = None
        drawer_tip_advance_completed = False
        release_clear_frames = 0
        source_drawer_geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if pick_place_grasp_strategy(selection_id) == "drawer_front_side":
            points = np.asarray(raw.drawer.get_bbox_points(), dtype=float)
            drawer_low = np.min(points, axis=0)
            drawer_high = np.max(points, axis=0)
            prefix = str(raw.drawer.naming_prefix)
            slide_joint_id = next(
                index for index in range(int(raw.sim.model.njnt))
                if str(raw.sim.model.joint_id2name(index) or "").startswith(prefix)
                and int(raw.sim.model.jnt_type[index]) == 2
            )
            slide_body_id = int(raw.sim.model.jnt_bodyid[slide_joint_id])
            slide_body_rotation = np.asarray(
                raw.sim.data.body_xmat[slide_body_id], dtype=float,
            ).reshape(3, 3)
            slide_axis_world = slide_body_rotation @ np.asarray(
                raw.sim.model.jnt_axis[slide_joint_id], dtype=float,
            )
            source_drawer_geometry = (drawer_low, drawer_high, slide_axis_world)

        def action_for(
            world_target: np.ndarray, *, closed: bool, limit: float = .8,
            torso_command: float = 0.0, rotation: np.ndarray | None = None,
            gripper_command: float | None = None,
        ) -> np.ndarray:
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            error = np.asarray(controller.world_to_origin_frame(world_target), dtype=float) - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
            action = np.zeros_like(low, dtype=np.float32)
            action[:3] = np.clip(error / .05, -limit, limit)
            if rotation is not None:
                action[3:6] = np.asarray(rotation, dtype=np.float32)
            resolved_gripper_command = (
                float(gripper_command) if gripper_command is not None else (1.0 if closed else -1.0)
            )
            if closed and selection_id in {"VLA82-004", "VLA82-017"}:
                resolved_gripper_command = pick_place_bottle_aperture_command(
                    selection_id,
                    aperture=live_gripper_aperture,
                    previous_aperture=previous_gripper_aperture,
                    grasped=raw_grasp,
                )
            action[layout.gripper] = resolved_gripper_command
            action[layout.torso] = float(torso_command)
            action[layout.base_mode] = -1.0
            return np.clip(action, low, high)

        def base_reach_action(obj: np.ndarray) -> tuple[np.ndarray, float]:
            base_pos, base_rot = base_controller.get_base_pose()
            delta = obj[:2] - np.asarray(base_pos, dtype=float)[:2]
            distance = float(np.linalg.norm(delta))
            action = np.zeros_like(low, dtype=np.float32)
            reach_threshold = pick_place_base_reach_threshold(selection_id)
            if distance > reach_threshold:
                world_step = delta / max(distance, 1e-6) * min(max(distance - reach_threshold, .04), .12)
                local_step = np.asarray(base_rot, dtype=float)[:2, :2].T @ world_step
                action[layout.base[0]:layout.base[0] + 2] = np.clip(local_step / .08, -.5, .5)
                action[layout.base_mode] = 1.0
            else:
                action[layout.base_mode] = -1.0
            return np.clip(action, low, high), distance

        def cabinet_robot_collision() -> bool:
            """Gate the diagnostic base pulses on real cabinet/robot contacts."""
            try:
                for contact_index in range(int(raw.sim.data.ncon)):
                    contact = raw.sim.data.contact[contact_index]
                    names = (
                        str(raw.sim.model.geom_id2name(contact.geom1) or "").lower(),
                        str(raw.sim.model.geom_id2name(contact.geom2) or "").lower(),
                    )
                    has_cabinet = any("cab_" in name or "cabinet" in name for name in names)
                    has_robot = any(
                        "robot" in name or "panda" in name or "finger" in name or "gripper" in name
                        for name in names
                    )
                    if has_cabinet and has_robot:
                        return True
            except (AttributeError, TypeError, IndexError):
                return False
            return False

        def cabinet_alignment_action(axis: str, eef: np.ndarray, obj: np.ndarray) -> np.ndarray:
            _, base_rotation = base_controller.get_base_pose()
            local_xy = cabinet_alignment_local_command(
                axis=axis,
                eef_xy=eef[:2],
                object_xy=obj[:2],
                base_rotation=np.asarray(base_rotation, dtype=float),
                threshold=self.CABINET_ALIGN_THRESHOLD,
            )
            action = np.zeros_like(low, dtype=np.float32)
            action[layout.base[0]:layout.base[0] + 2] = local_xy
            action[layout.gripper] = -1.0
            action[layout.base_mode] = 1.0
            return np.clip(action, low, high)

        def bypass_waypoints(eef: np.ndarray, obj: np.ndarray) -> tuple[float, float, float]:
            """Derive a safe under-door route from live fixture geometry."""
            model, data = raw.sim.model, raw.sim.data

            def bounds(name: str) -> tuple[np.ndarray, np.ndarray]:
                geom_id = model.geom_name2id(name)
                size = np.asarray(model.geom_size[geom_id], dtype=float)
                rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                extent = np.abs(rotation) @ size
                center = np.asarray(data.geom_xpos[geom_id], dtype=float)
                return center - extent, center + extent

            mobile_name = next(
                (str(model.geom_id2name(index) or "") for index in range(int(model.ngeom))
                 if "mobilebase" in str(model.geom_id2name(index) or "").lower() and "pedestal" in str(model.geom_id2name(index) or "").lower()),
                "",
            )
            if not mobile_name:
                raise ValueError("mobile-base collision geometry unavailable")
            mobile_low, mobile_high = bounds(mobile_name)
            corridor_low = min(float(eef[0]), float(obj[0])) - .02
            corridor_high = max(float(eef[0]), float(obj[0])) + .07
            door_lows: list[float] = []
            door_highs: list[float] = []
            for index in range(int(model.ngeom)):
                name = str(model.geom_id2name(index) or "")
                if "stack_" not in name.lower() or "door" not in name.lower():
                    continue
                door_low, door_high = bounds(name)
                # A RoboCasa kitchen has repeated stack fixtures along one
                # wall.  Only doors local to this base's current aisle can
                # block the route; remote copies must not turn this into a
                # multi-room detour.
                if abs(float((door_low[1] + door_high[1]) / 2.0) - float(eef[1])) > 1.0:
                    continue
                if float(door_high[0]) >= corridor_low and float(door_low[0]) <= corridor_high:
                    door_lows.append(float(door_low[1]))
                    door_highs.append(float(door_high[1]))
            if not door_lows or not door_highs:
                raise ValueError("blocking stack-door geometry unavailable")
            exit_y = cabinet_bypass_y_target(
                door_y_mins=door_lows, door_y_maxs=door_highs,
                base_half_y=(float(mobile_high[1]) - float(mobile_low[1])) / 2.0,
                current_y=float(eef[1]),
            )
            # Keep a 10 cm remaining arm correction at the cabinet so the
            # wrist never has to drive the mobile base through the door plane.
            return exit_y, float(obj[0]) + .10, float(obj[1])

        def base_axis_action(axis: str, current: np.ndarray, desired: float) -> np.ndarray:
            axis_index = {"x": 0, "y": 1}[axis]
            _, base_rotation = base_controller.get_base_pose()
            world_direction = np.zeros(2, dtype=float)
            world_direction[axis_index] = np.sign(float(desired) - float(current[axis_index]))
            local_direction = np.asarray(base_rotation, dtype=float)[:2, :2].T @ world_direction
            local_direction /= max(float(np.max(np.abs(local_direction))), 1.0)
            action = np.zeros_like(low, dtype=np.float32)
            # The route is collision-gated every public step.  Use the
            # controller's bounded full-scale translation so the base can
            # actually clear the measured door interval before its cap.
            action[layout.base[0]:layout.base[0] + 2] = np.clip(local_direction, -1.0, 1.0)
            action[layout.gripper] = -1.0
            action[layout.base_mode] = 1.0
            return np.clip(action, low, high)

        def mobile_base_fixture_collision() -> bool:
            """Detect a real moving-base collision with furniture on the route."""
            try:
                for contact_index in range(int(raw.sim.data.ncon)):
                    contact = raw.sim.data.contact[contact_index]
                    names = (
                        str(raw.sim.model.geom_id2name(contact.geom1) or "").lower(),
                        str(raw.sim.model.geom_id2name(contact.geom2) or "").lower(),
                    )
                    if is_material_mobile_base_fixture_contact(
                        names, distance=float(contact.dist),
                    ):
                        return True
            except (AttributeError, TypeError, IndexError):
                return False
            return False

        def drawer_entry_fixture_contact() -> tuple[str, str, float] | None:
            """Return the first material robot contact that blocks drawer entry."""
            robot_tokens = ("robot0_", "mobilebase0_", "gripper0_", "panda")
            fixture_tokens = ("counter_", "stack_", "cabinet", "drawer", "sink", "stove")
            try:
                for contact_index in range(int(raw.sim.data.ncon)):
                    contact = raw.sim.data.contact[contact_index]
                    names = (
                        str(raw.sim.model.geom_id2name(int(contact.geom1)) or "").lower(),
                        str(raw.sim.model.geom_id2name(int(contact.geom2)) or "").lower(),
                    )
                    first_robot = any(token in names[0] for token in robot_tokens)
                    second_robot = any(token in names[1] for token in robot_tokens)
                    first_fixture = any(token in names[0] for token in fixture_tokens)
                    second_fixture = any(token in names[1] for token in fixture_tokens)
                    if (
                        ((first_robot and second_fixture) or (second_robot and first_fixture))
                        and float(contact.dist) < -1e-4
                    ):
                        return names[0], names[1], float(contact.dist)
            except (AttributeError, TypeError, IndexError):
                return None
            return None

        for step in range(1, pick_place_step_budget(selection_id) + 1):
            base_distance: float | None = None
            alignment_axis: str | None = None
            alignment_error: float | None = None
            eef = np.asarray(raw.sim.data.site_xpos[eef_site], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
            raw_grasp = bool(raw._check_grasp(robot.gripper["right"], obj_model))
            exact_contact = _selected_geom_has_gripper_contact(raw, object_geoms)
            two_finger_contact = _selected_geom_has_two_finger_contacts(raw, object_geoms)
            two_pad_contact = selected_geom_has_two_finger_pad_contacts(raw, object_geoms)
            selected_contact_details = selected_gripper_contact_details(raw, object_geoms)
            target_contact_details = selected_gripper_contact_details(raw, target_collision_geoms)
            support_normal_alignment_deg = None
            support_yaw_alignment_deg = None
            if selection_id == "VLA82-055":
                support_geom_id = raw.sim.model.geom_name2id(physical_object_geoms[0])
                support_rotation = np.asarray(
                    raw.sim.data.geom_xmat[support_geom_id], dtype=float,
                ).reshape(3, 3)
                support_normal_alignment_deg = flat_support_normal_alignment_degrees(support_rotation)
                support_yaw_alignment_deg = flat_support_yaw_alignment_degrees(support_rotation)
            pad_center = selected_gripper_pad_center(raw)
            named_pad_positions = {
                str(raw.sim.model.geom_id2name(index) or ""): np.asarray(
                    raw.sim.data.geom_xpos[index], dtype=float,
                )
                for index in range(int(raw.sim.model.ngeom))
                if is_gripper_pad_geometry_name(str(raw.sim.model.geom_id2name(index) or ""))
            }
            snapshot = snapshot_from_environment(environment, step, contract=self.contract)
            gripper_values = np.asarray(snapshot.gripper_qpos, dtype=float).reshape(-1)
            live_gripper_aperture = (
                float(np.max(gripper_values) - np.min(gripper_values))
                if gripper_values.size >= 2 else 0.0
            )
            target_contact = any(contact.object_id == "obj" and contact.target_id == target_id for contact in snapshot.contact_evidence)
            inner_bottom_contact = bool(
                drawer_pick_place
                and selected_geoms_have_contact(raw, object_geoms, drawer_floor_names)
            )
            inside_xy = bool(
                np.all(obj[:2] >= target_low[:2])
                and np.all(obj[:2] <= target_high[:2])
            )
            torso_qpos = float(robot.composite_controller.part_controllers["torso"].joint_pos[0])
            torso_command = 0.0
            yaw_alignment_error: float | None = None
            side_entry_world: np.ndarray | None = None
            tool_axis_world: np.ndarray | None = None
            target_tool_axis_world: np.ndarray | None = None
            tool_alignment_error: float | None = None
            opening_axis_world: np.ndarray | None = None
            target_opening_axis_world: np.ndarray | None = None
            opening_alignment_error: float | None = None
            drawer_depth_error: float | None = None
            drawer_long_axis: np.ndarray | None = None
            object_world_half_extents: np.ndarray | None = None
            object_vertical_half_extent: float | None = None
            gravity_release_ready = False
            drawer_front_stage_world: np.ndarray | None = None
            drawer_front_insert_world: np.ndarray | None = None
            drawer_entry_collision = False
            if drawer_pick_place:
                release_geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                release_rotation = np.asarray(raw.sim.data.geom_xmat[release_geom_id], dtype=float).reshape(3, 3)
                release_size = np.asarray(raw.sim.model.geom_size[release_geom_id], dtype=float)
                object_vertical_half_extent = rotated_geom_vertical_half_extent(
                    geom_size=release_size, geom_rotation=release_rotation,
                )
                drawer_long_axis = release_rotation[:, int(np.argmax(release_size))]
                object_world_half_extents = np.abs(release_rotation) @ release_size
                drawer_depth_error = drawer_depth_alignment_error(
                    long_axis=drawer_long_axis,
                    target_low=target_low, target_high=target_high,
                )
                release[2] = inside_release_height(
                    target_low_z=float(drawer_floor_top if drawer_floor_top is not None else target_low[2]),
                    geom_size=release_size,
                    geom_rotation=release_rotation,
                )
                live_robot_base = np.asarray(base_controller.get_base_pose()[0], dtype=float)
                depth_axis, depth_coordinate = drawer_release_depth_coordinate(
                    target_low=target_low, target_high=target_high,
                    robot_base=live_robot_base,
                    world_half_extents=object_world_half_extents,
                )
                release[depth_axis] = drawer_release_front_bias(
                    selection_id=selection_id, coordinate=depth_coordinate,
                    robot_coordinate=float(live_robot_base[depth_axis]),
                    active=state in {"transit", "lower", "release", "settle"},
                )
            elif (
                selection_id == "VLA82-004"
                and str(getattr(request, "target_fixture", "")) == "counter"
            ):
                release_geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                release_rotation = np.asarray(
                    raw.sim.data.geom_xmat[release_geom_id], dtype=float,
                ).reshape(3, 3)
                release_size = np.asarray(raw.sim.model.geom_size[release_geom_id], dtype=float)
                release[2] = pick_place_counter_release_height(
                    support_top=release_support_top,
                    geom_size=release_size,
                    geom_rotation=release_rotation,
                    clearance=pick_place_release_clearance(selection_id),
                )
            released_target_state = pick_place_released_target_transition(
                raw_grasp=raw_grasp,
                target_contact=target_contact,
                object_center=obj,
                target_low=target_low,
                target_high=target_high,
                spatial_relation=target.spatial_relation,
                release_started=state in {"release", "settle"},
                exact_gripper_contact=exact_contact,
                clearance_ready=(state == "settle"),
            )
            if state != "settle" and released_target_state == "settle":
                previous_settle_obj = obj.copy()
                stable_release_steps = 0
                state, state_steps = "settle", 0
            existing_grasp_state = pick_place_existing_grasp_transition(
                state=state,
                raw_grasp=raw_grasp,
                two_pad_contact=two_pad_contact,
            )
            if existing_grasp_state != state:
                grasp_offset = eef - obj
                grasp_source = obj.copy()
                state, state_steps = existing_grasp_state, 0
            upright_axis_error: float | None = None
            phase = "place" if state in {
                "orient_for_drawer", "align_drawer_depth_yaw", "relieve_drawer_wrist",
                "retreat_base_for_drawer",
                "cabinet_settle", "transit", "transit_front", "lower_front", "insert_drawer",
                "tilt_drawer_entry", "lower_tilted_drawer_entry", "advance_tilted_drawer_entry",
                "upright_for_counter", "counter_lower_base_reposition", "upright_recenter_for_counter",
                "lower", "release", "settle",
            } else "grasp"
            if should_lock_vla004_wrist_for_close(
                selection_id, state=state, any_gripper_contact=exact_contact,
            ):
                action = action_for(eef.copy(), closed=True, limit=.10)
            elif state == "align_yaw":
                geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                object_rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                object_size = np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                narrow = object_rotation[:2, int(np.argmin(object_size[:2]))]
                opening = gripper_opening_axis_from_named_pads(named_pad_positions)
                yaw_alignment_error = shortest_axis_alignment_error(opening, narrow)
                action = action_for(
                    eef, closed=False,
                    rotation=np.array((0., 0., np.clip(
                        yaw_alignment_error / .35,
                        -pick_place_pregrasp_yaw_limit(selection_id),
                        pick_place_pregrasp_yaw_limit(selection_id),
                    ))),
                )
                state_steps += 1
                if abs(yaw_alignment_error) <= np.deg2rad(5.) or state_steps >= 28:
                    state, state_steps = pick_place_post_yaw_alignment_state(selection_id), 0
            elif pick_place_uses_pad_center_alignment(selection_id) and pad_center is not None and state in {"cabinet_front_stage", "cabinet_front_insert", "approach", "descend", "close", "side_orient", "side_preapproach", "side_axis_align", "side_descend", "side_center", "drawer_side_orient", "drawer_front_stage", "drawer_front_insert"}:
                # The bottle is almost as wide as the Panda opening.  Align
                # its actual native grasp pads (not the wrist origin / outer
                # finger shells) before closing, otherwise an apparent two-
                # finger contact cannot become a physical hold.
                grasp_geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                grasp_object_rotation = np.asarray(
                    raw.sim.data.geom_xmat[grasp_geom_id], dtype=float,
                ).reshape(3, 3)
                aligned_obj = pick_place_grasp_target(
                    selection_id,
                    object_center=obj,
                    object_rotation=grasp_object_rotation,
                ) + pick_place_pad_alignment_offset(selection_id)
                if pick_place_uses_high_grasp_target(selection_id):
                    aligned_obj = pick_place_high_grasp_target(
                        selection_id,
                        object_center=obj,
                        object_rotation=grasp_object_rotation,
                        object_half_length=float(np.max(
                            np.asarray(raw.sim.model.geom_size[grasp_geom_id], dtype=float),
                        )),
                    ) + pick_place_pad_alignment_offset(selection_id)
                if state in {"side_orient", "side_preapproach", "side_axis_align", "side_descend", "side_center", "drawer_side_orient", "drawer_front_stage", "drawer_front_insert"}:
                    geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                    geom_rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                    geom_size = np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                    if pick_place_grasp_strategy(selection_id) == "drawer_front_side":
                        if source_drawer_geometry is None:
                            raise EnvironmentValidationError("drawer-front strategy has no live drawer geometry")
                        live_base, _ = base_controller.get_base_pose()
                        drawer_front_stage_world, drawer_front_insert_world, aligned_obj = drawer_front_side_waypoints(
                            object_center=obj,
                            robot_base=live_base,
                            drawer_low=source_drawer_geometry[0],
                            drawer_high=source_drawer_geometry[1],
                            slide_axis_world=source_drawer_geometry[2],
                            geom_rotation=geom_rotation,
                            geom_size=geom_size,
                        )
                        over_door_height = drawer_over_door_height(
                            selection_id,
                            drawer_high_z=float(source_drawer_geometry[1][2]),
                            nominal_height=float(drawer_front_stage_world[2]),
                        )
                        drawer_front_stage_world[2] = over_door_height
                        drawer_front_insert_world[2] = over_door_height
                        side_entry = drawer_front_insert_world
                    else:
                        live_base, _ = base_controller.get_base_pose()
                        entry_reference = pick_place_side_entry_reference(
                            selection_id, eef=eef, base=live_base,
                        )
                        side_entry = pick_place_side_entry_point(
                            object_center=obj, eef=entry_reference,
                            geom_rotation=geom_rotation, geom_size=geom_size,
                        )
                    side_entry_world = side_entry.copy()
                    tool_axis_world = np.asarray(pad_center, dtype=float) - eef
                    outward = side_entry - obj
                    orientation_entry = side_entry
                    if pick_place_grasp_strategy(selection_id) == "drawer_front_side":
                        target_tool_axis_world = drawer_front_target_tool_axis(
                            object_center=obj, inserted_side_point=side_entry,
                        )
                        orientation_entry = side_entry.copy()
                        orientation_entry[2] = obj[2]
                    else:
                        target_tool_axis_world = -outward / max(float(np.linalg.norm(outward)), 1e-9)
                    target_opening_axis_world = horizontal_side_target_opening_axis(
                        geom_rotation=geom_rotation, geom_size=geom_size,
                    )
                    opening_axis_world = gripper_opening_axis_world_from_named_pads(named_pad_positions)
                    _, base_rotation = base_controller.get_base_pose()
                    if state == "drawer_side_orient":
                        rotation_command, tool_alignment_error = horizontal_side_grasp_tool_command(
                            tool_axis_world=tool_axis_world,
                            object_center=obj,
                            side_entry=orientation_entry,
                            base_rotation=base_rotation,
                        )
                        roll_command, opening_alignment_error = horizontal_side_grasp_roll_command(
                            opening_axis_world=opening_axis_world,
                            target_opening_axis_world=target_opening_axis_world,
                            tool_axis_world=tool_axis_world,
                            base_rotation=base_rotation,
                        )
                        if tool_alignment_error <= np.deg2rad(6.):
                            rotation_command = roll_command
                        action = action_for(eef, closed=False, rotation=rotation_command)
                        state_steps += 1
                        next_state = drawer_front_orientation_transition(
                            tool_alignment_error, opening_alignment_error,
                        )
                        if next_state != state:
                            state, state_steps = next_state, 0
                        elif state_steps >= 96:
                            self.trace.append({
                                "step": step, "state": state,
                                "event": "drawer_front_orientation_cap_exhausted",
                                "tool_alignment_error_rad": tool_alignment_error,
                                "opening_alignment_error_rad": opening_alignment_error,
                            })
                            return
                    elif state in {"drawer_front_stage", "drawer_front_insert"}:
                        contact = drawer_entry_fixture_contact()
                        drawer_entry_collision = contact is not None
                        target_pad = (
                            drawer_front_stage_world
                            if state == "drawer_front_stage"
                            else drawer_front_insert_world
                        )
                        desired = pick_place_pad_center_target(
                            eef=eef, pad_center=pad_center, obj=target_pad,
                        )
                        distance = float(np.linalg.norm(desired - eef))
                        next_state = drawer_front_motion_transition(
                            state, distance, drawer_entry_collision,
                        )
                        if next_state == "failed":
                            self.trace.append({
                                "step": step, "state": state,
                                "event": "drawer_front_collision_abort",
                                "contact": contact,
                                "distance": distance,
                            })
                            return
                        action = action_for(desired, closed=False, limit=.30)
                        state_steps += 1
                        if next_state != state:
                            state, state_steps = next_state, 0
                        elif state_steps >= drawer_front_waypoint_step_cap(selection_id):
                            self.trace.append({
                                "step": step, "state": state,
                                "event": "drawer_front_waypoint_cap_exhausted",
                                "distance": distance,
                            })
                            return
                    elif state == "side_orient":
                        rotation_command, tool_alignment_error = horizontal_side_grasp_tool_command(
                            tool_axis_world=tool_axis_world,
                            object_center=obj,
                            side_entry=side_entry,
                            base_rotation=base_rotation,
                        )
                        action = action_for(eef, closed=False, rotation=rotation_command)
                        state_steps += 1
                        if tool_alignment_error <= np.deg2rad(6.):
                            state, state_steps = pick_place_post_side_orient_state(selection_id), 0
                    elif state == "side_preapproach":
                        desired_object = side_entry + np.array((0., 0., self.APPROACH_HEIGHT))
                        desired = pick_place_pad_center_target(eef=eef, pad_center=pad_center, obj=desired_object)
                        action = action_for(desired, closed=False, limit=.40)
                        if float(np.linalg.norm(desired - eef)) <= pick_place_side_preapproach_tolerance(selection_id):
                            state, state_steps = (
                                "side_axis_align"
                                if pick_place_grasp_strategy(selection_id) == "horizontal_side"
                                else "side_descend",
                                0,
                            )
                    elif state == "side_axis_align":
                        rotation_command, opening_alignment_error = horizontal_side_grasp_roll_command(
                            opening_axis_world=opening_axis_world,
                            target_opening_axis_world=target_opening_axis_world,
                            tool_axis_world=tool_axis_world,
                            base_rotation=base_rotation,
                        )
                        action = action_for(eef, closed=False, rotation=rotation_command)
                        state_steps += 1
                        if opening_alignment_error <= np.deg2rad(6.):
                            state, state_steps = "side_descend", 0
                    elif state == "side_descend":
                        desired = pick_place_pad_center_target(eef=eef, pad_center=pad_center, obj=side_entry)
                        action = action_for(desired, closed=False, limit=.32)
                        if float(np.linalg.norm(desired - eef)) <= .035:
                            state, state_steps = "side_center", 0
                    else:
                        desired = pick_place_pad_center_target(eef=eef, pad_center=pad_center, obj=obj)
                        action = action_for(desired, closed=False, limit=.20)
                        if pick_place_side_center_ready(
                            selection_id,
                            distance=float(np.linalg.norm(desired - eef)),
                            exact_contact=exact_contact,
                            two_pad_contact=two_pad_contact,
                        ):
                            state, state_steps = "close", 0
                elif state == "cabinet_front_stage":
                    _, base_rotation = base_controller.get_base_pose()
                    desired_object = aligned_obj.copy()
                    desired_object[:2] += cabinet_outward_from_base_rotation(base_rotation) * .18
                    desired_object[2] += .015
                    desired = pick_place_pad_center_target(eef=eef, pad_center=pad_center, obj=desired_object)
                    action = action_for(desired, closed=False, limit=.45)
                    state_steps += 1
                    if float(np.linalg.norm(desired - eef)) <= .04:
                        state, state_steps = pick_place_post_cabinet_stage_state(selection_id), 0
                elif state == "cabinet_front_insert":
                    desired = pick_place_pad_center_target(eef=eef, pad_center=pad_center, obj=aligned_obj)
                    action = action_for(desired, closed=False, limit=.40)
                    state_steps += 1
                    if float(np.linalg.norm(desired - eef)) <= .035:
                        state, state_steps = "close", 0
                elif state == "approach":
                    desired = pick_place_pad_center_target(
                        eef=eef, pad_center=pad_center, obj=aligned_obj + np.array((
                            0., 0., pick_place_approach_height(
                                selection_id, default=self.APPROACH_HEIGHT,
                            ),
                        )),
                    )
                    action = action_for(
                        desired, closed=False,
                        limit=pick_place_approach_limit(selection_id),
                        torso_command=pick_place_torso_command(
                            eef_z=float(eef[2]), target_z=float(desired[2]), torso_qpos=torso_qpos,
                        ),
                    )
                    if float(np.linalg.norm(desired - eef)) <= .04:
                        state, state_steps = "descend", 0
                elif state == "descend":
                    desired = pick_place_pad_center_target(eef=eef, pad_center=pad_center, obj=aligned_obj)
                    action = action_for(desired, closed=False, limit=.40)
                    state = pick_place_descent_transition(
                        selection_id=selection_id,
                        distance=float(np.linalg.norm(desired - eef)), any_gripper_contact=exact_contact,
                    )
                    if state == "close":
                        state_steps = 0
                else:
                    tracking_target = pick_place_pad_center_target(eef=eef, pad_center=pad_center, obj=aligned_obj)
                    desired = pick_place_close_target(selection_id, eef=eef, tracking_target=tracking_target)
                    action = action_for(desired, closed=True, limit=.25)
                    state_steps += 1
                    if pick_place_close_transition_for_selection(
                        selection_id,
                        raw_grasp=raw_grasp, two_pad_contact=two_pad_contact,
                        broad_two_finger_contact=two_finger_contact,
                    ) == "secure":
                        grasp_offset = eef - obj
                        grasp_source = obj.copy()
                        state, state_steps = "secure", 0
                    elif state_steps >= self.CLOSE_STEPS:
                        state, state_steps = pick_place_retry_grasp_state(selection_id), 0
            elif state in {"bypass_y_exit", "bypass_x_cross", "bypass_y_return"}:
                try:
                    if bypass_route is None:
                        bypass_route = bypass_waypoints(eef, obj)
                    exit_y, cross_x, return_y = bypass_route
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                    self.trace.append({"step": step, "state": state, "event": "bypass_geometry_unavailable", "error": str(error)})
                    return
                axis, desired = (
                    ("y", exit_y) if state == "bypass_y_exit" else
                    ("x", cross_x) if state == "bypass_x_cross" else
                    ("y", return_y)
                )
                alignment_axis, alignment_error = axis, abs(float(eef[0 if axis == "x" else 1]) - float(desired))
                if mobile_base_fixture_collision():
                    self.trace.append({
                        "step": step, "state": state, "event": "bypass_collision_abort",
                        "axis": axis, "axis_error": alignment_error,
                    })
                    return
                action = base_axis_action(axis, eef, desired)
                state_steps += 1
                if alignment_error <= self.BYPASS_AXIS_THRESHOLD:
                    state, state_steps = (
                        ("bypass_x_cross", 0) if state == "bypass_y_exit" else
                        ("bypass_y_return", 0) if state == "bypass_x_cross" else
                        (pick_place_post_bypass_state(selection_id), 0)
                    )
                elif state_steps >= self.BYPASS_SEGMENT_STEPS:
                    self.trace.append({
                        "step": step, "state": state, "event": "bypass_segment_cap_exhausted",
                        "axis": axis, "axis_error": alignment_error,
                    })
                    return
            elif state in {"cabinet_x_align", "cabinet_y_align"}:
                alignment_axis = "x" if state == "cabinet_x_align" else "y"
                alignment_error = abs(float(eef[0 if alignment_axis == "x" else 1] - obj[0 if alignment_axis == "x" else 1]))
                if cabinet_robot_collision():
                    self.trace.append({
                        "step": step, "state": state, "event": "cabinet_alignment_collision_abort",
                        "axis": alignment_axis, "axis_error": alignment_error,
                    })
                    return
                action = cabinet_alignment_action(alignment_axis, eef, obj)
                state_steps += 1
                if alignment_error <= self.CABINET_ALIGN_THRESHOLD:
                    state, state_steps = ("cabinet_y_align", 0) if alignment_axis == "x" else ("approach", 0)
                elif state_steps >= self.CABINET_ALIGN_STEPS:
                    self.trace.append({
                        "step": step, "state": state, "event": "cabinet_alignment_cap_exhausted",
                        "axis": alignment_axis, "axis_error": alignment_error,
                    })
                    return
            elif state == "reach_base":
                action, base_distance = base_reach_action(obj)
                state_steps += 1
                next_state = pick_place_base_reach_state(
                    distance=base_distance, steps=state_steps, maximum_steps=pick_place_base_reach_step_budget(selection_id),
                    threshold=pick_place_base_reach_threshold(selection_id),
                )
                if next_state == "failed":
                    self.trace.append({"step": step, "state": state, "event": "base_reach_cap_exhausted", "base_distance": base_distance})
                    return
                if next_state == "approach":
                    state, state_steps = pick_place_post_base_reach_state(selection_id), 0
            elif state in {"cabinet_front_stage", "cabinet_front_insert"}:
                _, base_rotation = base_controller.get_base_pose()
                outward = cabinet_outward_from_base_rotation(base_rotation)
                desired = obj.copy()
                if state == "cabinet_front_stage":
                    desired[:2] += outward * .18
                    desired[2] += .015
                torso_command = pick_place_torso_command(
                    eef_z=float(eef[2]), target_z=float(desired[2]), torso_qpos=torso_qpos,
                )
                action = action_for(desired, closed=False, limit=.45, torso_command=torso_command)
                state_steps += 1
                if float(np.linalg.norm(desired - eef)) <= .04:
                    state, state_steps = ("cabinet_front_insert", 0) if state == "cabinet_front_stage" else ("close", 0)
                elif state_steps >= self.CABINET_ALIGN_STEPS:
                    self.trace.append({
                        "step": step, "state": state, "event": "cabinet_front_approach_cap_exhausted",
                        "distance": float(np.linalg.norm(desired - eef)),
                    })
                    return
            elif state == "approach":
                desired = obj + np.array((0., 0., self.APPROACH_HEIGHT))
                torso_command = pick_place_torso_command(eef_z=float(eef[2]), target_z=float(desired[2]), torso_qpos=torso_qpos)
                action = action_for(
                    desired, closed=False,
                    limit=pick_place_approach_limit(selection_id),
                    torso_command=torso_command,
                )
                if float(np.linalg.norm(desired - eef)) <= .04:
                    state, state_steps = "descend", 0
            elif state == "descend":
                action = action_for(obj, closed=False, limit=.45)
                if pick_place_descent_reached(
                    distance=float(np.linalg.norm(obj - eef)), any_gripper_contact=exact_contact,
                ):
                    state, state_steps = "close", 0
            elif state == "close":
                close_target = pick_place_close_world_target(
                    controller_profile_id, eef=eef, obj=obj, any_gripper_contact=exact_contact,
                )
                action = action_for(close_target, closed=True, limit=.35)
                state_steps += 1
                if pick_place_close_transition(
                    raw_grasp=raw_grasp, two_pad_contact=two_pad_contact,
                    broad_two_finger_contact=two_finger_contact,
                ) == "secure":
                    grasp_offset = eef - obj
                    grasp_source = obj.copy()
                    state, state_steps = "secure", 0
                elif state_steps >= self.CLOSE_STEPS:
                    state, state_steps = "approach", 0
            elif state == "secure":
                secure_target = pick_place_secure_hold_target(
                    selection_id,
                    eef=eef,
                    obj=obj,
                    grasp_offset=grasp_offset,
                )
                action = action_for(
                    secure_target, closed=True, limit=.10,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                secure_next = pick_place_secure_transition(
                    selection_id, raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                    secure_steps=state_steps,
                )
                if secure_next == "lift":
                    state, state_steps = "lift", 0
                    desired = pick_place_lift_target(
                        source=source,
                        grasp_source=grasp_source if grasp_source is not None else obj,
                        grasp_offset=grasp_offset if grasp_offset is not None else np.zeros(3),
                        lift_height=lift_command_height,
                    )
                    action = action_for(
                        desired, closed=True, limit=pick_place_initial_lift_limit(selection_id),
                        torso_command=pick_place_lift_torso_command(controller_profile_id, grasped=True),
                        gripper_command=pick_place_gripper_hold_command(selection_id),
                    )
                elif secure_next == "close":
                    state, state_steps = "close", 0
            elif state == "lift":
                desired = pick_place_lift_target(
                    source=source,
                    grasp_source=grasp_source if grasp_source is not None else obj,
                    grasp_offset=grasp_offset if grasp_offset is not None else np.zeros(3),
                    lift_height=lift_command_height,
                )
                action = action_for(
                    desired, closed=True, limit=pick_place_lift_limit(selection_id, state_steps),
                    torso_command=pick_place_lift_torso_command(controller_profile_id, grasped=raw_grasp),
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                if not pick_place_transport_hold_valid(
                    selection_id=selection_id,
                    drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                ):
                    state, state_steps = "close", 0
                elif float(obj[2] - source[2]) >= pick_place_lift_proof_height(
                    selection_id, required_lift_height,
                ):
                    state, state_steps = pick_place_post_lift_state(controller_profile_id), 0
            elif state == "cabinet_extract":
                robot_base, _ = base_controller.get_base_pose()
                outward = cabinet_outward_from_positions(
                    object_xy=obj[:2], robot_base_xy=np.asarray(robot_base, dtype=float)[:2],
                )
                if cabinet_extract_start_xy is None:
                    cabinet_extract_start_xy = obj[:2].copy()
                desired = cabinet_extraction_target(
                    source=grasp_source if grasp_source is not None else source,
                    grasp_offset=grasp_offset if grasp_offset is not None else np.zeros(3),
                    outward=outward,
                    lift_height=pick_place_extraction_lift_height(selection_id, nominal=self.LIFT_HEIGHT),
                )
                action = action_for(
                    desired, closed=True,
                    limit=cabinet_extraction_limit(
                        controller_profile_id, actual_selection_id=selection_id,
                    ),
                )
                state_steps += 1
                extracted = cabinet_extraction_distance(
                    object_xy=obj[:2],
                    extraction_start_xy=cabinet_extract_start_xy,
                    outward=outward,
                )
                transport_hold_valid = pick_place_transport_hold_valid(
                    selection_id=selection_id,
                    drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                )
                transport_lost_grasp_frames = update_transport_lost_grasp_frames(
                    hold_valid=transport_hold_valid,
                    count=transport_lost_grasp_frames,
                )
                extraction_next = cabinet_extract_transition(
                    hold_valid=transport_hold_valid,
                    lost_frames=transport_lost_grasp_frames,
                    extracted=extracted,
                    threshold=cabinet_extraction_threshold(selection_id),
                    next_state=cabinet_extract_next_state(controller_profile_id),
                )
                if extraction_next == "retry":
                    cabinet_extract_start_xy = None
                    state, state_steps = "cabinet_front_stage", 0
                elif extraction_next != "cabinet_extract":
                    state, state_steps = extraction_next, 0
                elif state_steps >= cabinet_extraction_step_budget(
                    controller_profile_id, actual_selection_id=selection_id,
                ):
                    self.trace.append({
                        "step": step, "state": state, "event": "cabinet_extraction_cap_exhausted",
                        "extracted_distance": extracted,
                    })
                    return
            elif state == "cabinet_settle":
                action = action_for(
                    eef, closed=True, limit=.10,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                if not raw_grasp:
                    state, state_steps = "cabinet_front_stage", 0
                elif state_steps >= cabinet_settle_step_requirement(controller_profile_id):
                    state, state_steps = "transit", 0
            elif state == "orient_for_drawer":
                state_steps += 1
                if not raw_grasp:
                    action = action_for(eef, closed=True, limit=.10)
                    state, state_steps = "align_yaw", 0
                elif drawer_orientation_pose_ready(vertical_half_extent=object_vertical_half_extent):
                    action = action_for(eef, closed=True, limit=.10)
                    drawer_horizontal_latched = update_drawer_horizontal_latch(
                        previous=drawer_horizontal_latched,
                        vertical_half_extent=object_vertical_half_extent,
                    )
                    grasp_offset = eef - obj
                    state, state_steps = "align_drawer_depth_yaw", 0
                elif state_steps >= 120:
                    action = action_for(eef, closed=True, limit=.10)
                    self.trace.append({
                        "step": step, "state": state, "event": "drawer_orientation_cap_exhausted",
                        "object_vertical_half_extent": object_vertical_half_extent,
                    })
                    return
                elif drawer_long_axis is not None:
                    leveling_rotation = drawer_horizontal_rotation_command(
                        long_axis=drawer_long_axis,
                        target_low=target_low,
                        target_high=target_high,
                        magnitude=drawer_orientation_control_magnitude(selection_id),
                    )
                    if selection_id == "VLA82-005":
                        drawer_leveling_rotation = drawer_leveling_rotation_latch(
                            drawer_leveling_rotation, leveling_rotation,
                        )
                        leveling_rotation = drawer_leveling_rotation
                    action = action_for(
                        eef, closed=True, limit=.10,
                        rotation=leveling_rotation,
                    )
                else:
                    action = action_for(eef, closed=True, limit=.10)
            elif state == "align_drawer_depth_yaw":
                geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                object_rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                object_size = np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
                long_axis = object_rotation[:, int(np.argmax(object_size))]
                drawer_depth_error = drawer_depth_alignment_error(
                    long_axis=long_axis, target_low=target_low, target_high=target_high,
                )
                action = action_for(
                    eef, closed=True, limit=.10,
                    rotation=np.array((0., 0., np.clip(drawer_depth_error / .35, -.5, .5))),
                )
                state_steps += 1
                if not raw_grasp:
                    state, state_steps = "align_yaw", 0
                elif abs(drawer_depth_error) <= np.deg2rad(5.):
                    grasp_offset = eef - obj
                    state, state_steps = "relieve_drawer_wrist", 0
                elif state_steps >= 40:
                    self.trace.append({
                        "step": step, "state": state, "event": "drawer_depth_alignment_cap_exhausted",
                        "drawer_depth_alignment_deg": float(np.degrees(abs(drawer_depth_error))),
                    })
                    return
            elif state == "relieve_drawer_wrist":
                arm_qpos = np.asarray(robot._joint_positions, dtype=float)
                joint5 = float(arm_qpos[4])
                state_steps += 1
                if not raw_grasp:
                    action = action_for(eef, closed=False, limit=.10)
                    state, state_steps = "align_yaw", 0
                elif not drawer_wrist_relief_required(joint5=joint5):
                    action = action_for(eef, closed=True, limit=.10)
                    grasp_offset = eef - obj
                    state, state_steps = "retreat_base_for_drawer", 0
                elif state_steps >= 32:
                    self.trace.append({
                        "step": step, "state": state, "event": "drawer_wrist_relief_cap_exhausted",
                        "joint5": joint5,
                    })
                    return
                else:
                    # The tool's long axis is aligned with world Y. Rotating
                    # about Y changes wrist redundancy without changing its
                    # drawer-depth alignment or collision envelope.
                    action = action_for(
                        eef, closed=True, limit=.10, rotation=np.array((0., .4, 0.)),
                    )
            elif state == "retreat_base_for_drawer":
                base_position, base_rotation = base_controller.get_base_pose()
                base_retreat_needed = drawer_base_retreat_required(
                    base=base_position, release=release,
                    minimum_distance=drawer_base_retreat_distance(selection_id),
                )
                torso_reserve_needed = drawer_torso_reserve_required(
                    selection_id=selection_id, torso_qpos=torso_qpos,
                )
                action = np.zeros_like(low, dtype=np.float32)
                action[layout.gripper] = 1.0
                action[layout.base_mode] = 1.0
                state_steps += 1
                if not drawer_hold_valid(
                    raw_grasp=raw_grasp, two_finger_contact=two_finger_contact,
                ):
                    state, state_steps = "align_yaw", 0
                elif not base_retreat_needed and not torso_reserve_needed:
                    grasp_offset = eef - obj
                    state, state_steps = "transit", 0
                elif base_retreat_needed and mobile_base_fixture_collision():
                    self.trace.append({
                        "step": step, "state": state, "event": "drawer_base_retreat_collision_abort",
                    })
                    return
                elif state_steps >= 80:
                    self.trace.append({
                        "step": step, "state": state, "event": "drawer_base_retreat_cap_exhausted",
                        "base_world": np.asarray(base_position, dtype=float).round(6).tolist(),
                    })
                    return
                else:
                    if base_retreat_needed:
                        local_direction = drawer_base_retreat_local_direction(
                            base=base_position, release=release, base_rotation=base_rotation,
                        )
                        action[layout.base[0]:layout.base[0] + 2] = np.clip(local_direction * .5, -.5, .5)
                    if torso_reserve_needed:
                        torso_command = .5
                        action[layout.torso] = torso_command
            elif state == "transit":
                held_offset = (
                    eef - obj if drawer_pick_place
                    else grasp_offset if grasp_offset is not None else np.zeros(3)
                )
                cabinet_target_final_stage = True
                if (
                    controller_profile_id == "VLA82-056"
                    and str(getattr(request, "target_fixture", "")) == "cabinet"
                ):
                    live_base_position, _ = base_controller.get_base_pose()
                    object_waypoint, cabinet_target_clearance_latched, cabinet_target_height_latched, cabinet_target_final_stage = cabinet_target_transport_waypoint(
                        selection_id=controller_profile_id,
                        object_center=obj,
                        release=release,
                        target_low=target_low,
                        target_high=target_high,
                        robot_base=live_base_position,
                        lift_height=self.LIFT_HEIGHT,
                        cleared=cabinet_target_clearance_latched,
                        raised=cabinet_target_height_latched,
                    )
                    desired = cabinet_target_eef_waypoint(
                        object_waypoint=object_waypoint,
                        eef=eef,
                        held_offset=held_offset,
                        raising=(
                            cabinet_target_clearance_latched
                            and not cabinet_target_height_latched
                            and not cabinet_target_final_stage
                        ),
                    )
                else:
                    desired = release + held_offset + np.array((0., 0., self.LIFT_HEIGHT))
                horizontal_distance = float(np.linalg.norm(obj[:2] - release[:2]))
                base_transport = not drawer_pick_place and pick_place_base_transport_required_for_selection(
                    controller_profile_id,
                    source_fixture=str(getattr(request, "source_fixture", "")),
                    target_fixture=str(getattr(request, "target_fixture", "")),
                    horizontal_distance=horizontal_distance,
                )
                if not drawer_pick_place and cabinet_transport_initial_base_x is not None:
                    live_base_position, _ = base_controller.get_base_pose()
                    base_transport = base_transport or pick_place_base_transport_latched(
                        controller_profile_id,
                        horizontal_distance=horizontal_distance,
                        initial_base_x=cabinet_transport_initial_base_x,
                        base_x=float(live_base_position[0]),
                    )
                base_transport_stage: str | None = None
                if base_transport:
                    live_base_position, _ = base_controller.get_base_pose()
                    if cabinet_transport_initial_base_x is None:
                        cabinet_transport_initial_base_x = float(live_base_position[0])
                    base_transport_stage = pick_place_cabinet_transport_stage(
                        base_x=float(live_base_position[0]),
                        initial_base_x=cabinet_transport_initial_base_x,
                        object_y=float(obj[1]),
                        release_y=float(release[1]),
                        arm_finish_y_tolerance=pick_place_cabinet_arm_finish_y_tolerance(
                            controller_profile_id,
                        ),
                    )
                    if base_transport_stage == "arm_finish":
                        base_transport = False
                if drawer_pick_place:
                    joint5 = float(np.asarray(robot._joint_positions, dtype=float)[4])
                    drawer_xy_reached = drawer_transit_xy_reached(
                        selection_id,
                        distance=float(np.linalg.norm(obj[:2] - release[:2])),
                    )
                    if (
                        drawer_depth_error is not None
                        and abs(float(drawer_depth_error)) <= np.deg2rad(25.)
                    ):
                        drawer_depth_alignment_latched = True
                    drawer_horizontal_latched = update_drawer_horizontal_latch(
                        previous=drawer_horizontal_latched,
                        vertical_half_extent=object_vertical_half_extent,
                    )
                    transit_stage = drawer_transit_control_stage(
                        vertical_half_extent=object_vertical_half_extent,
                        depth_alignment_error=drawer_depth_error,
                        xy_reached=drawer_xy_reached,
                        joint5=joint5,
                        depth_alignment_latched=drawer_depth_alignment_latched,
                        horizontal_latched=drawer_horizontal_latched,
                        vertical_threshold=drawer_vertical_transport_threshold(selection_id),
                    )
                    if transit_stage == "level":
                        action = action_for(
                            eef, closed=True, limit=.10,
                            rotation=drawer_transport_level_rotation_command(
                                long_axis=drawer_long_axis,
                                target_low=target_low,
                                target_high=target_high,
                            ) if drawer_long_axis is not None else np.zeros(3),
                        )
                    elif transit_stage == "align_depth":
                        action = action_for(
                            eef, closed=True, limit=.10,
                            rotation=np.array((0., 0., np.clip(drawer_depth_error / .35, -.5, .5))),
                        )
                    elif transit_stage == "translate":
                        action = action_for(
                            drawer_transit_world_target(eef=eef, desired=desired),
                            closed=True, limit=.45,
                        )
                    elif transit_stage == "relieve_wrist":
                        if drawer_wrist_relief_baseline is None:
                            drawer_wrist_relief_baseline = joint5
                        drawer_wrist_relief_direction, flipped = drawer_wrist_relief_feedback(
                            joint5=joint5,
                            baseline_joint5=drawer_wrist_relief_baseline,
                            attempts=drawer_wrist_relief_steps,
                            direction=drawer_wrist_relief_direction,
                            already_flipped=drawer_wrist_relief_flipped,
                        )
                        if flipped and not drawer_wrist_relief_flipped:
                            drawer_wrist_relief_baseline = joint5
                            drawer_wrist_relief_steps = 0
                        drawer_wrist_relief_flipped = flipped
                        drawer_wrist_relief_steps += 1
                        action = action_for(
                            eef, closed=True, limit=.10,
                            rotation=np.array((0., drawer_wrist_relief_direction, 0.)),
                        )
                    else:
                        action = action_for(eef, closed=True, limit=.10)
                elif base_transport:
                    if mobile_base_fixture_collision():
                        self.trace.append({
                            "step": step,
                            "state": state,
                            "event": "base_transport_collision_abort",
                            "horizontal_distance": horizontal_distance,
                        })
                        return
                    _, base_rotation = base_controller.get_base_pose()
                    action = np.zeros_like(low, dtype=np.float32)
                    if base_transport_stage == "clear_x":
                        transport_from, transport_to = (0., 0.), (1., 0.)
                    elif base_transport_stage == "return_x":
                        transport_from, transport_to = (1., 0.), (0., 0.)
                    else:
                        transport_from = (0., float(obj[1]))
                        transport_to = (0., float(release[1]))
                    action[layout.base[0]:layout.base[0] + 2] = pick_place_base_transport_local_direction(
                        object_center=transport_from,
                        release=transport_to,
                        base_rotation=base_rotation,
                        magnitude=pick_place_base_transport_magnitude(controller_profile_id),
                    )
                    action[layout.gripper] = pick_place_gripper_hold_command(selection_id)
                    action[layout.base_mode] = 1.0
                else:
                    if pick_place_uses_height_controlled_transit(selection_id):
                        transit_rotation = None
                        if pick_place_transit_preserves_shelf_orientation(selection_id):
                            _, base_rotation = base_controller.get_base_pose()
                            transit_rotation, _, _ = shelf_package_rotation_command(
                                object_rotation=support_rotation,
                                base_rotation=base_rotation,
                                maximum=.25,
                            )
                        action = action_for(
                            desired, closed=True,
                            limit=pick_place_transit_limit(selection_id),
                            rotation=transit_rotation,
                            gripper_command=pick_place_gripper_hold_command(selection_id),
                        )
                    elif pick_place_uses_direct_transit(selection_id):
                        horizontal_target = np.asarray(desired, dtype=float).copy()
                        horizontal_target[2] = float(eef[2])
                        action = action_for(
                            horizontal_target, closed=True,
                            limit=pick_place_transit_limit(selection_id),
                        )
                    else:
                        action = oracle_horizontal_action(
                            controller, eef, desired, low, high, layout=layout,
                            limit=pick_place_transit_limit(selection_id),
                        )
                torso_command = pick_place_transit_torso_command(
                    controller_profile_id,
                    eef_z=float(eef[2]),
                    target_z=float(desired[2]),
                    torso_qpos=torso_qpos,
                )
                action[layout.torso] = torso_command
                transport_hold_valid = pick_place_transport_hold_valid(
                    selection_id=selection_id,
                    drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                )
                transport_lost_grasp_frames = update_transport_lost_grasp_frames(
                    hold_valid=transport_hold_valid,
                    count=transport_lost_grasp_frames,
                )
                if transport_hold_recovery_required(transport_lost_grasp_frames):
                    drawer_depth_alignment_latched = False
                    drawer_horizontal_latched = False
                    drawer_wrist_relief_baseline = None
                    drawer_wrist_relief_steps = 0
                    drawer_wrist_relief_direction = .4
                    drawer_wrist_relief_flipped = False
                    state, state_steps = ("cabinet_front_stage", 0) if controller_profile_id == "VLA82-004" else ("close", 0)
                elif drawer_pick_place and drawer_transit_ready(
                    xy_reached=float(np.linalg.norm(obj[:2] - release[:2])) <= drawer_transit_xy_tolerance(selection_id),
                    vertical_half_extent=object_vertical_half_extent,
                    depth_alignment_error=drawer_depth_error,
                    joint5=float(np.asarray(robot._joint_positions, dtype=float)[4]),
                    vertical_threshold=drawer_vertical_transport_threshold(selection_id),
                ):
                    grasp_offset = eef - obj
                    state, state_steps = pick_place_post_transit_state(
                        selection_id, front_insert=drawer_front_insert,
                    ), 0
                elif not drawer_pick_place and cabinet_target_final_stage and pick_place_transit_ready_for_selection(
                    selection_id,
                    eef=eef,
                    desired=desired,
                    object_center=obj,
                    release=release,
                ):
                    if pick_place_refreshes_grasp_offset_before_lower(selection_id):
                        grasp_offset = eef - obj
                    state, state_steps = pick_place_post_transit_state(selection_id), 0
            elif state == "level_for_shelf":
                support_geom_id = raw.sim.model.geom_name2id(physical_object_geoms[0])
                object_rotation = np.asarray(
                    raw.sim.data.geom_xmat[support_geom_id], dtype=float,
                ).reshape(3, 3)
                _, base_rotation = base_controller.get_base_pose()
                leveling_rotation, leveling_error_deg, leveling_yaw_error_deg = shelf_package_rotation_command(
                    object_rotation=object_rotation,
                    base_rotation=base_rotation,
                    maximum=.40,
                )
                leveling_target = shelf_leveling_position_target(
                    release=release,
                    eef=eef,
                    object_center=obj,
                    lift_height=self.LIFT_HEIGHT,
                )
                action = action_for(
                    leveling_target, closed=True, limit=.35, rotation=leveling_rotation,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                next_state = shelf_leveling_transition(
                    hold_valid=pick_place_transport_hold_valid(
                        selection_id=selection_id,
                        drawer_pick_place=drawer_pick_place,
                        raw_grasp=raw_grasp,
                        two_finger_contact=two_finger_contact,
                    ),
                    alignment_error_deg=leveling_error_deg,
                    yaw_error_deg=leveling_yaw_error_deg,
                    xy_distance=float(np.linalg.norm(obj[:2] - release[:2])),
                    steps=state_steps,
                )
                if next_state == "failed":
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "shelf_leveling_cap_exhausted",
                        "support_normal_alignment_deg": leveling_error_deg,
                        "support_yaw_alignment_deg": leveling_yaw_error_deg,
                    })
                    return
                if next_state != state:
                    if next_state in {"lower", "transit"}:
                        grasp_offset = eef - obj
                    state, state_steps = next_state, 0
            elif state == "upright_for_counter":
                geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                object_axis = np.asarray(
                    raw.sim.data.geom_xmat[geom_id], dtype=float,
                ).reshape(3, 3)[:, 2]
                _, base_rotation = base_controller.get_base_pose()
                upright_rotation, upright_axis_error = upright_axis_rotation_command(
                    object_axis=object_axis, base_rotation=base_rotation,
                )
                action = action_for(
                    eef, closed=True, limit=.10, rotation=upright_rotation,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                if not raw_grasp:
                    state, state_steps = "cabinet_front_stage", 0
                elif upright_axis_error <= np.deg2rad(12.):
                    grasp_offset = eef - obj
                    counter_lower_base_start_x = float(base_controller.get_base_pose()[0][0])
                    state, state_steps = pick_place_post_upright_state(selection_id), 0
                elif state_steps >= 100:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "counter_upright_orientation_cap_exhausted",
                        "upright_axis_error_deg": float(np.degrees(upright_axis_error)),
                    })
                    return
            elif state == "counter_lower_base_reposition":
                base_position, base_rotation = base_controller.get_base_pose()
                if counter_lower_base_start_x is None:
                    counter_lower_base_start_x = float(base_position[0])
                desired_base_x = float(counter_lower_base_start_x) + (
                    pick_place_counter_upright_base_reposition_direction(selection_id)
                    * pick_place_counter_upright_base_reposition_distance(selection_id)
                )
                remaining = desired_base_x - float(base_position[0])
                action = np.zeros_like(low, dtype=np.float32)
                action[layout.gripper] = pick_place_gripper_hold_command(selection_id)
                action[layout.base_mode] = 1.0
                action[layout.base[0]:layout.base[0] + 2] = np.clip(
                    np.asarray(base_rotation, dtype=float)[:2, :2].T @ np.array((remaining, 0.0)) / .08,
                    -.35, .35,
                )
                state_steps += 1
                if not raw_grasp:
                    state, state_steps = "cabinet_front_stage", 0
                elif mobile_base_fixture_collision():
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "counter_lower_base_reposition_collision_abort",
                    })
                    return
                elif abs(remaining) <= .020:
                    grasp_offset = eef - obj
                    state, state_steps = pick_place_post_counter_base_reposition_state(selection_id), 0
                elif state_steps >= 60:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "counter_lower_base_reposition_cap_exhausted",
                        "remaining_x": remaining,
                    })
                    return
            elif state == "upright_recenter_for_counter":
                held_offset = eef - obj
                desired = release + held_offset
                desired[2] = float(eef[2])
                geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                object_axis = np.asarray(
                    raw.sim.data.geom_xmat[geom_id], dtype=float,
                ).reshape(3, 3)[:, 2]
                _, base_rotation = base_controller.get_base_pose()
                upright_rotation, upright_axis_error = upright_axis_rotation_command(
                    object_axis=object_axis, base_rotation=base_rotation,
                    tolerance=np.deg2rad(2.), maximum=.45,
                )
                action = action_for(
                    desired, closed=True, limit=.25, rotation=upright_rotation,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                if not raw_grasp:
                    state, state_steps = "cabinet_front_stage", 0
                elif (
                    float(np.linalg.norm(obj[:2] - release[:2])) <= .035
                    and upright_axis_error <= np.deg2rad(12.)
                ):
                    grasp_offset = eef - obj
                    state, state_steps = "lower", 0
                elif state_steps >= 120:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "counter_upright_recenter_cap_exhausted",
                        "horizontal_distance": float(np.linalg.norm(obj[:2] - release[:2])),
                        "upright_axis_error_deg": float(np.degrees(upright_axis_error)),
                    })
                    return
            elif state == "tilt_drawer_entry":
                contact = drawer_entry_fixture_contact()
                if contact is not None:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "drawer_tip_tilt_collision_abort", "contact": contact,
                    })
                    return
                _, base_rotation = base_controller.get_base_pose()
                world_rotation = np.zeros(3, dtype=float)
                depth_axis = int(np.argmax(target_high[:2] - target_low[:2]))
                orthogonal_axis = 1 - depth_axis
                world_rotation[orthogonal_axis] = .32 * drawer_pitch_rotation_direction(
                    target_low=target_low, target_high=target_high,
                    robot_base=base_controller.get_base_pose()[0],
                )
                local_rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3).T @ world_rotation
                action = action_for(
                    eef, closed=True, limit=.10, rotation=local_rotation,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                if drawer_tip_tilt_ready(eef=eef, object_center=obj):
                    grasp_offset = eef - obj
                    state, state_steps = "lower_tilted_drawer_entry", 0
                elif state_steps >= 80:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "drawer_tip_tilt_cap_exhausted",
                        "eef_object_vertical_offset": float(eef[2] - obj[2]),
                    })
                    return
            elif state == "lower_tilted_drawer_entry":
                contact = drawer_entry_fixture_contact()
                if contact is not None:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "drawer_tip_lower_collision_abort", "contact": contact,
                    })
                    return
                held_offset = eef - obj
                desired = release + held_offset
                desired[2] = drawer_tip_contact_height(float(desired[2]))
                action = action_for(
                    desired, closed=True, limit=.55,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                if not pick_place_transport_hold_valid(
                    selection_id=selection_id, drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                ):
                    state, state_steps = "close", 0
                elif (
                    drawer_tip_release_ready(
                        inner_bottom_contact=inner_bottom_contact,
                        inside_xy=inside_xy,
                        fixture_collision=False,
                    )
                    and float(obj[2]) <= float(target_high[2]) + .08
                ):
                    state, state_steps = "release", 0
                elif target_contact and not inner_bottom_contact and not drawer_tip_advance_completed:
                    drawer_tip_advance_world = drawer_tip_advance_target(
                        object_center=obj,
                        target_low=target_low,
                        target_high=target_high,
                        robot_base=base_controller.get_base_pose()[0],
                    )
                    grasp_offset = eef - obj
                    state, state_steps = "advance_tilted_drawer_entry", 0
                elif state_steps >= 180:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "drawer_tip_lower_cap_exhausted",
                        "eef_world": eef.round(6).tolist(),
                        "object_world": obj.round(6).tolist(),
                    })
                    return
            elif state == "advance_tilted_drawer_entry":
                contact = drawer_entry_fixture_contact()
                if contact is not None:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "drawer_tip_advance_collision_abort", "contact": contact,
                    })
                    return
                if drawer_tip_advance_world is None:
                    raise ValueError("drawer tip advance has no latched world target")
                held_offset = eef - obj
                desired = drawer_tip_advance_world + held_offset
                action = action_for(
                    desired, closed=True, limit=.45,
                    gripper_command=pick_place_gripper_hold_command(selection_id),
                )
                state_steps += 1
                if not pick_place_transport_hold_valid(
                    selection_id=selection_id, drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                ):
                    state, state_steps = "close", 0
                elif drawer_tip_release_ready(
                    inner_bottom_contact=inner_bottom_contact,
                    inside_xy=inside_xy,
                    fixture_collision=False,
                ):
                    state, state_steps = "release", 0
                elif float(np.linalg.norm(obj[:2] - drawer_tip_advance_world[:2])) <= .025:
                    drawer_tip_advance_completed = True
                    grasp_offset = eef - obj
                    state, state_steps = "lower_tilted_drawer_entry", 0
                elif state_steps >= 120:
                    self.trace.append({
                        "step": step, "state": state,
                        "event": "drawer_tip_advance_cap_exhausted",
                        "object_world": obj.round(6).tolist(),
                        "target_world": drawer_tip_advance_world.round(6).tolist(),
                    })
                    return
            elif state == "transit_front":
                if object_world_half_extents is None or drawer_floor_top is None:
                    raise ValueError("drawer insertion requires live object extents and floor support")
                staging, _ = drawer_front_insertion_targets(
                    target_low=target_low, target_high=target_high,
                    robot_base=base_controller.get_base_pose()[0],
                    world_half_extents=object_world_half_extents,
                    floor_top_z=drawer_floor_top,
                )
                held_offset = eef - obj
                high_staging = staging.copy()
                high_staging[2] = float(obj[2])
                action = action_for(high_staging + held_offset, closed=True, limit=.45)
                state_steps += 1
                transport_hold_valid = pick_place_transport_hold_valid(
                    selection_id=selection_id,
                    drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                )
                transport_lost_grasp_frames = update_transport_lost_grasp_frames(
                    hold_valid=transport_hold_valid,
                    count=transport_lost_grasp_frames,
                )
                if transport_hold_recovery_required(transport_lost_grasp_frames):
                    next_state = pick_place_lost_hold_transition(
                        drawer_pick_place=drawer_pick_place,
                        selection_id=selection_id,
                        target_contact=target_contact,
                        inside_xy=inside_xy,
                        release_pose_ready=float(np.linalg.norm(obj - release)) <= .035,
                    )
                    if next_state == "settle":
                        previous_settle_obj = obj.copy()
                        stable_release_steps = 0
                    state, state_steps = next_state, 0
                elif float(np.linalg.norm(obj[:2] - staging[:2])) <= .035:
                    grasp_offset = eef - obj
                    state, state_steps = "lower_front", 0
            elif state == "lower_front":
                if object_world_half_extents is None or drawer_floor_top is None:
                    raise ValueError("drawer insertion requires live object extents and floor support")
                staging, _ = drawer_front_insertion_targets(
                    target_low=target_low, target_high=target_high,
                    robot_base=base_controller.get_base_pose()[0],
                    world_half_extents=object_world_half_extents,
                    floor_top_z=drawer_floor_top,
                )
                held_offset = eef - obj
                desired = staging + held_offset
                state_steps += 1
                pose_correction = drawer_lower_pose_correction_required(
                    vertical_half_extent=object_vertical_half_extent,
                    horizontal_latched=True,
                    release_threshold=.075,
                )
                yaw_correction = not pose_correction and (
                    drawer_depth_error is None or abs(float(drawer_depth_error)) > np.deg2rad(25.)
                )
                if pose_correction:
                    action = action_for(
                        eef, closed=True, limit=.10,
                        rotation=drawer_lower_rotation_command(
                            long_axis=(drawer_long_axis if drawer_long_axis is not None else (0., 0., 1.)),
                            target_low=target_low, target_high=target_high,
                        ),
                    )
                elif yaw_correction:
                    action = action_for(
                        eef, closed=True, limit=.10,
                        rotation=np.array((0., 0., np.clip(float(drawer_depth_error or 0.0) / .35, -.5, .5))),
                    )
                else:
                    torso_command = -.5 if torso_qpos > .02 and float(obj[2]) > float(staging[2]) + .03 else 0.0
                    action = action_for(
                        desired, closed=True, limit=.8, torso_command=torso_command,
                    )
                if not pick_place_transport_hold_valid(
                    selection_id=selection_id, drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                ):
                    state, state_steps = "close", 0
                elif (
                    float(np.linalg.norm(obj[:2] - staging[:2])) <= .05
                    and float(obj[2]) <= float(staging[2]) + .035
                ):
                    grasp_offset = eef - obj
                    state, state_steps = "insert_drawer", 0
                elif state_steps >= 120:
                    self.trace.append({
                        "step": step, "state": state, "event": "drawer_front_lower_cap_exhausted",
                        "arm_qpos": np.asarray(robot._joint_positions, dtype=float).round(6).tolist(),
                        "eef_world": eef.round(6).tolist(), "object_world": obj.round(6).tolist(),
                    })
                    return
            elif state == "insert_drawer":
                if object_world_half_extents is None or drawer_floor_top is None:
                    raise ValueError("drawer insertion requires live object extents and floor support")
                _, inserted = drawer_front_insertion_targets(
                    target_low=target_low, target_high=target_high,
                    robot_base=base_controller.get_base_pose()[0],
                    world_half_extents=object_world_half_extents,
                    floor_top_z=drawer_floor_top,
                )
                held_offset = eef - obj
                desired = inserted + held_offset
                action = action_for(desired, closed=True, limit=.45)
                if not pick_place_transport_hold_valid(
                    selection_id=selection_id, drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                ):
                    if target_contact and inside_xy:
                        previous_settle_obj = obj.copy()
                        stable_release_steps = 0
                        state, state_steps = "settle", 0
                    else:
                        state, state_steps = "close", 0
                elif (
                    float(np.linalg.norm(obj[:2] - inserted[:2])) <= .04
                    and float(obj[2]) <= float(inserted[2]) + .05
                ):
                    state, state_steps = "release", 0
            elif state == "lower":
                held_offset = (eef - obj) if drawer_pick_place else (grasp_offset if grasp_offset is not None else np.zeros(3))
                desired = release + held_offset
                rotation = None
                if selection_id == "VLA82-055":
                    support_geom_id = raw.sim.model.geom_name2id(physical_object_geoms[0])
                    object_rotation = np.asarray(
                        raw.sim.data.geom_xmat[support_geom_id], dtype=float,
                    ).reshape(3, 3)
                    _, base_rotation = base_controller.get_base_pose()
                    rotation, _, _ = shelf_package_rotation_command(
                        object_rotation=object_rotation,
                        base_rotation=base_rotation,
                        maximum=.35,
                    )
                torso_command = pick_place_lower_torso_command(
                    selection_id,
                    eef_z=float(eef[2]),
                    target_z=float(desired[2]),
                    torso_qpos=torso_qpos,
                )
                drawer_base_recenter = drawer_pick_place and drawer_lower_base_recenter_enabled(
                    selection_id,
                ) and drawer_lower_base_recenter_required(
                    object_center=obj, release=release,
                )
                drawer_pose_correction = drawer_pick_place and drawer_lower_pose_correction_required(
                    vertical_half_extent=object_vertical_half_extent,
                    horizontal_latched=drawer_horizontal_latched,
                    release_threshold=drawer_lower_pose_release_threshold(selection_id),
                )
                drawer_yaw_correction = drawer_pick_place and not drawer_pose_correction and (
                    drawer_depth_error is not None
                    and abs(drawer_depth_error) > drawer_lower_yaw_tolerance(selection_id)
                )
                drawer_joint_correction = drawer_pick_place and not drawer_pose_correction and not drawer_yaw_correction and drawer_wrist_relief_required(
                    joint5=float(np.asarray(robot._joint_positions, dtype=float)[4]),
                )
                if drawer_locked_descent_enabled(selection_id):
                    drawer_base_recenter = False
                    if not drawer_locked_descent_keeps_pose_correction(selection_id):
                        drawer_pose_correction = False
                    drawer_yaw_correction = False
                    drawer_joint_correction = False
                lower_torso_command = drawer_lower_torso_command(
                    selection_id=selection_id,
                    joint5=float(np.asarray(robot._joint_positions, dtype=float)[4]),
                    torso_qpos=torso_qpos,
                ) if drawer_joint_correction else 0.0
                if drawer_base_recenter:
                    _, base_rotation = base_controller.get_base_pose()
                    action = np.zeros_like(low, dtype=np.float32)
                    action[layout.base[0]:layout.base[0] + 2] = drawer_lower_base_recenter_local_direction(
                        object_center=obj, release=release, base_rotation=base_rotation,
                    )
                    action[layout.gripper] = 1.0
                    action[layout.base_mode] = 1.0
                elif drawer_pose_correction:
                    lower_rotation = drawer_lower_rotation_command(
                        long_axis=(drawer_long_axis if drawer_long_axis is not None else (0., 0., 1.)),
                        target_low=target_low, target_high=target_high,
                    )
                    action = action_for(
                        eef, closed=True, limit=.10,
                        rotation=lower_rotation,
                    )
                elif drawer_yaw_correction:
                    action = action_for(
                        eef, closed=True, limit=.10,
                        rotation=np.array((0., 0., np.clip(float(drawer_depth_error) / .35, -.5, .5))),
                    )
                elif drawer_joint_correction and lower_torso_command:
                    torso_command = lower_torso_command
                    action = action_for(
                        desired, closed=True, limit=pick_place_lower_limit(selection_id),
                        torso_command=torso_command,
                    )
                elif drawer_joint_correction:
                    action = action_for(eef, closed=True, limit=.10, rotation=np.array((0., .4, 0.)))
                else:
                    if drawer_pick_place:
                        torso_command = drawer_vertical_descent_torso_command(
                            object_z=float(obj[2]),
                            release_z=float(release[2]),
                            torso_qpos=torso_qpos,
                        )
                    if pick_place_counter_upright_stage_required(selection_id):
                        geom_id = raw.sim.model.geom_name2id(object_geoms[0])
                        object_axis = np.asarray(
                            raw.sim.data.geom_xmat[geom_id], dtype=float,
                        ).reshape(3, 3)[:, 2]
                        _, base_rotation = base_controller.get_base_pose()
                        rotation, upright_axis_error = upright_axis_rotation_command(
                            object_axis=object_axis,
                            base_rotation=base_rotation,
                            # Tight maintenance prevents the held bottle from
                            # sagging as the hand performs the vertical descent.
                            tolerance=np.deg2rad(2.),
                            maximum=.45,
                        )
                    action = action_for(
                        desired,
                        closed=True,
                        limit=pick_place_lower_limit(selection_id),
                        torso_command=torso_command,
                        rotation=rotation,
                    )
                if (
                    drawer_pick_place and drawer_floor_top is not None
                    and not drawer_base_recenter
                    and drawer_gravity_release_pose_ready(
                        pose_correction=drawer_pose_correction,
                        yaw_correction=drawer_yaw_correction,
                        joint_correction=drawer_joint_correction,
                    )
                ):
                    drop_high = target_high.copy()
                    drop_high[2] = drawer_gravity_release_ceiling(
                        floor_top_z=drawer_floor_top,
                        vertical_half_extent=float(object_vertical_half_extent or 0.0),
                        maximum_drop=pick_place_maximum_gravity_drop(selection_id),
                    )
                    gravity_release_ready = drawer_drop_ready(
                        object_center=obj, target_low=target_low, target_high=drop_high,
                    )
                transport_hold_valid = pick_place_transport_hold_valid(
                    selection_id=selection_id,
                    drawer_pick_place=drawer_pick_place,
                    raw_grasp=raw_grasp,
                    two_finger_contact=two_finger_contact,
                )
                transport_lost_grasp_frames = update_transport_lost_grasp_frames(
                    hold_valid=transport_hold_valid,
                    count=transport_lost_grasp_frames,
                )
                if transport_hold_recovery_required(transport_lost_grasp_frames):
                    next_state = pick_place_lost_hold_transition(
                        drawer_pick_place=drawer_pick_place,
                        selection_id=selection_id,
                        target_contact=target_contact,
                        inside_xy=inside_xy,
                        release_pose_ready=float(np.linalg.norm(obj - release)) <= .035,
                    )
                    if next_state == "settle":
                        previous_settle_obj = obj.copy()
                        stable_release_steps = 0
                    state, state_steps = next_state, 0
                elif drawer_release_orientation_ready(
                    selection_id, vertical_half_extent=object_vertical_half_extent,
                ) and (
                    pick_place_target_contact_release_ready(
                        selection_id,
                        target_contact=target_contact,
                        inside_xy=inside_xy,
                        eef_distance=float(np.linalg.norm(desired - eef)),
                        support_alignment_deg=support_normal_alignment_deg,
                        support_yaw_alignment_deg=support_yaw_alignment_deg,
                    )
                    or gravity_release_ready
                ):
                    state, state_steps = "release", 0
            elif state == "release":
                post_release_retreat_target = pick_place_latched_retreat_target(
                    previous=post_release_retreat_target,
                    eef=eef,
                    object_center=obj,
                    released_supported=(not raw_grasp and target_contact),
                    selection_id=selection_id,
                )
                release_motion_target = eef
                if drawer_pick_place and not raw_grasp and exact_contact:
                    release_motion_target = drawer_post_release_interior_retreat_target(
                        eef=eef, object_center=obj,
                        target_low=target_low, target_high=target_high,
                    )
                elif post_release_retreat_target is not None:
                    release_motion_target = post_release_retreat_target
                elif drawer_pick_place and not raw_grasp and not target_contact:
                    release_motion_target = drawer_post_release_retreat_target(eef=eef, object_center=obj)
                action = action_for(release_motion_target, closed=False, limit=.12)
                state_steps += 1
                release_clear_frames = update_release_clear_frames(
                    raw_grasp=raw_grasp,
                    target_contact=target_contact,
                    exact_gripper_contact=exact_contact,
                    count=release_clear_frames,
                )
                retreat_complete = pick_place_retreat_complete(
                    selection_id,
                    eef=eef,
                    object_center=obj,
                    retreat_target=post_release_retreat_target,
                )
                if release_clearance_ready(release_clear_frames) and retreat_complete:
                    previous_settle_obj = obj.copy()
                    stable_release_steps = 0
                    state, state_steps = "settle", 0
            else:  # settle
                action = action_for(eef, closed=False, limit=.12)
                state_steps += 1
                stable_release_steps = update_stable_release_counter(
                    previous=previous_settle_obj, current=obj,
                    target_contact=target_contact and not raw_grasp,
                    count=stable_release_steps,
                )
                previous_settle_obj = obj.copy()
                if pick_place_recorded_stability_ready(stable_release_steps):
                    self.trace.append({"step": step, "state": "settle", "event": "pick_place_complete", "object_world": obj.round(6).tolist()})
                    return
            live_base_position, _ = base_controller.get_base_pose()
            self.trace.append({
                "step": step, "state": state, "phase": phase, "raw_grasp": raw_grasp,
                "exact_gripper_object_contact": exact_contact, "exact_object_target_contact": target_contact,
                "exact_two_finger_object_contact": two_finger_contact,
                "exact_two_pad_object_contact": two_pad_contact,
                "transport_lost_grasp_frames": transport_lost_grasp_frames,
                "selected_gripper_object_contacts": selected_contact_details,
                "exact_gripper_target_contact": bool(target_contact_details),
                "selected_gripper_target_contacts": target_contact_details,
                "support_normal_alignment_deg": support_normal_alignment_deg,
                "support_yaw_alignment_deg": support_yaw_alignment_deg,
                "pad_center_world": None if pad_center is None else np.asarray(pad_center, dtype=float).round(6).tolist(),
                "side_entry_world": None if side_entry_world is None else side_entry_world.round(6).tolist(),
                "drawer_front_stage_world": None if drawer_front_stage_world is None else drawer_front_stage_world.round(6).tolist(),
                "drawer_front_insert_world": None if drawer_front_insert_world is None else drawer_front_insert_world.round(6).tolist(),
                "drawer_entry_collision": drawer_entry_collision,
                "tool_axis_world": None if tool_axis_world is None else tool_axis_world.round(6).tolist(),
                "target_tool_axis_world": None if target_tool_axis_world is None else target_tool_axis_world.round(6).tolist(),
                "tool_alignment_error_rad": tool_alignment_error,
                "opening_axis_world": None if opening_axis_world is None else opening_axis_world.round(6).tolist(),
                "target_opening_axis_world": None if target_opening_axis_world is None else target_opening_axis_world.round(6).tolist(),
                "opening_alignment_error_rad": opening_alignment_error,
                "base_distance": base_distance,
                "cabinet_alignment_axis": alignment_axis, "cabinet_alignment_axis_error": alignment_error,
                "opening_axis_alignment_deg": None if yaw_alignment_error is None else float(np.degrees(abs(yaw_alignment_error))),
                "drawer_depth_alignment_deg": None if drawer_depth_error is None else float(np.degrees(abs(drawer_depth_error))),
                "upright_axis_error_deg": None if upright_axis_error is None else float(np.degrees(upright_axis_error)),
                "object_vertical_half_extent": object_vertical_half_extent,
                "drawer_gravity_release_ready": gravity_release_ready,
                "torso_qpos": torso_qpos, "torso_command": torso_command,
                "arm_qpos": np.asarray(robot._joint_positions, dtype=float).round(6).tolist(),
                "joint5": float(np.asarray(robot._joint_positions, dtype=float)[4]),
                "base_world": np.asarray(live_base_position, dtype=float).round(6).tolist(),
                "eef_world": eef.round(6).tolist(), "object_world": obj.round(6).tolist(),
                "release_world": release.round(6).tolist(), "action": np.asarray(action, dtype=float).round(6).tolist(),
            })
            previous_gripper_aperture = live_gripper_aperture
            yield phase, action
        self.trace.append({"state": state, "event": "pick_place_cap_exhausted"})


def torso_probe_action(
    low: np.ndarray, high: np.ndarray, *, layout: ActionLayout, command: float,
) -> np.ndarray:
    """Issue one stationary-base torso calibration command through the public action API."""
    action = np.zeros_like(np.asarray(low, dtype=np.float32))
    action[layout.torso] = float(command)
    action[layout.gripper] = -1.0
    action[layout.base_mode] = -1.0
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def axis2_pad_descent_action(
    low: np.ndarray, high: np.ndarray, *, layout: ActionLayout,
) -> np.ndarray:
    """Lower the two finger pads with the measured Panda arm axis-2 response.

    This is intentionally a small public-control pulse, rather than a pose
    write or an unbounded downward servo.  It is used only after the torso has
    reached its lower travel limit and the next live pad observation remains
    above the selected can's side-wall band.
    """
    action = np.zeros_like(np.asarray(low, dtype=np.float32))
    action[layout.arm[0] + 2] = -.20
    action[layout.gripper] = -1.0
    action[layout.base_mode] = -1.0
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def vla82_001_pregrasp_action(
    low: np.ndarray, high: np.ndarray, *, layout: ActionLayout,
) -> np.ndarray:
    """Keep VLA82-001's measured-horizontal fresh wrist pose unchanged.

    The default PandaOmron reset has a <=3 mm two-pad height difference.
    Its former axis-5 pre-rotation created the observed ~38 mm skew, so the
    trash-can path must not rotate before its fresh XY probes and descent.
    """
    action = np.zeros_like(np.asarray(low, dtype=np.float32))
    action[layout.gripper] = -1.0
    action[layout.base_mode] = -1.0
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def level_pads_action(
    low: np.ndarray, high: np.ndarray, *, layout: ActionLayout,
) -> np.ndarray:
    """Use the fixed-scene best public rotation to reduce pad-height skew.

    The saved ± axis-3/4/5 probe selected axis 4, negative direction: it
    reduced the two-pad height difference fastest.  The action remains a small
    open-gripper public step and is followed by fresh pad / XY observations.
    """
    action = np.zeros_like(np.asarray(low, dtype=np.float32))
    action[layout.arm[0] + 4] = -.20
    action[layout.gripper] = -1.0
    action[layout.base_mode] = -1.0
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def needs_pad_leveling(pad_z: Sequence[float], *, lower: float, upper: float, tolerance: float = .003) -> bool:
    """Level only when one live pad is valid and its mate remains above the rim."""
    values = np.asarray(tuple(float(value) for value in pad_z), dtype=float)
    return bool(
        values.size >= 2
        and np.any((values >= float(lower)) & (values <= float(upper)))
        and float(np.max(values)) > float(upper)
        and float(np.ptp(values)) > float(tolerance)
    )


def should_issue_grasp_vertical_descent(*, direct_action_active: bool, contact_hold_active: bool, vertical_gap: float) -> bool:
    """Guard the arm-Z fallback so it cannot overwrite a direct grasp action."""
    return not direct_action_active and not contact_hold_active and float(vertical_gap) > .045


def grasp_closure_target(eef: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Place the gripper at the observed can centre before closing its fingers."""
    target = np.asarray(obj, dtype=float).copy()
    target[2] += .005
    return target


def deposit_vertical_descend_action(low: np.ndarray, high: np.ndarray, *, layout: ActionLayout | None = None, closed: bool = False) -> np.ndarray:
    """Return the calibrated public PandaOmron local-Z descent command."""
    action = np.zeros_like(np.asarray(low, dtype=np.float32))
    resolved = layout or ActionLayout.canonical(int(action.size))
    # The third OSC translation degree of freedom is the local-Z command;
    # it is relative to the controller's live right-arm split.
    action[resolved.arm[0] + 2] = -.70
    action[resolved.gripper] = 1.0 if closed else -1.0
    action[resolved.base_mode] = -1.0
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def deposit_downward_orientation_action(low: np.ndarray, high: np.ndarray, *, layout: ActionLayout | None = None) -> np.ndarray:
    """Use the calibrated wrist-yaw command that gives the can a vertical approach."""
    action = np.zeros_like(np.asarray(low, dtype=np.float32))
    resolved = layout or ActionLayout.canonical(int(action.size))
    # The sixth right-arm degree of freedom is the calibrated wrist posture.
    action[resolved.arm[0] + 5] = .6
    action[resolved.gripper] = -1.0
    action[resolved.base_mode] = -1.0
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def deposit_base_micro_align_action(
    low: np.ndarray, high: np.ndarray, *, layout: ActionLayout,
    eef_world: np.ndarray, can_world: np.ndarray, base_rotation: np.ndarray,
) -> np.ndarray:
    """Centimetre-scale base correction in live controller slots only."""
    action = np.zeros_like(np.asarray(low, dtype=np.float32))
    # PandaOmron's mobile-base command translates the arm workspace opposite
    # the controller-frame vector.  This sign follows the live 4-step probe
    # (the unreversed command increased XY error by 32.9 mm).
    world_delta = np.clip(np.asarray(eef_world, dtype=float)[:2] - np.asarray(can_world, dtype=float)[:2], -.035, .035)
    local_delta = np.asarray(base_rotation, dtype=float)[:2, :2].T @ world_delta
    action[layout.base[0]:layout.base[0] + 2] = np.clip(local_delta / .04, -.35, .35)
    action[layout.gripper] = -1.0
    action[layout.base_mode] = 1.0
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def deposit_descend_world_target(eef: np.ndarray, can: np.ndarray, cavity_center: np.ndarray) -> np.ndarray:
    """One public-control descent waypoint with live can-centering feedback."""
    current_eef = np.asarray(eef, dtype=float)
    current_can = np.asarray(can, dtype=float)
    center = np.asarray(cavity_center, dtype=float)
    target = current_eef.copy()
    target[:2] += center[:2] - current_can[:2]
    target[2] -= .01
    return target


def deposit_transport_world_target(
    eef: np.ndarray,
    can: np.ndarray,
    cavity_center: np.ndarray,
    *,
    rim_z: float,
    rim_clearance: float,
    can_bottom_offset: float,
) -> np.ndarray:
    """Live closed-loop transit target for a held can above a receptacle.

    The measured can centre is the controlled quantity.  Recomputing from the
    current end-effector pose preserves any non-zero grasp transform, rather
    than attempting to replay the transform captured at lift time.
    """
    current_eef = np.asarray(eef, dtype=float)
    current_can = np.asarray(can, dtype=float)
    center = np.asarray(cavity_center, dtype=float)
    target = current_eef.copy()
    target[:2] += center[:2] - current_can[:2]
    desired_can_z = float(rim_z) + float(rim_clearance) - float(can_bottom_offset)
    target[2] += desired_can_z - current_can[2]
    return target


def deposit_transport_axis_target(
    eef: np.ndarray,
    can: np.ndarray,
    cavity_center: np.ndarray,
) -> np.ndarray:
    """One-axis, live-can XY waypoint that limits arm-axis coupling."""
    current_eef = np.asarray(eef, dtype=float)
    current_can = np.asarray(can, dtype=float)
    center = np.asarray(cavity_center, dtype=float)
    target = current_eef.copy()
    xy_error = center[:2] - current_can[:2]
    axis = int(np.argmax(np.abs(xy_error)))
    target[axis] += xy_error[axis]
    return target


def update_deposit_axis_calibration(
    *, previous_abs_error: float, current_abs_error: float, sign: float, worsening_steps: int,
) -> tuple[float, int]:
    """Reverse a single public-control axis only after two measured regressions."""
    worsening = int(worsening_steps) + 1 if float(current_abs_error) > float(previous_abs_error) + .0005 else 0
    if worsening >= 2:
        return -float(sign), 0
    return float(sign), worsening


def deposit_requires_base_transit(*, horizontal_distance: float, base_distance: float) -> bool:
    """Use the mobile base once a source lies outside the arm's close reach."""
    return bool(float(horizontal_distance) > .06 and float(base_distance) > .30)


def base_feedback_command(jacobian: np.ndarray, desired_eef_delta: np.ndarray, *, maximum: float = .30) -> np.ndarray:
    """Bounded 2-D public base command from a measured local Jacobian."""
    jac = np.asarray(jacobian, dtype=float).reshape(2, 2)
    delta = np.asarray(desired_eef_delta, dtype=float).reshape(2)
    return np.clip(np.linalg.pinv(jac, rcond=1e-4) @ delta, -float(maximum), float(maximum))


def estimate_eef_xy_jacobian(samples: Mapping[int, Mapping[float, np.ndarray]], *, amplitude: float) -> np.ndarray:
    """Estimate d(EEF world XY)/d(public arm XY action) from symmetric probes."""
    width = max(samples.keys(), default=1) + 1
    result = np.zeros((2, width), dtype=float)
    for axis in range(width):
        positive = samples.get(axis, {}).get(1.0)
        negative = samples.get(axis, {}).get(-1.0)
        if positive is not None and negative is not None:
            result[:, axis] = (np.asarray(positive, dtype=float).reshape(2) - np.asarray(negative, dtype=float).reshape(2)) / (2.0 * float(amplitude))
    return result


def eef_xy_damped_command(jacobian: np.ndarray, desired_eef_delta: np.ndarray, *, damping: float = 1e-5, maximum: float = .30) -> np.ndarray:
    """Bounded public arm-XY correction through an observed EEF Jacobian."""
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 2:
        raise ValueError(f"EEF XY Jacobian must have two output rows, got {matrix.shape}")
    target = np.asarray(desired_eef_delta, dtype=float).reshape(2)
    inverse = matrix.T @ np.linalg.inv(matrix @ matrix.T + float(damping) * np.eye(2))
    return np.clip(inverse @ target, -float(maximum), float(maximum))


def can_grasp_xy_feedback_jacobian(jacobian: np.ndarray, *, requires_axis2_pad_band: bool) -> np.ndarray:
    """Choose XY-only feedback columns while the axis-2 pad height is guarded."""
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 2:
        raise ValueError(f"EEF XY Jacobian must have two output rows, got {matrix.shape}")
    if requires_axis2_pad_band:
        if matrix.shape[1] < 2:
            raise ValueError(f"axis-2-safe XY refinement needs two arm axes, got {matrix.shape}")
        return matrix[:, :2]
    return matrix


def estimate_held_can_jacobian(samples: Mapping[int, Sequence[np.ndarray]], *, amplitude: float) -> np.ndarray:
    """Estimate d(can_xyz)/d(public arm translation action) from ± probes."""
    result = np.zeros((3, 3), dtype=float)
    for axis in range(3):
        responses = [np.asarray(value, dtype=float).reshape(3) for value in samples.get(axis, ())]
        if responses:
            result[:, axis] = np.mean(np.asarray(responses), axis=0) / float(amplitude)
    return result


def held_can_damped_command(jacobian: np.ndarray, desired_can_delta: np.ndarray, *, damping: float = 1e-5, maximum: float = .18) -> np.ndarray:
    """Bounded damped least-squares public action from measured held-can motion."""
    matrix = np.asarray(jacobian, dtype=float).reshape(3, 3)
    target = np.asarray(desired_can_delta, dtype=float).reshape(3)
    inverse = matrix.T @ np.linalg.inv(matrix @ matrix.T + float(damping) * np.eye(3))
    return np.clip(inverse @ target, -float(maximum), float(maximum))


def deposit_opening_release_ready(can: np.ndarray, cavity_min: np.ndarray, cavity_max: np.ndarray, can_bottom_offset: float) -> bool:
    """Require a centered can bottom 2--8cm above the open receptacle rim."""
    point = np.asarray(can, dtype=float)
    lower, upper = np.asarray(cavity_min, dtype=float), np.asarray(cavity_max, dtype=float)
    xy_inside = DepositStateMachine._xy_inside(point, lower, upper)
    # MuJoCo models expose ``bottom_offset`` in the object-local frame, so a
    # cylinder bottom is negative (for a 5.5cm can: -0.055).  Add it to the
    # world COM; do not mistake this signed value for a positive half-height.
    rim_clearance = float(point[2] + float(can_bottom_offset) - upper[2])
    return bool(xy_inside and .02 <= rim_clearance <= .08)


def deposit_grasp_verified(*, raw_grasp: bool, two_pad_contact: bool) -> bool:
    """Only a true two-pad pinch may advance a can to the lift state."""
    return bool(raw_grasp) and bool(two_pad_contact)


def live_pad_side_band(raw: Any, *, can_center_z: float, can_half_height: float, margin: float = .012) -> tuple[bool, tuple[float, ...]]:
    """Read the two physical Panda pad centres and test their side-wall band."""
    pads = []
    for index in range(int(raw.sim.model.ngeom)):
        name = raw.sim.model.geom_id2name(index) or ""
        if "finger" in name.lower() and "pad_collision" in name.lower():
            pads.append(float(raw.sim.data.geom_xpos[index][2]))
    lower, upper = float(can_center_z) - float(can_half_height) + float(margin), float(can_center_z) + float(can_half_height) - float(margin)
    return bool(len(pads) >= 2 and all(lower <= value <= upper for value in pads)), tuple(pads)


def pad_midpoint_world(raw: Any) -> np.ndarray:
    """Return the live midpoint of exactly the two physical finger pad geoms."""
    pads = [
        np.asarray(raw.sim.data.geom_xpos[index], dtype=float)
        for index in range(int(raw.sim.model.ngeom))
        if "finger" in (raw.sim.model.geom_id2name(index) or "").lower()
        and "pad_collision" in (raw.sim.model.geom_id2name(index) or "").lower()
    ]
    if len(pads) != 2:
        raise ValueError(f"WholeBodyIK requires exactly two finger pad_collision geoms, got {len(pads)}")
    return np.mean(np.asarray(pads, dtype=float), axis=0)


def eef_target_for_pad_midpoint(*, eef_world: np.ndarray, live_pad_midpoint: np.ndarray, desired_pad_midpoint: np.ndarray) -> np.ndarray:
    """Translate an absolute EEF goal so the observed pad midpoint reaches its target."""
    return np.asarray(eef_world, dtype=float) + np.asarray(desired_pad_midpoint, dtype=float) - np.asarray(live_pad_midpoint, dtype=float)


def grasp_pad_target(can_world: np.ndarray, half_height: float, *, margin: float = .012, inset: float = .003) -> np.ndarray:
    """Aim the pad midpoint just inside the upper half of the can side band."""
    target = np.asarray(can_world, dtype=float).copy()
    target[2] += float(half_height) - float(margin) - float(inset)
    return target


class WholeBodyGraspGate:
    """Small observation-only gate for the first WholeBodyIK can acquisition."""

    LIMITS = {"approach_safe": 160, "descend_to_side_band": 160, "close": 64, "lift": 64}

    def __init__(self) -> None:
        self.state = "approach_safe"
        self.state_steps = 0
        self.pinch_steps = 0
        self.failed = False

    def observe(self, *, horizontal_distance: float, pad_in_side_band: bool, raw_grasp: bool, two_pad_contact: bool) -> str:
        self.state_steps += 1
        if self.state_steps > self.LIMITS[self.state]:
            self.failed = True
            return self.state
        centered = float(horizontal_distance) <= .004
        if self.state == "approach_safe" and centered:
            self.state, self.state_steps = "descend_to_side_band", 0
        elif self.state == "descend_to_side_band":
            if float(horizontal_distance) > .030:
                self.state, self.state_steps = "approach_safe", 0
            elif centered and bool(pad_in_side_band):
                self.state, self.state_steps = "close", 0
                self.pinch_steps = 0
        elif self.state == "close":
            if not (centered and bool(pad_in_side_band)):
                self.state, self.state_steps, self.pinch_steps = "descend_to_side_band", 0, 0
            else:
                self.pinch_steps = self.pinch_steps + 1 if deposit_grasp_verified(raw_grasp=raw_grasp, two_pad_contact=two_pad_contact) else 0
                if self.pinch_steps >= 3:
                    self.state, self.state_steps = "lift", 0
        return self.state


class WholeBodyDepositGate:
    """Observation-only transport, release, and settle gate after a verified lift."""

    LIMITS = {
        "raise_above_rim": 128,
        "transit_above_cavity": 128,
        "lower_inside": 128,
        "release": 32,
        "settle": 48,
    }

    def __init__(self, *, cavity_min: np.ndarray, cavity_max: np.ndarray, signed_bottom_offset: float, slot_xy: np.ndarray | None = None) -> None:
        self.cavity_min = np.asarray(cavity_min, dtype=float)
        self.cavity_max = np.asarray(cavity_max, dtype=float)
        self.signed_bottom_offset = float(signed_bottom_offset)
        self.slot_xy = (
            (self.cavity_min[:2] + self.cavity_max[:2]) / 2.0
            if slot_xy is None else np.asarray(slot_xy, dtype=float)
        )
        self.state = "lift_verified"
        self.state_steps = 0
        self.stable_open_steps = 0
        self.failed = False

    def _transition(self, state: str) -> str:
        if state != self.state:
            self.state = state
            self.state_steps = 0
        return self.state

    def _xy_safe(self, can_position: np.ndarray) -> bool:
        return DepositStateMachine._xy_inside(np.asarray(can_position, dtype=float), self.cavity_min, self.cavity_max)

    def _slot_aligned(self, can_position: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(can_position, dtype=float)[:2] - self.slot_xy)) <= .005

    def observe(self, *, raw_grasp: bool, two_pad_contact: bool, can_position: np.ndarray, pose_stable: bool) -> str:
        can = np.asarray(can_position, dtype=float)
        pinch = deposit_grasp_verified(raw_grasp=raw_grasp, two_pad_contact=two_pad_contact)
        if self.state in {"lift_verified", "raise_above_rim", "transit_above_cavity", "lower_inside"} and not pinch:
            return self._transition("reacquire")
        if self.state == "lift_verified":
            self._transition("raise_above_rim")
        bottom = float(can[2] + self.signed_bottom_offset)
        if self.state == "raise_above_rim" and bottom >= float(self.cavity_max[2]) + .04:
            self._transition("transit_above_cavity")
        elif self.state == "transit_above_cavity" and self._xy_safe(can) and self._slot_aligned(can):
            self._transition("lower_inside")
        elif self.state == "lower_inside":
            center_inside = float(self.cavity_min[2]) + .05 <= float(can[2]) <= float(self.cavity_max[2]) - .05
            if self._xy_safe(can) and self._slot_aligned(can) and bottom < float(self.cavity_max[2]) and center_inside:
                self._transition("release")
        elif self.state == "release" and not raw_grasp:
            self._transition("settle")
            self.stable_open_steps = 0
        elif self.state == "settle":
            self.stable_open_steps = self.stable_open_steps + 1 if (not raw_grasp and self._xy_safe(can) and pose_stable) else 0
            if self.stable_open_steps >= 8:
                self._transition("complete")
        if self.state in self.LIMITS:
            self.state_steps += 1
            if self.state_steps > self.LIMITS[self.state]:
                self.failed = True
        return self.state


class DepositStateMachine:
    """Pure guarded progression for a single physical trash-can deposit."""

    LIMITS = {"reach": 192, "grasp": 400, "secure": 18, "lift": 56,
              "above_cavity": 360, "descend": 144, "release": 18, "settle": 48}

    def __init__(self) -> None:
        self.state = "reach"

    def limit_for(self, state: str) -> int:
        return int(self.LIMITS[state])

    @staticmethod
    def _xy_inside(position: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, margin: float = .018) -> bool:
        point, low, high = np.asarray(position, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        return bool(np.all(point[:2] >= low[:2] + margin) and np.all(point[:2] <= high[:2] - margin))

    def advance(self, *, distance: float, grasped: bool, can_position: np.ndarray, cavity_min: np.ndarray, cavity_max: np.ndarray, horizontal_distance: float | None = None, cavity_center_distance: float | None = None, object_bottom_offset: float = .055) -> str:
        point, low, high = np.asarray(can_position, dtype=float), np.asarray(cavity_min, dtype=float), np.asarray(cavity_max, dtype=float)
        if self.state in {"lift", "above_cavity", "descend"} and not grasped:
            # A can is not transported after the physical two-pad pinch is
            # gone.  The caller returns to the live can pose and reacquires.
            self.state = "reach"
        elif self.state == "reach" and distance <= .045 and (horizontal_distance is None or horizontal_distance <= .015):
            self.state = "grasp"
        elif self.state == "grasp" and grasped:
            self.state = "lift"
        elif self.state == "lift" and grasped and point[2] >= low[2] + .11:
            self.state = "above_cavity"
        elif self.state == "above_cavity" and grasped and (cavity_center_distance is None or cavity_center_distance <= .015) and deposit_opening_release_ready(point, low, high, object_bottom_offset):
            self.state = "release"
        elif self.state == "descend" and grasped and self._xy_inside(point, low, high) and low[2] + .05 <= point[2] <= high[2] - .05:
            self.state = "release"
        elif self.state == "release" and not grasped:
            self.state = "settle"
        return self.state


class CanGraspController:
    """Independent, guarded physical grasp sequence for one cylindrical can.

    It owns only the acquisition sequence.  The deposit state machine takes
    over after ``ready_for_deposit`` so no transport code can re-run wrist
    orientation or a stale reach waypoint.
    """

    LIMITS = {
        "init_orientation": 24,
        "xy_center": 128,
        "torso_descend": 24,
        "axis2_pad_descend": 96,
        "level_pads": 48,
        "final_xy_refine": 64,
        "close": 48,
        "torso_lift": 28,
    }
    MAX_TORSO_PULSES = 14
    MAX_AXIS2_PULSES = 128
    AXIS2_XY_DRIFT_LIMIT = .012
    LEVEL_XY_DRIFT_LIMIT = .030

    def __init__(self) -> None:
        self.state = "init_orientation"
        self.state_steps = 0
        self.stable_steps = 0
        self.lift_start_z: float | None = None
        self.torso_pulse_start_z: float | None = None
        self.last_torso_world_dz: float | None = None
        self.torso_pulses = 0
        self.axis2_pulse_start_pad_z: tuple[float, ...] | None = None
        self.last_axis2_pad_dz: float | None = None
        self.axis2_pulses = 0
        self.requires_axis2_pad_band = False
        self.level_previous_difference: float | None = None
        self.level_nonimproving_steps = 0
        self.failed = False

    def _transition(self, state: str) -> str:
        self.state = state
        self.state_steps = 0
        if state != "close":
            self.stable_steps = 0
        if state != "level_pads":
            self.level_previous_difference = None
            self.level_nonimproving_steps = 0
        return self.state

    def recenter(self) -> str:
        """Return only to lateral centering; orientation is never repeated."""
        return self._transition("xy_center")

    def begin_torso_pulse(self, eef_z: float) -> None:
        if self.state != "torso_descend" or self.torso_pulse_start_z is not None:
            return
        self.torso_pulse_start_z = float(eef_z)
        self.torso_pulses += 1
        if self.torso_pulses > self.MAX_TORSO_PULSES:
            # The torso is at its physical lower travel limit.  Continue with
            # the separately calibrated arm-axis-2 pad descent instead of
            # falsely declaring an unobserved pinch impossible.
            self.torso_pulse_start_z = None
            self.requires_axis2_pad_band = True
            self._transition("axis2_pad_descend")

    def begin_axis2_pulse(self, pad_z: Sequence[float]) -> None:
        """Record the live pad heights before one bounded negative axis-2 step."""
        if self.state != "axis2_pad_descend" or self.axis2_pulse_start_pad_z is not None:
            return
        self.axis2_pulse_start_pad_z = tuple(float(value) for value in pad_z)
        self.axis2_pulses += 1
        if self.axis2_pulses > self.MAX_AXIS2_PULSES:
            self.failed = True

    @property
    def ready_for_deposit(self) -> bool:
        return self.state == "ready_for_deposit"

    def observe(
        self, *, orientation_complete: bool, horizontal_distance: float,
        vertical_gap: float, raw_grasp: bool, two_pad_contact: bool, eef_z: float,
        pad_in_side_band: bool = False, pad_z: Sequence[float] = (), pad_level_required: bool = False,
    ) -> str:
        if self.state in self.LIMITS:
            self.state_steps += 1
            if self.state_steps > self.LIMITS[self.state]:
                self.failed = True
                return self.state
        if self.state == "init_orientation" and orientation_complete:
            return self._transition("xy_center")
        pulse_center_threshold = .006 if self.torso_pulses > 0 else .004
        if self.state == "xy_center" and float(horizontal_distance) <= pulse_center_threshold:
            return self._transition("torso_descend")
        if self.state == "torso_descend" and self.torso_pulse_start_z is not None:
            self.last_torso_world_dz = float(eef_z) - self.torso_pulse_start_z
            self.torso_pulse_start_z = None
            if self.last_torso_world_dz > -.002:
                # A non-descending public torso pulse is the live saturation
                # signal.  Its only safe recovery is the calibrated small
                # axis-2 pad descent; do not continue issuing the saturated
                # torso command and do not promote a high pad position to a
                # close attempt.
                self.requires_axis2_pad_band = True
                return self._transition("axis2_pad_descend")
            if float(vertical_gap) <= .022:
                return self._transition("final_xy_refine")
            return self.recenter()
        if self.state == "axis2_pad_descend" and self.axis2_pulse_start_pad_z is not None:
            before = np.asarray(self.axis2_pulse_start_pad_z, dtype=float)
            after = np.asarray(tuple(float(value) for value in pad_z), dtype=float)
            if before.shape == after.shape and before.size >= 2:
                self.last_axis2_pad_dz = float(np.mean(after - before))
            self.axis2_pulse_start_pad_z = None
        if self.state in {"torso_descend", "close"} and float(horizontal_distance) > .005:
            return self.recenter()
        if self.state == "axis2_pad_descend":
            # Keep axis-2 fixed while re-centering.  The final XY state returns
            # here until both pads are physically in the side-wall band.
            if float(horizontal_distance) > self.AXIS2_XY_DRIFT_LIMIT:
                return self._transition("final_xy_refine")
            if bool(pad_level_required):
                return self._transition("level_pads")
            if bool(pad_in_side_band):
                return self._transition("final_xy_refine")
            return self.state
        if self.state == "level_pads":
            # The selected axis-4 rotation has bounded XY coupling.  Keep
            # leveling through that small drift, then final-refine XY only
            # after the two pads have been leveled.
            if float(horizontal_distance) > self.LEVEL_XY_DRIFT_LIMIT:
                return self._transition("final_xy_refine")
            difference = float(np.ptp(np.asarray(pad_z, dtype=float))) if len(tuple(pad_z)) >= 2 else float("inf")
            if self.level_previous_difference is not None:
                self.level_nonimproving_steps = self.level_nonimproving_steps + 1 if difference >= self.level_previous_difference - .0001 else 0
            self.level_previous_difference = difference
            if self.level_nonimproving_steps >= 4:
                self.failed = True
                return self.state
            if difference <= .003:
                return self._transition("final_xy_refine")
            return self.state
        if self.state == "torso_descend" and float(horizontal_distance) <= .012 and float(vertical_gap) <= .022:
            return self._transition("final_xy_refine")
        final_xy_threshold = .004
        if self.state == "final_xy_refine" and float(horizontal_distance) <= final_xy_threshold:
            if self.requires_axis2_pad_band and not bool(pad_in_side_band):
                return self._transition("axis2_pad_descend")
            return self._transition("close")
        if self.state == "close":
            self.stable_steps = self.stable_steps + 1 if deposit_grasp_verified(raw_grasp=raw_grasp, two_pad_contact=two_pad_contact) else 0
            if self.stable_steps >= 3:
                self.lift_start_z = float(eef_z)
                return self._transition("torso_lift")
        if self.state == "torso_lift":
            if not deposit_grasp_verified(raw_grasp=raw_grasp, two_pad_contact=two_pad_contact):
                return self.recenter()
            if self.lift_start_z is not None and float(eef_z) - self.lift_start_z >= .045:
                return self._transition("ready_for_deposit")
        return self.state


def vla82_whole_body_deposit_plan() -> tuple[tuple[str, np.ndarray], ...]:
    """Bind every named VLA82-001 source can to one fixed cavity slot."""
    from .environment import vla82_trash_can_cavity_slot_centers

    object_ids = ("obj", "annotated_01", "annotated_02")
    slots = tuple(np.asarray(slot, dtype=float) for slot in vla82_trash_can_cavity_slot_centers())
    if len(object_ids) != len(slots) or len({tuple(slot) for slot in slots}) != len(slots):
        raise ValueError("VLA82-001 requires three distinct source-to-slot bindings")
    return tuple(zip(object_ids, slots))


class WholeBodyDepositExpert:
    """Bounded WholeBodyIK grasp-to-release cycles for all three source cans."""

    SAFE_PAD_HEIGHT = .13
    LIFT_HEIGHT = .12
    LIFT_PROOF = .045

    def __init__(self, contract: Any):
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    def run(self, environment: Any) -> Iterable[tuple[str, np.ndarray]]:
        plan = vla82_whole_body_deposit_plan()
        for index, (object_id, slot_xy) in enumerate(plan):
            completed = yield from self._run_one(environment, object_id, slot_xy)
            if completed is not True:
                return
            if index + 1 < len(plan):
                exited = yield from self._exit_open_bin(environment, after_object_id=object_id)
                if exited is not True:
                    return

    def _run_one(self, environment: Any, object_id: str, slot_xy: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
        raw = _raw_environment(environment)
        robot = raw.robots[0]
        controller = robot.composite_controller
        layout = WholeBodyActionLayout.from_controller(controller, action_dim=int(np.prod(environment.action_space.shape)))
        eef_id = robot.eef_site_id["right"]
        obj_model = raw.objects[object_id]
        body_name = getattr(obj_model, "root_body", f"{getattr(obj_model, 'naming_prefix', '')}object")
        body_id = raw.sim.model.body_name2id(body_name)
        signed_bottom_offset = float(np.asarray(getattr(obj_model, "bottom_offset", (0.0, 0.0, -.055)), dtype=float).reshape(-1)[-1])
        half_height = abs(signed_bottom_offset)
        eef_rotation = np.asarray(raw.sim.data.site_xmat[eef_id], dtype=float).reshape(3, 3)
        from robosuite.utils.transform_utils import mat2quat, quat2axisangle
        axis_angle = np.asarray(quat2axisangle(mat2quat(eef_rotation)), dtype=np.float32)
        # WholeBodyIK accepts an absolute torso target. Capture it once so a
        # changing observed joint position cannot ratchet the body while the
        # right-arm absolute pose is being tracked.
        torso_hold_qpos = np.asarray(controller.part_controllers["torso"].joint_pos, dtype=float).copy()
        gate = WholeBodyGraspGate()
        source_z = float(raw.sim.data.body_xpos[body_id][2])
        lift_pad_target: np.ndarray | None = None
        # Keep the approach primitive independently unit-testable; a real
        # deposit episode is required to provide an explicit target cavity
        # before the post-lift transport phase is entered.
        target_geometries = getattr(self.contract, "target_geometries", {})
        target = next(iter(target_geometries.values()), None)
        cavity_min = cavity_max = None
        if target is not None:
            cavity_min, cavity_max = np.asarray(target.min_corner, dtype=float), np.asarray(target.max_corner, dtype=float)
        deposit_gate: WholeBodyDepositGate | None = None
        last_open_can: np.ndarray | None = None

        for step in range(sum(WholeBodyGraspGate.LIMITS.values()) + 800):
            eef = np.asarray(raw.sim.data.site_xpos[eef_id], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
            pad_mid = pad_midpoint_world(raw)
            pad_in_side_band, pad_z = live_pad_side_band(raw, can_center_z=float(obj[2]), can_half_height=half_height)
            horizontal_distance = float(np.linalg.norm(pad_mid[:2] - obj[:2]))
            raw_grasp = bool(raw._check_grasp(robot.gripper["right"], obj_model))
            two_pad = selected_geom_has_two_finger_pad_contacts(raw, tuple(self.contract.object_geom_names.get(object_id, ())))
            previous = deposit_gate.state if deposit_gate is not None else gate.state
            if deposit_gate is None and gate.state != "lift":
                state = gate.observe(
                    horizontal_distance=horizontal_distance, pad_in_side_band=pad_in_side_band,
                    raw_grasp=raw_grasp, two_pad_contact=two_pad,
                )
            elif deposit_gate is None:
                state = gate.state
            else:
                pose_stable = last_open_can is not None and float(np.linalg.norm(obj - last_open_can)) <= .001
                state = deposit_gate.observe(raw_grasp=raw_grasp, two_pad_contact=two_pad, can_position=obj, pose_stable=pose_stable)
                if state in {"release", "settle"}:
                    last_open_can = obj.copy()
                if state == "reacquire":
                    self.trace.append({"object_id": object_id, "stage": state, "event": "lost_two_pad_pinch_during_transport"})
                    return False
                if deposit_gate.failed:
                    self.trace.append({"object_id": object_id, "stage": state, "event": "whole_body_deposit_stage_cap_exhausted", "stage_steps": deposit_gate.state_steps})
                    return False
            if gate.failed:
                self.trace.append({"object_id": object_id, "stage": state, "event": "whole_body_grasp_cap_exhausted"})
                return False
            if state == "lift" and previous != "lift":
                lift_pad_target = pad_mid + np.array((0.0, 0.0, self.LIFT_HEIGHT))
                self.trace.append({"object_id": object_id, "stage": "lift", "event": "two_pad_pinch_verified", "pinch_steps": gate.pinch_steps})
            if deposit_gate is None and state == "lift" and float(obj[2] - source_z) >= self.LIFT_PROOF:
                self.trace.append({"object_id": object_id, "stage": "lift", "event": "whole_body_lift_verified", "source_can_z": source_z, "final_can_z": float(obj[2]), "can_lift": float(obj[2] - source_z)})
                if cavity_min is None or cavity_max is None:
                    self.trace.append({"object_id": object_id, "stage": "lift", "event": "missing_target_cavity"})
                    return False
                deposit_gate = WholeBodyDepositGate(
                    cavity_min=cavity_min, cavity_max=cavity_max,
                    signed_bottom_offset=signed_bottom_offset, slot_xy=slot_xy,
                )
                state = deposit_gate.observe(raw_grasp=raw_grasp, two_pad_contact=two_pad, can_position=obj, pose_stable=False)
            if state == "approach_safe":
                desired_pad = obj + np.array((0.0, 0.0, self.SAFE_PAD_HEIGHT))
                closed = False
                eef_target = eef_target_for_pad_midpoint(
                    eef_world=eef, live_pad_midpoint=pad_mid, desired_pad_midpoint=desired_pad,
                )
                desired_can = None
            elif state in {"descend_to_side_band", "close"}:
                desired_pad = grasp_pad_target(obj, half_height)
                closed = state == "close"
                eef_target = eef_target_for_pad_midpoint(
                    eef_world=eef, live_pad_midpoint=pad_mid, desired_pad_midpoint=desired_pad,
                )
                desired_can = None
            elif state == "lift":
                desired_pad = np.asarray(lift_pad_target, dtype=float)
                closed = True
                eef_target = eef_target_for_pad_midpoint(
                    eef_world=eef, live_pad_midpoint=pad_mid, desired_pad_midpoint=desired_pad,
                )
                desired_can = None
            else:
                # Command 45 mm bottom clearance so the observed 40 mm
                # physical gate remains strict even with sub-millimetre IK
                # steady-state error.
                above_z = float(cavity_max[2]) + .045 - signed_bottom_offset
                inside_z = float(np.clip(float(cavity_max[2]) - .09, float(cavity_min[2]) + .06, float(cavity_max[2]) - .06))
                desired_can = obj.copy()
                if state == "raise_above_rim":
                    desired_can[2] = above_z
                elif state == "transit_above_cavity":
                    desired_can[:2] = slot_xy
                    desired_can[2] = above_z
                elif state == "lower_inside":
                    desired_can[:2] = slot_xy
                    desired_can[2] = inside_z
                closed = state not in {"release", "settle", "complete"}
                desired_pad = pad_mid.copy()
                eef_target = eef + desired_can - obj
            action = whole_body_pose_action(
                controller, layout, position=eef_target, axis_angle=axis_angle,
                torso_qpos=torso_hold_qpos, closed=closed,
            )
            self.trace.append({
                "object_id": object_id, "slot_xy": np.asarray(slot_xy, dtype=float).round(6).tolist(),
                "step": step + 1, "stage": state, "previous_stage": previous,
                "eef_world": eef.round(6).tolist(), "pad_midpoint_world": pad_mid.round(6).tolist(),
                "desired_pad_midpoint": np.asarray(desired_pad, dtype=float).round(6).tolist(),
                "can_world": obj.round(6).tolist(), "horizontal_distance": horizontal_distance,
                "source_can_z": source_z, "can_z": float(obj[2]),
                "desired_can_world": None if desired_can is None else desired_can.round(6).tolist(),
                "can_bottom": float(obj[2] + signed_bottom_offset),
                "cavity_rim_z": None if cavity_max is None else float(cavity_max[2]),
                "pad_z": list(pad_z), "pad_in_side_band": pad_in_side_band,
                "pad_height_difference": float(np.ptp(np.asarray(pad_z, dtype=float))) if len(pad_z) >= 2 else None,
                "raw_grasp": raw_grasp, "two_pad_contact": two_pad, "pinch_steps": gate.pinch_steps,
                "right_gripper_index": layout.right_gripper[0], "closed_command": closed,
                "action": action.round(6).tolist(),
            })
            yield "place" if state in {"release", "settle", "complete"} else "deposit", action
            if state == "complete":
                self.trace.append({"object_id": object_id, "slot_xy": np.asarray(slot_xy, dtype=float).round(6).tolist(), "stage": "settle", "event": "whole_body_deposit_complete", "final_can_z": float(obj[2])})
                return True

        self.trace.append({"object_id": object_id, "stage": "deposit", "event": "whole_body_cycle_cap_exhausted"})
        return False

    def _exit_open_bin(self, environment: Any, *, after_object_id: str) -> Iterable[tuple[str, np.ndarray]]:
        """Open-gripper vertical withdrawal after one stable release."""
        raw = _raw_environment(environment)
        robot = raw.robots[0]
        controller = robot.composite_controller
        layout = WholeBodyActionLayout.from_controller(controller, action_dim=int(np.prod(environment.action_space.shape)))
        eef_id = robot.eef_site_id["right"]
        target = next(iter(getattr(self.contract, "target_geometries", {}).values()), None)
        if target is None:
            self.trace.append({"object_id": after_object_id, "stage": "exit_bin", "event": "missing_target_cavity"})
            return False
        eef_rotation = np.asarray(raw.sim.data.site_xmat[eef_id], dtype=float).reshape(3, 3)
        from robosuite.utils.transform_utils import mat2quat, quat2axisangle
        axis_angle = np.asarray(quat2axisangle(mat2quat(eef_rotation)), dtype=np.float32)
        torso_hold_qpos = np.asarray(controller.part_controllers["torso"].joint_pos, dtype=float).copy()
        exit_z = float(np.asarray(target.max_corner, dtype=float)[2]) + .12
        for step in range(64):
            eef = np.asarray(raw.sim.data.site_xpos[eef_id], dtype=float)
            if float(eef[2]) >= exit_z - .003:
                self.trace.append({"object_id": after_object_id, "stage": "exit_bin", "event": "open_vertical_exit_complete", "eef_z": float(eef[2]), "required_eef_z": exit_z})
                return True
            eef_target = eef.copy()
            eef_target[2] = exit_z
            action = whole_body_pose_action(controller, layout, position=eef_target, axis_angle=axis_angle, torso_qpos=torso_hold_qpos, closed=False)
            self.trace.append({"object_id": after_object_id, "step": step + 1, "stage": "exit_bin", "eef_world": eef.round(6).tolist(), "desired_eef_z": exit_z, "closed_command": False, "action": action.round(6).tolist()})
            yield "deposit", action
        self.trace.append({"object_id": after_object_id, "stage": "exit_bin", "event": "open_vertical_exit_cap_exhausted", "required_eef_z": exit_z})
        return False


class MultiObjectDepositExpert:
    """Physical bounded three-can deposit controller with per-can traces."""

    APPROACH_HEIGHT = .13
    GRASP_HEIGHT = .022
    LIFT_CLEARANCE = .16
    RIM_CLEARANCE = .11
    RELEASE_Z_ABOVE_FLOOR = .085
    MAX_RECENTERS = 6

    def __init__(self, contract: Any):
        self.contract = contract
        self.trace: list[dict[str, Any]] = []

    def run(self, environment: Any) -> Iterable[tuple[str, np.ndarray]]:
        raw = _raw_environment(environment)
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        low, high = _action_bounds(environment)
        layout = ActionLayout.from_env(environment)
        eef_id = robot.eef_site_id["right"]
        target = next(iter(self.contract.target_geometries.values()))
        cavity_min, cavity_max = np.asarray(target.min_corner, dtype=float), np.asarray(target.max_corner, dtype=float)
        cavity_center = (cavity_min + cavity_max) / 2.0
        base_controller = robot.composite_controller.part_controllers["base"]

        # The fresh-scene +/-0.5 base probe is retained in the episode audit
        # generated on 2026-08-02. It coupled unexpectedly into 9--287mm EEF
        # Z motion, so it is explicitly rejected for manipulation episodes.
        # Do not repeat a known-unsafe calibration before grasping.
        base_jacobian = np.zeros((2, 2), dtype=float)
        base_calibrated = False
        base_navigation_enabled = False
        self.trace.append({
            "object_id": "obj", "stage": "base_calibration", "event": "base_control_disabled_after_unsafe_probe",
            "jacobian": base_jacobian.round(8).tolist(), "rank": 0,
        })

        def action_for(point: np.ndarray, *, closed: bool, limit: float) -> np.ndarray:
            eef = np.asarray(raw.sim.data.site_xpos[eef_id], dtype=float)
            error = np.asarray(controller.world_to_origin_frame(point), dtype=float) - np.asarray(controller.world_to_origin_frame(eef), dtype=float)
            action = np.zeros_like(low, dtype=np.float32)
            action[layout.arm[0]:layout.arm[0] + 3] = np.clip(error / .05, -limit, limit)
            action[layout.gripper] = 1.0 if closed else -1.0
            action[layout.base_mode] = -1.0
            return np.clip(action, low, high)

        def base_reach_action(object_position: np.ndarray) -> tuple[np.ndarray, float]:
            base_position, base_rotation = base_controller.get_base_pose()
            delta = np.asarray(object_position, dtype=float)[:2] - np.asarray(base_position, dtype=float)[:2]
            distance = float(np.linalg.norm(delta))
            action = np.zeros_like(low, dtype=np.float32)
            if distance > .30:
                world_step = delta / max(distance, 1e-6) * min(distance - .30, .12)
                local_step = np.asarray(base_rotation, dtype=float)[:2, :2].T @ world_step
                action[layout.base[0]:layout.base[0] + 2] = np.clip(local_step / .08, -.5, .5)
                action[layout.base_mode] = 1.0
            else:
                action[layout.base_mode] = -1.0
            return np.clip(action, low, high), distance

        for object_id in ("obj", "annotated_01", "annotated_02"):
            obj_model = raw.objects[object_id]
            body_name = getattr(obj_model, "root_body", f"{getattr(obj_model, 'naming_prefix', '')}object")
            body_id = raw.sim.model.body_name2id(body_name)
            machine, state_steps, recenters = DepositStateMachine(), 0, 0
            can_grasp = CanGraspController()
            transport_regrasp_attempts = 0
            completed = False
            # VLA82-001 begins with a physically level pad pair.  Retain that
            # fresh reset pose and let the XY probes measure it directly.
            wrist_prepared = True
            reach_xy_window: list[float] = []
            base_micro_align_steps = 0
            base_micro_align_used = False
            base_feedback_attempted = False
            base_feedback_steps = 0
            base_feedback_start_error: float | None = None
            transport_axis_sign = np.ones(2, dtype=float)
            transport_previous_error: list[float | None] = [None, None]
            transport_worsening_steps = np.zeros(2, dtype=int)
            transport_x_limit = .55
            held_probe_plan: list[tuple[int, float]] = [(axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
            held_probe_pending: dict[str, Any] | None = None
            held_probe_samples: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
            held_jacobian: np.ndarray | None = None
            held_feedback_steps = 0
            held_command_maximum = .60
            held_previous_error_norm: float | None = None
            held_error_window: list[float] = []
            held_refresh_count = 0
            grasp_wrist_sign = .6
            grasp_wrist_reset_steps = 0
            grasp_descent_probe_steps = 0
            grasp_descent_probe_start_z: float | None = None
            grasp_descent_probe_waiting = False
            grasp_wrist_scan_used = False
            grasp_pose_verified = True
            torso_probe_plan: list[float] = []
            torso_probe_steps = 0
            torso_probe_settle_steps = 0
            torso_probe_start_z: float | None = None
            torso_probe_start_qpos: float | None = None
            torso_probe_waiting = False
            torso_probe_results: list[tuple[float, float]] = []
            torso_down_sign: float | None = None
            xy_probe_plan: list[tuple[int, float]] = [(axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
            xy_probe_pending: dict[str, Any] | None = None
            xy_probe_samples: dict[int, dict[float, np.ndarray]] = {0: {}, 1: {}, 2: {}}
            xy_jacobian: np.ndarray | None = None
            xy_reprobe_count = 0
            xy_feedback_errors: list[float] = []
            xy_feedback_steps = 0
            xy_feedback_sign = 1.0
            grasp_orientation_settle_steps = 0
            source = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float).copy()
            can_bottom_offset = float(np.asarray(getattr(obj_model, "bottom_offset", (0.0, 0.0, .055)), dtype=float).reshape(-1)[-1])
            grasp_offset: np.ndarray | None = None
            grasp_stable_steps = 0
            contact_hold: np.ndarray | None = None
            maximum_steps = sum(DepositStateMachine.LIMITS.values()) + self.MAX_RECENTERS * DepositStateMachine.LIMITS["grasp"]
            for overall_step in range(maximum_steps):
                obj = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
                eef = np.asarray(raw.sim.data.site_xpos[eef_id], dtype=float)
                pad_in_side_band, pad_z = live_pad_side_band(raw, can_center_z=float(obj[2]), can_half_height=abs(can_bottom_offset))
                pad_band_lower = float(obj[2]) - abs(can_bottom_offset) + .012
                pad_band_upper = float(obj[2]) + abs(can_bottom_offset) - .012
                pad_level_required = needs_pad_leveling(pad_z, lower=pad_band_lower, upper=pad_band_upper)
                if xy_probe_pending is not None:
                    axis = int(xy_probe_pending["axis"])
                    sign = float(xy_probe_pending["sign"])
                    response = eef[:2] - np.asarray(xy_probe_pending["eef_xy"], dtype=float)
                    xy_probe_samples[axis][sign] = response
                    self.trace.append({
                        "object_id": object_id, "stage": "can_grasp_xy_probe", "event": "eef_xy_response",
                        "axis": axis, "sign": sign, "eef_xy_delta": response.round(8).tolist(),
                    })
                    xy_probe_pending = None
                grasped = bool(raw._check_grasp(robot.gripper["right"], obj_model))
                contact = _selected_geom_has_gripper_contact(raw, tuple(self.contract.object_geom_names.get(object_id, ())))
                two_pad = selected_geom_has_two_finger_pad_contacts(raw, tuple(self.contract.object_geom_names.get(object_id, ())))
                # A one-frame raw grasp is only an impact transient. Require
                # three post-step observations while the wrist is held before
                # allowing lift; otherwise the prior controller immediately
                # drove the fingertip target through the seated can.
                grasp_stable_steps = grasp_stable_steps + 1 if deposit_grasp_verified(raw_grasp=grasped, two_pad_contact=two_pad) else 0
                stable_grasp = grasp_stable_steps >= 3
                if machine.state == "grasp" and can_grasp.state == "legacy" and grasp_descent_probe_waiting:
                    dz = float(eef[2] - float(grasp_descent_probe_start_z))
                    self.trace.append({
                        "object_id": object_id, "stage": "grasp_wrist_probe", "event": "local_z_world_response",
                        "wrist_sign": grasp_wrist_sign, "world_dz": dz,
                        "eef_rotation": np.asarray(raw.sim.data.site_xmat[eef_id], dtype=float).reshape(3, 3).round(6).tolist(),
                    })
                    grasp_descent_probe_waiting = False
                    if dz <= -.002:
                        grasp_pose_verified = True
                    elif not grasp_wrist_scan_used:
                        grasp_wrist_scan_used = True
                        grasp_wrist_sign = -.6
                        grasp_wrist_reset_steps = 20
                    else:
                        self.trace.append({"object_id": object_id, "stage": "grasp_wrist_probe", "event": "no_descending_wrist_pose"})
                        # The torso controller is a delta-position controller.
                        # Its full public command gives a measurable response;
                        # the calibrated negative direction is tested first.
                        torso_probe_plan = [-1.0, 1.0]
                if machine.state == "grasp" and can_grasp.state == "legacy" and torso_probe_waiting:
                    torso_controller = robot.composite_controller.part_controllers["torso"]
                    dz = float(eef[2] - float(torso_probe_start_z))
                    dq = float(torso_controller.joint_pos[0] - float(torso_probe_start_qpos))
                    torso_probe_results.append((float(torso_down_sign), dz))
                    self.trace.append({
                        "object_id": object_id, "stage": "torso_probe", "event": "torso_world_response",
                        "torso_action": torso_down_sign, "torso_qpos_delta": dq, "eef_world_dz": dz,
                        "torso_qpos": float(torso_controller.joint_pos[0]),
                    })
                    torso_probe_waiting = False
                    torso_probe_steps = 0
                    if dz <= -.005:
                        torso_probe_plan = []
                        grasp_pose_verified = True
                        self.trace.append({
                            "object_id": object_id, "stage": "torso_probe", "event": "torso_descending_direction_selected",
                            "torso_action": torso_down_sign, "eef_world_dz": dz,
                        })
                    elif not torso_probe_plan:
                        self.trace.append({"object_id": object_id, "stage": "torso_probe", "event": "torso_no_descending_direction"})
                        break
                # Read the response of the *previous* closed-gripper probe
                # only while the physical two-pad gate still holds.  No state
                # is written; this is observed after a public env.step.
                if held_probe_pending is not None:
                    if deposit_grasp_verified(raw_grasp=grasped, two_pad_contact=two_pad):
                        axis = int(held_probe_pending["axis"])
                        sign = float(held_probe_pending["sign"])
                        delta = obj - np.asarray(held_probe_pending["can_world"], dtype=float)
                        held_probe_samples[axis].append(delta / sign)
                        self.trace.append({
                            "object_id": object_id, "stage": "held_jacobian_probe", "event": "held_probe_response",
                            "axis": axis, "sign": sign, "can_delta": delta.round(8).tolist(),
                            "two_pad": True,
                        })
                    else:
                        self.trace.append({"object_id": object_id, "stage": "held_jacobian_probe", "event": "held_probe_gate_lost"})
                    held_probe_pending = None
                grasp_target = obj + np.array((0., 0., self.GRASP_HEIGHT))
                # The reach guard is intentionally measured against its own
                # overhead waypoint, not the later fingertip seating target.
                # Comparing with the 2.2-cm grasp target kept the controller
                # forever 10.8cm away after a correct 13cm approach.
                reach_target = obj + np.array((0., 0., self.APPROACH_HEIGHT))
                distance = float(np.linalg.norm(np.asarray(controller.world_to_origin_frame(reach_target)) - np.asarray(controller.world_to_origin_frame(eef))))
                horizontal_distance = float(np.linalg.norm(obj[:2] - eef[:2]))
                cavity_center_distance = float(np.linalg.norm(obj[:2] - cavity_center[:2]))
                old_state = machine.state
                if old_state == "grasp":
                    old_grasp_substate = can_grasp.state
                    can_grasp.observe(
                        orientation_complete=wrist_prepared and grasp_orientation_settle_steps <= 0,
                        horizontal_distance=horizontal_distance,
                        vertical_gap=float(eef[2] - obj[2]),
                        raw_grasp=grasped,
                        two_pad_contact=two_pad,
                        eef_z=float(eef[2]),
                        pad_in_side_band=pad_in_side_band,
                        pad_z=pad_z,
                        pad_level_required=pad_level_required,
                    )
                    if old_grasp_substate == "torso_descend" and can_grasp.last_torso_world_dz is not None:
                        self.trace.append({
                            "object_id": object_id, "stage": "can_grasp", "event": "torso_pulse_response",
                            "world_dz": can_grasp.last_torso_world_dz, "pulses": can_grasp.torso_pulses,
                        })
                        xy_feedback_errors = []
                        xy_feedback_steps = 0
                        can_grasp.last_torso_world_dz = None
                    if can_grasp.failed:
                        self.trace.append({"object_id": object_id, "stage": "can_grasp", "event": "progress_cap_exhausted", "substate": can_grasp.state})
                        break
                progression_grasp = can_grasp.ready_for_deposit if old_state == "grasp" else stable_grasp
                state = machine.advance(distance=distance, horizontal_distance=horizontal_distance, cavity_center_distance=cavity_center_distance, object_bottom_offset=can_bottom_offset, grasped=progression_grasp, can_position=obj, cavity_min=cavity_min, cavity_max=cavity_max)
                if state == "reach":
                    reach_xy_window.append(horizontal_distance)
                    reach_xy_window = reach_xy_window[-11:]
                    # After the wrist has been prepared, distinguish a true
                    # translational stall from ordinary approach convergence.
                    # One bounded base correction is enough to recenter the
                    # arm's reachable workspace; all following commands
                    # return to normal arm mode and remeasure world XY.
                    if (
                        base_navigation_enabled and wrist_prepared and not base_micro_align_used and len(reach_xy_window) == 11
                        and horizontal_distance > .005 and reach_xy_window[0] - reach_xy_window[-1] < .001
                    ):
                        base_micro_align_used = True
                        base_micro_align_steps = 4
                        self.trace.append({
                            "object_id": object_id, "stage": "reach", "event": "base_micro_align_begin",
                            "horizontal_distance": horizontal_distance,
                            "window_improvement": reach_xy_window[0] - reach_xy_window[-1],
                        })
                else:
                    reach_xy_window.clear()
                if state != old_state:
                    state_steps = 0
                    if state == "lift":
                        grasp_offset = eef - obj
                        held_probe_plan = [(axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
                        held_probe_pending = None
                        held_probe_samples = {0: [], 1: [], 2: []}
                        held_jacobian = None
                        held_feedback_steps = 0
                        held_previous_error_norm = None
                        held_error_window.clear()
                        held_refresh_count = 0
                    elif state == "grasp":
                        grasp_orientation_settle_steps = 0
                        grasp_wrist_sign = 0.0
                        grasp_wrist_reset_steps = 0
                        grasp_descent_probe_steps = 0
                        grasp_descent_probe_waiting = False
                        grasp_wrist_scan_used = False
                        grasp_pose_verified = True
                        torso_probe_plan = []
                        torso_probe_steps = 0
                        torso_probe_settle_steps = 0
                        torso_probe_waiting = False
                        torso_probe_results = []
                        torso_down_sign = None
                    elif state == "reach" and old_state in {"lift", "above_cavity", "descend"}:
                        # Lost-pinch recovery is tied to the *live* can pose,
                        # never the stale source waypoint.  Reapply the
                        # verified wrist pre-rotation before its bounded
                        # world-XYZ approach and grasp seek.
                        transport_regrasp_attempts += 1
                        self.trace.append({
                            "object_id": object_id, "stage": old_state,
                            "event": "lost_grasp_recover_to_live_pose",
                            "attempt": transport_regrasp_attempts,
                            "can_world": obj.round(6).tolist(),
                        })
                        if transport_regrasp_attempts > 2:
                            return
                        grasp_offset = None
                        contact_hold = None
                        grasp_stable_steps = 0
                        wrist_prepared = True
                if state == "grasp" and state_steps > 0 and horizontal_distance > .030:
                    # Recenter the dedicated grasp controller only.  Never
                    # restart wrist orientation or return to the global reach
                    # state, which used to create an orientation/recenter loop.
                    recenters += 1
                    self.trace.append({"object_id": object_id, "stage": "can_grasp", "event": "recenter_after_xy_drift", "horizontal_distance": horizontal_distance, "attempt": recenters})
                    if recenters > self.MAX_RECENTERS:
                        break
                    can_grasp.recenter()
                grasp_override_action: np.ndarray | None = None
                if state == "reach":
                    if not wrist_prepared and state_steps < 20:
                        # Keep the freshly horizontal pad plane; the legacy
                        # axis-5 pre-rotation is explicitly prohibited here.
                        action, desired, closed, limit = vla82_001_pregrasp_action(low, high, layout=layout), None, False, .0
                    else:
                        wrist_prepared = True
                        base_action, base_distance = base_reach_action(obj)
                        # ObjectPlayEnv's PandaOmron base pose is a world-frame
                        # calibration reference, not a direct source-object
                        # reach metric.  Only enable mobile transit beyond the
                        # empirically reachable arm radius; otherwise it drives
                        # the overhead EEF goal away while the arm could reach.
                        if base_navigation_enabled and base_micro_align_steps > 0:
                            _, base_rotation = base_controller.get_base_pose()
                            action = deposit_base_micro_align_action(
                                low, high, layout=layout, eef_world=eef, can_world=obj, base_rotation=np.asarray(base_rotation),
                            )
                            base_micro_align_steps -= 1
                            desired, closed, limit = None, False, .0
                        elif base_navigation_enabled and base_feedback_steps > 0:
                            command = base_feedback_command(base_jacobian, obj[:2] - eef[:2])
                            action = np.zeros_like(low, dtype=np.float32)
                            action[layout.base[0]:layout.base[0] + 2] = command
                            action[layout.gripper] = -1.0
                            action[layout.base_mode] = 1.0
                            base_feedback_steps -= 1
                            desired, closed, limit = None, False, .0
                            if base_feedback_steps == 0 and base_feedback_start_error is not None:
                                improved = base_feedback_start_error - horizontal_distance
                                self.trace.append({
                                    "object_id": object_id, "stage": state, "event": "base_feedback_window",
                                    "start_error": base_feedback_start_error, "end_error": horizontal_distance,
                                    "improvement": improved,
                                })
                        elif (base_navigation_enabled and not base_feedback_attempted and base_calibrated
                              and deposit_requires_base_transit(horizontal_distance=horizontal_distance, base_distance=base_distance)):
                            base_feedback_attempted = True
                            base_feedback_steps = 5
                            base_feedback_start_error = horizontal_distance
                            command = base_feedback_command(base_jacobian, obj[:2] - eef[:2])
                            action = np.zeros_like(low, dtype=np.float32)
                            action[layout.base[0]:layout.base[0] + 2] = command
                            action[layout.gripper] = -1.0
                            action[layout.base_mode] = 1.0
                            base_feedback_steps -= 1
                            desired, closed, limit = None, False, .0
                        elif horizontal_distance <= .045 and eef[2] > reach_target[2] + .015:
                            # In the calibrated wrist posture, the direct
                            # local-Z public action is the verified way to
                            # close the overhead approach gap before pinch.
                            action = deposit_vertical_descend_action(low, high, layout=layout)
                            desired, closed, limit = None, False, .0
                        else:
                            desired, closed, limit = obj + np.array((0., 0., self.APPROACH_HEIGHT)), False, .70
                elif state == "grasp":
                    # CanGraspController is the sole producer of physical
                    # grasp actions.  The legacy diagnostics below may still
                    # observe contact, but cannot overwrite this public action
                    # at the final dispatch point.
                    if can_grasp.state == "init_orientation":
                        # The wrist was oriented in the global reach state;
                        # wait for its actual dynamics to settle before taking
                        # any Jacobian sample.  Do not rotate it a second time.
                        grasp_override_action = torso_probe_action(low, high, layout=layout, command=0.0)
                        grasp_orientation_settle_steps = max(0, grasp_orientation_settle_steps - 1)
                    elif can_grasp.state in {"xy_center", "final_xy_refine"}:
                        if can_grasp.requires_axis2_pad_band:
                            # The fallback's safety contract forbids a later
                            # XY correction from undoing the measured pad
                            # descent.  Discard any stale axis-2 probe left
                            # from the pre-saturation calibration.
                            xy_probe_plan = [(axis, sign) for axis, sign in xy_probe_plan if axis < 2]
                        if xy_probe_plan:
                            axis, sign = xy_probe_plan.pop(0)
                            grasp_override_action = torso_probe_action(low, high, layout=layout, command=0.0)
                            grasp_override_action[layout.arm[0] + axis] = .20 * sign
                            xy_probe_pending = {"axis": axis, "sign": sign, "eef_xy": eef[:2].copy()}
                        elif xy_jacobian is None:
                            xy_jacobian = estimate_eef_xy_jacobian(xy_probe_samples, amplitude=.20)
                            rank = int(np.linalg.matrix_rank(xy_jacobian, tol=1e-5))
                            condition = float(np.linalg.cond(xy_jacobian)) if rank == 2 else float("inf")
                            self.trace.append({
                                "object_id": object_id, "stage": "can_grasp_xy_probe",
                                "event": "jacobian_ready" if rank == 2 and condition < 1e4 else "jacobian_invalid",
                                "jacobian": xy_jacobian.round(8).tolist(), "rank": rank, "condition": condition,
                            })
                            if rank < 2 or condition >= 1e4:
                                if xy_reprobe_count >= 2:
                                    can_grasp.failed = True
                                else:
                                    xy_reprobe_count += 1
                                    axis_count = 2 if can_grasp.requires_axis2_pad_band else (3 if can_grasp.state == "final_xy_refine" or can_grasp.torso_pulses == 0 else 2)
                                    xy_probe_plan = [(axis, sign) for axis in range(axis_count) for sign in (1.0, -1.0)]
                                    xy_probe_samples = {axis: {} for axis in range(axis_count)}
                                    xy_jacobian = None
                            grasp_override_action = torso_probe_action(low, high, layout=layout, command=0.0)
                        else:
                            xy_error = obj[:2] - eef[:2]
                            error_norm = float(np.linalg.norm(xy_error))
                            xy_feedback_steps += 1
                            xy_feedback_errors.append(error_norm)
                            xy_feedback_errors = xy_feedback_errors[-12:]
                            if len(xy_feedback_errors) >= 4 and xy_feedback_errors[-4] - xy_feedback_errors[-1] < -.001:
                                xy_feedback_sign *= -1.0
                                xy_feedback_errors = [error_norm]
                                self.trace.append({"object_id": object_id, "stage": "can_grasp_xy_probe", "event": "feedback_sign_reversed", "error": error_norm, "sign": xy_feedback_sign})
                            twelve_step_improvement = xy_feedback_errors[0] - xy_feedback_errors[-1] if len(xy_feedback_errors) >= 12 else None
                            if xy_feedback_steps >= 24 and twelve_step_improvement is not None and twelve_step_improvement < .0005 and xy_reprobe_count < 1:
                                xy_reprobe_count += 1
                                axis_count = 2 if can_grasp.requires_axis2_pad_band else (3 if can_grasp.state == "final_xy_refine" or can_grasp.torso_pulses == 0 else 2)
                                xy_probe_plan = [(axis, sign) for axis in range(axis_count) for sign in (1.0, -1.0)]
                                xy_probe_samples = {axis: {} for axis in range(axis_count)}
                                xy_jacobian = None
                                xy_feedback_errors = []
                                xy_feedback_steps = 0
                                self.trace.append({"object_id": object_id, "stage": "can_grasp_xy_probe", "event": "feedback_stalled_reprobe", "error": error_norm, "improvement": twelve_step_improvement, "attempt": xy_reprobe_count})
                                grasp_override_action = torso_probe_action(low, high, layout=layout, command=0.0)
                            else:
                                grasp_override_action = torso_probe_action(low, high, layout=layout, command=0.0)
                                feedback_jacobian = can_grasp_xy_feedback_jacobian(
                                    xy_jacobian[:, :2] if can_grasp.state != "final_xy_refine" and can_grasp.torso_pulses > 0 else xy_jacobian,
                                    requires_axis2_pad_band=can_grasp.requires_axis2_pad_band,
                                )
                                command = xy_feedback_sign * eef_xy_damped_command(feedback_jacobian, xy_error, maximum=.20)
                                grasp_override_action[layout.arm[0]:layout.arm[0] + command.size] = command
                    elif can_grasp.state == "torso_descend":
                        can_grasp.begin_torso_pulse(float(eef[2]))
                        if can_grasp.state == "axis2_pad_descend":
                            grasp_override_action = axis2_pad_descent_action(low, high, layout=layout)
                            can_grasp.begin_axis2_pulse(pad_z)
                        else:
                            grasp_override_action = torso_probe_action(low, high, layout=layout, command=-1.0)
                    elif can_grasp.state == "axis2_pad_descend":
                        grasp_override_action = axis2_pad_descent_action(low, high, layout=layout)
                        can_grasp.begin_axis2_pulse(pad_z)
                    elif can_grasp.state == "level_pads":
                        grasp_override_action = level_pads_action(low, high, layout=layout)
                    elif can_grasp.state == "close":
                        grasp_override_action = torso_probe_action(low, high, layout=layout, command=0.0)
                        grasp_override_action[layout.gripper] = 1.0
                    elif can_grasp.state == "torso_lift":
                        xy_target = eef.copy()
                        xy_target[:2] = obj[:2]
                        xy_hold = action_for(xy_target, closed=True, limit=.08)
                        grasp_override_action = torso_probe_action(low, high, layout=layout, command=1.0)
                        grasp_override_action[layout.arm[0]:layout.arm[0] + 2] = xy_hold[layout.arm[0]:layout.arm[0] + 2]
                        grasp_override_action[layout.gripper] = 1.0
                    direct_grasp_action = False
                    if torso_probe_plan and torso_probe_steps == 0 and torso_probe_settle_steps == 0 and not torso_probe_waiting:
                        torso_down_sign = torso_probe_plan.pop(0)
                        torso_probe_steps = 1
                        torso_probe_start_z = float(eef[2])
                        torso_probe_start_qpos = float(robot.composite_controller.part_controllers["torso"].joint_pos[0])
                    if torso_probe_steps > 0:
                        action = torso_probe_action(low, high, layout=layout, command=float(torso_down_sign))
                        direct_grasp_action = True
                        torso_probe_steps -= 1
                        if torso_probe_steps == 0:
                            torso_probe_settle_steps = 6
                        desired, closed, limit = None, False, .0
                    elif torso_probe_settle_steps > 0:
                        # Let the position controller converge to the command
                        # before measuring it. A zero delta retains its goal.
                        action = torso_probe_action(low, high, layout=layout, command=0.0)
                        direct_grasp_action = True
                        torso_probe_settle_steps -= 1
                        if torso_probe_settle_steps == 0:
                            torso_probe_waiting = True
                        desired, closed, limit = None, False, .0
                    elif grasp_pose_verified and torso_down_sign is not None and eef[2] > obj[2] + .065:
                        action = torso_probe_action(low, high, layout=layout, command=float(torso_down_sign))
                        direct_grasp_action = True
                        desired, closed, limit = None, False, .0
                    elif grasp_wrist_reset_steps > 0:
                        action = np.zeros_like(low, dtype=np.float32)
                        action[layout.arm[0] + 5] = grasp_wrist_sign
                        action[layout.gripper] = -1.0
                        action[layout.base_mode] = -1.0
                        direct_grasp_action = True
                        if grasp_wrist_reset_steps == 20:
                            self.trace.append({
                                "object_id": object_id, "stage": "grasp_wrist_probe", "event": "wrist_reset_begin",
                                "wrist_sign": grasp_wrist_sign, "eef_world": eef.round(6).tolist(),
                                "eef_rotation": np.asarray(raw.sim.data.site_xmat[eef_id], dtype=float).reshape(3, 3).round(6).tolist(),
                            })
                        grasp_wrist_reset_steps -= 1
                        if grasp_wrist_reset_steps == 0:
                            grasp_descent_probe_steps = 5
                        desired, closed, limit = None, False, .0
                    elif grasp_descent_probe_steps > 0:
                        if grasp_descent_probe_start_z is None:
                            grasp_descent_probe_start_z = float(eef[2])
                        action = deposit_vertical_descend_action(low, high, layout=layout)
                        direct_grasp_action = True
                        grasp_descent_probe_steps -= 1
                        if grasp_descent_probe_steps == 0:
                            grasp_descent_probe_waiting = True
                        desired, closed, limit = None, False, .0
                    elif not grasp_pose_verified:
                        action = np.zeros_like(low, dtype=np.float32)
                        action[layout.gripper] = -1.0
                        action[layout.base_mode] = -1.0
                        direct_grasp_action = True
                        desired, closed, limit = None, False, .0
                    elif two_pad:
                        contact_hold = eef.copy()
                    vertical_gap = float(eef[2] - obj[2])
                    if contact_hold is not None:
                        desired, closed, limit = contact_hold, True, .20
                    elif should_issue_grasp_vertical_descent(
                        direct_action_active=direct_grasp_action,
                        contact_hold_active=False,
                        vertical_gap=vertical_gap,
                    ):
                        # A feedback world waypoint is more robust than
                        # repeating an open-loop controller-frame vector:
                        # it requests just 1cm down from the *measured* EEF
                        # while re-closing any observed XY error each step.
                        action = deposit_vertical_descend_action(low, high, layout=layout)
                        desired, closed, limit = None, False, .0
                    elif not direct_grasp_action:
                        # Close only while re-registering the measured can
                        # centre; holding the offset EEF produced one-pad
                        # contact and must not be promoted to a grasp.
                        desired = grasp_closure_target(eef, obj)
                        closed, limit = True, .35
                elif state == "lift":
                    desired, closed, limit = source + (grasp_offset if grasp_offset is not None else 0.) + np.array((0., 0., self.LIFT_CLEARANCE)), True, .55
                elif state == "above_cavity":
                    # Close the loop on the measured can centre on every
                    # public control step.  A saved lift-time wrist offset
                    # drifts during a long transport and can falsely bring
                    # only the EEF near the cavity while leaving the can out.
                    xy_inside = DepositStateMachine._xy_inside(obj, cavity_min, cavity_max)
                    xy_error = cavity_center[:2] - obj[:2]
                    if held_probe_plan:
                        axis, sign = held_probe_plan.pop(0)
                        action = np.zeros_like(low, dtype=np.float32)
                        action[layout.arm[0] + axis] = .08 * sign
                        action[layout.gripper] = 1.0
                        action[layout.base_mode] = -1.0
                        held_probe_pending = {"axis": axis, "sign": sign, "can_world": obj.copy()}
                        desired, closed, limit = None, True, .0
                        active_transport_axis: int | None = None
                    elif held_jacobian is None:
                        held_jacobian = estimate_held_can_jacobian(held_probe_samples, amplitude=.08)
                        rank = int(np.linalg.matrix_rank(held_jacobian, tol=1e-5))
                        condition = float(np.linalg.cond(held_jacobian)) if rank == 3 else float("inf")
                        self.trace.append({
                            "object_id": object_id, "stage": "held_jacobian_probe",
                            "event": "held_jacobian_ready" if rank == 3 and condition < 1e4 else "held_jacobian_invalid",
                            "jacobian": held_jacobian.round(8).tolist(), "rank": rank, "condition": condition,
                        })
                        if rank < 3 or condition >= 1e4:
                            break
                        action = np.zeros_like(low, dtype=np.float32)
                        action[layout.gripper] = 1.0
                        action[layout.base_mode] = -1.0
                        desired, closed, limit = None, True, .0
                        active_transport_axis = None
                    else:
                        desired_can_z = float(cavity_max[2]) + self.RIM_CLEARANCE - can_bottom_offset
                        desired_can_delta = np.array((xy_error[0], xy_error[1], 0.0 if not xy_inside else desired_can_z - obj[2]))
                        error_norm = float(np.linalg.norm(desired_can_delta))
                        if held_previous_error_norm is not None and error_norm > held_previous_error_norm + .001:
                            held_command_maximum = min(held_command_maximum, .30)
                            self.trace.append({
                                "object_id": object_id, "stage": state, "event": "held_transport_speed_reduced",
                                "previous_error_norm": held_previous_error_norm, "error_norm": error_norm,
                            })
                        held_previous_error_norm = error_norm
                        held_error_window.append(error_norm)
                        held_error_window = held_error_window[-10:]
                        command = held_can_damped_command(held_jacobian, desired_can_delta, maximum=held_command_maximum)
                        action = np.zeros_like(low, dtype=np.float32)
                        action[layout.arm[0]:layout.arm[0] + 3] = command
                        action[layout.gripper] = 1.0
                        action[layout.base_mode] = -1.0
                        held_feedback_steps += 1
                        if held_feedback_steps % 20 == 0:
                            self.trace.append({
                                "object_id": object_id, "stage": state, "event": "held_transport_progress",
                                "feedback_steps": held_feedback_steps, "can_cavity_error": desired_can_delta.round(8).tolist(),
                                "error_norm": error_norm,
                            })
                        # Keep the first verified local response model for a
                        # substantial transit window.  Reprobing every 20
                        # steps consumed the finite deposit budget before a
                        # 19cm transfer could make material progress.
                        stalled = len(held_error_window) == 10 and held_error_window[0] - held_error_window[-1] < .002
                        if (held_feedback_steps >= 80 or stalled) and held_refresh_count < 2 and not xy_inside:
                            held_probe_plan = [(axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
                            held_probe_samples = {0: [], 1: [], 2: []}
                            held_jacobian = None
                            held_feedback_steps = 0
                            held_error_window.clear()
                            held_refresh_count += 1
                            self.trace.append({
                                "object_id": object_id, "stage": state, "event": "held_jacobian_periodic_refresh",
                                "refresh": held_refresh_count, "reason": "stalled" if stalled else "80_step_window",
                            })
                        desired, closed, limit = None, True, .0
                        active_transport_axis = None
                    # Same-Y placement is a long X transfer.  Drive only the
                    # live can X error strongly; Y/Z are held at their live
                    # values so no three-axis saturated command can eject a
                    # verified two-pad grasp.
                    if desired is not None and abs(float(xy_error[0])) > .012:
                        previous_x = transport_previous_error[0]
                        current_x = abs(float(xy_error[0]))
                        if previous_x is not None and current_x > previous_x + .0005 and transport_x_limit > .20:
                            transport_x_limit = .20
                            self.trace.append({
                                "object_id": object_id, "stage": state, "event": "transport_x_speed_reduced",
                                "previous_abs_error": previous_x, "current_abs_error": current_x,
                            })
                        transport_previous_error[0] = current_x
                        desired = eef.copy()
                        desired[0] += xy_error[0]
                        active_transport_axis: int | None = 0
                        closed, limit = True, transport_x_limit
                    elif desired is not None and xy_inside:
                        desired = eef.copy()
                        desired_can_z = float(cavity_max[2]) + self.RIM_CLEARANCE - can_bottom_offset
                        desired[2] += desired_can_z - obj[2]
                        active_transport_axis: int | None = None
                        closed, limit = True, .20
                    elif desired is not None:
                        active_transport_axis = 1
                        previous = transport_previous_error[active_transport_axis]
                        if previous is not None:
                            sign, worsening = update_deposit_axis_calibration(
                                previous_abs_error=previous,
                                current_abs_error=abs(float(xy_error[active_transport_axis])),
                                sign=float(transport_axis_sign[active_transport_axis]),
                                worsening_steps=int(transport_worsening_steps[active_transport_axis]),
                            )
                            if sign != transport_axis_sign[active_transport_axis]:
                                self.trace.append({
                                    "object_id": object_id, "stage": state, "event": "transport_axis_sign_reversed",
                                    "axis": active_transport_axis, "previous_abs_error": previous,
                                    "current_abs_error": abs(float(xy_error[active_transport_axis])),
                                })
                            transport_axis_sign[active_transport_axis] = sign
                            transport_worsening_steps[active_transport_axis] = worsening
                        transport_previous_error[active_transport_axis] = abs(float(xy_error[active_transport_axis]))
                        desired = eef.copy()
                        desired[1] += xy_error[1]
                        closed, limit = True, .20
                elif state == "descend":
                    # The far-cavity trial showed that an open-loop local-Z
                    # action loses horizontal registration over a long drop.
                    # Recompute this 1-cm measured-world waypoint every step
                    # so the OSC corrects XY while retaining the two-pad hold.
                    desired = eef + np.array((0., 0., -.01))
                    closed, limit = True, .25
                else:
                    desired, closed, limit = eef, False, .18
                if grasp_override_action is not None:
                    action = grasp_override_action
                elif desired is not None:
                    action = action_for(np.asarray(desired, dtype=float), closed=closed, limit=limit)
                    if state == "above_cavity" and active_transport_axis is not None:
                        action[layout.arm[0] + active_transport_axis] *= transport_axis_sign[active_transport_axis]
                self.trace.append({
                    "object_id": object_id, "step": overall_step + 1, "stage": state,
                    "stage_step": state_steps, "stage_limit": machine.limit_for(state),
                    "eef_world": eef.round(6).tolist(), "can_world": obj.round(6).tolist(),
                    "cavity_min": cavity_min.round(6).tolist(), "cavity_max": cavity_max.round(6).tolist(),
                    "distance": distance, "horizontal_distance": horizontal_distance,
                    "grasped": grasped, "stable_grasp_steps": grasp_stable_steps,
                    "pad_z": list(pad_z), "pad_in_side_band": pad_in_side_band,
                    "pad_height_difference": float(np.ptp(np.asarray(pad_z, dtype=float))) if len(pad_z) >= 2 else None,
                    "pad_level_required": pad_level_required,
                    "exact_gripper_contact": contact, "two_pad_contact": two_pad,
                    "horizontal_projection_inside": DepositStateMachine._xy_inside(obj, cavity_min, cavity_max),
                    "rim_clearance": float(obj[2] + can_bottom_offset - cavity_max[2]),
                    "action": action.round(6).tolist(),
                })
                yield "place" if state in {"release", "settle"} else "deposit", action
                state_steps += 1
                if state == "settle" and state_steps >= machine.limit_for("settle"):
                    completed = True
                    break
                if state_steps >= machine.limit_for(state):
                    self.trace.append({
                        "object_id": object_id, "stage": state,
                        "event": "stage_limit_exhausted", "stage_limit": machine.limit_for(state),
                    })
                    break
            # A three-can source episode must not hide a first-can failure
            # behind later attempts.  Stop at the bounded failing object and
            # preserve its diagnostic trace for the next root-cause pass.
            if not completed:
                return


def deposit_expert_class_for_controller(selection_id: str, controller: Any) -> type[WholeBodyDepositExpert] | type[MultiObjectDepositExpert]:
    """Select WholeBodyIK only for the source-matched trash-can deposit."""
    if str(selection_id) == "VLA82-001" and hasattr(controller, "_whole_body_controller_action_split_indexes"):
        return WholeBodyDepositExpert
    return MultiObjectDepositExpert


def pick_place_execution_phases(phases: Sequence[str]) -> tuple[tuple[str, str], str] | None:
    """Map a single visible placement/insert label to real pick-place motion."""
    source = tuple(str(phase) for phase in phases)
    # The source recorder can emit adjacent copies of the same visible state
    # while the hand is stationary (for example ``place, place``).  They are
    # evidence of one placement, not two independent robot tasks.  Normalise
    # only adjacent duplicates; distinct source phases retain their order.
    normalized = tuple(
        phase for index, phase in enumerate(source)
        if index == 0 or phase != source[index - 1]
    )
    if normalized in {("place",), ("grasp", "place")}:
        return ("grasp", "place"), "place"
    if normalized == ("insert",):
        return ("grasp", "place"), "insert"
    if normalized == ("stack",):
        return ("grasp", "place"), "stack"
    return None


def pick_place_evaluation_spec_for_scene(
    spec: OperationSpec, scene: SceneRequest | Any,
) -> OperationSpec:
    """Retain the authoritative operation semantics for physical evaluation."""
    return spec


class CompositeExpert:
    """Ordered source phase executor; no terminal-state helper exists here."""

    def __init__(self, phases: Sequence[str], contract: Any | None = None):
        self.phases = tuple(phases)
        self.contract = contract
        self.controller_trace: list[dict[str, Any]] = []

    def run(self, environment: Any) -> Iterable[tuple[str, np.ndarray]]:
        if getattr(getattr(environment, "request", None), "selection_id", "") == "VLA82-001" and self.contract is not None:
            raw = _raw_environment(environment)
            controller = raw.robots[0].composite_controller
            primitive = deposit_expert_class_for_controller("VLA82-001", controller)(self.contract)
            yield from primitive.run(environment)
            self.controller_trace.extend(primitive.trace)
            return
        if self.contract is not None and set(self.phases) & {"wipe", "scrub"}:
            primitive = CleaningPrimitive(self.phases, self.contract)
            for phase, action in primitive.actions(environment):
                yield phase, action
            self.controller_trace.extend(primitive.trace)
            return
        if self.contract is not None and self.phases == ("close",):
            primitive = DrawerClosePrimitive(self.contract)
            for action in primitive.actions(environment):
                yield "close", action
            self.controller_trace.extend(primitive.trace)
            return
        if self.contract is not None and self.phases == ("open",):
            primitive = FixtureOpenPrimitive(self.contract)
            for action in primitive.actions(environment):
                yield "open", action
            self.controller_trace.extend(primitive.trace)
            return
        if self.contract is not None and self.phases == ("push",):
            primitive = RackPushPrimitive(self.contract)
            for action in primitive.actions(environment):
                yield "push", action
            self.controller_trace.extend(primitive.trace)
            return
        # A source annotation may record only the visible terminal phase
        # (``("place",)``) even though the robot must still perform the
        # complete grasp→lift→transfer→release sequence.  Route every
        # ordinary pick/place contract through the same physics-checked
        # controller; falling back to PrimitiveExpert here emits only a few
        # low-amplitude perturbations and can never create contact evidence.
        execution = pick_place_execution_phases(self.phases)
        if self.contract is not None and execution is not None:
            controller_phases, completion_phase = execution
            primitive = PickPlaceExpert(
                controller_phases,
                self.contract,
            )
            for phase, action in primitive.actions(environment):
                yield completion_phase if phase == "place" else phase, action
            self.controller_trace.extend(primitive.trace)
            return
        for index, phase in enumerate(self.phases):
            primitive: Any = KnobPrimitive(self.contract) if phase == "turn_knob" and self.contract is not None else PrimitiveExpert(phase, index)
            for action in primitive.actions(environment):
                yield phase, action
            if isinstance(primitive, KnobPrimitive):
                self.controller_trace.extend(primitive.trace)


def _raw_qpos(environment: Any) -> np.ndarray | None:
    raw = getattr(getattr(environment, "unwrapped", environment), "env", None)
    try:
        return np.asarray(raw.sim.data.qpos, dtype=float).copy()
    except (AttributeError, TypeError):
        return None


def _direct_write_detected(before_step: np.ndarray | None, after_step: np.ndarray | None, action: np.ndarray) -> bool:
    """Hook for instrumentation; step dynamics are admissible, external writes are not.

    The collector itself only observes state around a public step and never
    obtains a writable MuJoCo state handle.  Future controller instrumentation
    can replace this hook to flag an out-of-band mutation.
    """
    del before_step, after_step, action
    return False


def selected_gripper_pad_center(raw: Any) -> np.ndarray | None:
    """Return the native grasp-pad midpoint, when both Panda pads are present."""
    try:
        positions = [
            np.asarray(raw.sim.data.geom_xpos[index], dtype=float)
            for index in range(int(raw.sim.model.ngeom))
            if is_gripper_pad_geometry_name(str(raw.sim.model.geom_id2name(index) or ""))
        ]
        return np.mean(positions, axis=0) if len(positions) >= 2 else None
    except (AttributeError, TypeError, ValueError):
        return None


def gripper_opening_axis_from_named_pads(named_positions: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Return the horizontal closing axis from official gripper pad groups."""
    positions = {str(name).lower(): np.asarray(value, dtype=float) for name, value in named_positions.items()}
    paired = [value for name, value in positions.items() if "f1_pad_collision" in name or "f2_pad_collision" in name]
    middle = [value for name, value in positions.items() if "finger_middle_pad_collision" in name]
    if paired and middle:
        axis = np.mean(middle, axis=0)[:2] - np.mean(paired, axis=0)[:2]
    elif len(positions) >= 2:
        values = list(positions.values())
        axis = values[1][:2] - values[0][:2]
    else:
        axis = np.array((0.0, 1.0), dtype=float)
    return axis / max(float(np.linalg.norm(axis)), 1e-6)


def gripper_opening_axis_world_from_named_pads(
    named_positions: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Return the full world-space closing axis without discarding wrist roll."""
    positions = {
        str(name).lower(): np.asarray(value, dtype=float)
        for name, value in named_positions.items()
    }
    paired = [
        value for name, value in positions.items()
        if "f1_pad_collision" in name or "f2_pad_collision" in name
    ]
    middle = [
        value for name, value in positions.items()
        if "finger_middle_pad_collision" in name
    ]
    if paired and middle:
        axis = np.mean(middle, axis=0) - np.mean(paired, axis=0)
    elif len(positions) >= 2:
        values = list(positions.values())
        axis = values[1] - values[0]
    else:
        axis = np.array((0.0, 1.0, 0.0), dtype=float)
    return axis / max(float(np.linalg.norm(axis)), 1e-9)


def pick_place_pad_center_target(*, eef: Sequence[float], pad_center: Sequence[float], obj: Sequence[float]) -> np.ndarray:
    """Convert a desired pad midpoint into the wrist target accepted by the controller."""
    return np.asarray(eef, dtype=float) + np.asarray(obj, dtype=float) - np.asarray(pad_center, dtype=float)


def pick_place_pad_alignment_offset(selection_id: str) -> np.ndarray:
    """Compensate the water-bottle pad-center bias measured in the cabinet."""
    if str(selection_id) == "VLA82-017":
        return np.array((0.0, -0.018, 0.0), dtype=float)
    return np.zeros(3, dtype=float)


def pick_place_grasp_target(
    selection_id: str, *, object_center: Sequence[float],
    object_rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return the physical grasp affordance instead of the visual body center."""
    center = np.asarray(object_center, dtype=float)
    rotation = np.asarray(object_rotation, dtype=float).reshape(3, 3)
    if str(selection_id) == "VLA82-018":
        # The boxed drink's visual upper edge sits above its usable parallel-
        # jaw band.  Biasing 25 mm inward centres both pads on the box faces.
        return center - rotation[:, 2] * .025
    if str(selection_id) == "VLA82-040":
        return center + rotation[:, 2] * .055
    if str(selection_id) == "VLA82-046":
        return center.copy()
    if str(selection_id) == "VLA82-048":
        return center.copy()
    if str(selection_id) != "VLA82-015":
        return center.copy()
    return center + rotation[:, 0] * .105


def pick_place_uses_high_grasp_target(selection_id: str) -> bool:
    """Select an upper-body grasp for assets whose body origin is at the base."""
    return str(selection_id) in {"VLA82-004", "VLA82-020"}


def pick_place_high_grasp_target(
    selection_id: str, *, object_center: Sequence[float],
    object_rotation: Sequence[Sequence[float]], object_half_length: float,
) -> np.ndarray:
    """Return a high, physically interior side-grasp point below the spray head."""
    center = np.asarray(object_center, dtype=float)
    if not pick_place_uses_high_grasp_target(selection_id):
        return center.copy()
    axis = np.asarray(object_rotation, dtype=float).reshape(3, 3)[:, 2]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    offset = min(.045, max(0.0, float(object_half_length) - .015))
    return center + axis * offset


def pick_place_uses_pad_center_alignment(selection_id: str) -> bool:
    """Whether a source object needs native-pad rather than wrist alignment."""
    return str(selection_id) in {
        "VLA82-004", "VLA82-013", "VLA82-015", "VLA82-016", "VLA82-017", "VLA82-018", "VLA82-020", "VLA82-021", "VLA82-037", "VLA82-039", "VLA82-046", "VLA82-048", "VLA82-050", "VLA82-051", "VLA82-052", "VLA82-053", "VLA82-054", "VLA82-055", "VLA82-056", "VLA82-057", "VLA82-059",
    }


def pick_place_grasp_strategy(selection_id: str) -> str:
    """Select an explicit physical entry path for collision-sensitive assets."""
    if str(selection_id) == "VLA82-039":
        return "drawer_front_side"
    if str(selection_id) in {"VLA82-052", "VLA82-054", "VLA82-059"}:
        return "horizontal_side"
    return "side" if str(selection_id) == "VLA82-021" else "top"


def pick_place_side_preapproach_tolerance(selection_id: str) -> float:
    """Hand off VLA82-020 after its measured 5.8 cm IK approach deadzone."""
    return .065 if str(selection_id) == "VLA82-020" else .04


def pick_place_retry_grasp_state(selection_id: str) -> str:
    """Keep a failed attempt in its selected physical entry strategy."""
    if pick_place_grasp_strategy(selection_id) == "drawer_front_side":
        return "drawer_side_orient"
    if pick_place_grasp_strategy(selection_id) == "horizontal_side":
        return "side_orient"
    if pick_place_grasp_strategy(selection_id) == "top_extraction":
        return "cabinet_front_insert"
    return "side_preapproach" if pick_place_grasp_strategy(selection_id) == "side" else "approach"


def pick_place_post_cabinet_stage_state(selection_id: str) -> str:
    """Route box extraction through top-grasp yaw alignment at a safe point."""
    return "align_yaw" if pick_place_grasp_strategy(selection_id) == "top_extraction" else "cabinet_front_insert"


def pick_place_post_yaw_alignment_state(selection_id: str) -> str:
    """Enter the selected pad-centered approach after yaw alignment."""
    if pick_place_grasp_strategy(selection_id) == "top_extraction":
        return "cabinet_front_insert"
    return "side_preapproach" if pick_place_grasp_strategy(selection_id) == "side" else "approach"


def pick_place_post_base_reach_state(selection_id: str) -> str:
    """Enter a side pregrasp directly when the source geometry requires it."""
    if str(selection_id) in {"VLA82-053", "VLA82-056"}:
        # The mouse's 70-mm long side leaves almost no clearance inside the
        # Panda gripper.  Align the opening with its 50-mm short side before
        # descending so the inner pads, rather than the finger shells, pinch it.
        return "align_yaw"
    if pick_place_grasp_strategy(selection_id) == "horizontal_side":
        return "side_orient"
    return "side_preapproach" if pick_place_grasp_strategy(selection_id) == "side" else "approach"


def pick_place_post_side_orient_state(selection_id: str) -> str:
    """Clear cabinet doors before bringing a horizontal wrist into reach."""
    if str(selection_id) in {"VLA82-046", "VLA82-048", "VLA82-052", "VLA82-054", "VLA82-059"}:
        return "side_preapproach"
    return "bypass_y_exit" if pick_place_grasp_strategy(selection_id) == "horizontal_side" else "side_preapproach"


def pick_place_post_bypass_state(selection_id: str) -> str:
    """Resume the selected grasp strategy after the three-axis door bypass."""
    return "side_preapproach" if pick_place_grasp_strategy(selection_id) == "horizontal_side" else "approach"


def horizontal_side_grasp_tool_command(
    *, tool_axis_world: Sequence[float], object_center: Sequence[float],
    side_entry: Sequence[float], base_rotation: Sequence[float],
    tolerance: float = np.deg2rad(6.), max_command: float = .55,
) -> tuple[np.ndarray, float]:
    """Rotate the gripper travel axis from top-down to a selected object side."""
    outward = np.asarray(side_entry, dtype=float) - np.asarray(object_center, dtype=float)
    return fixture_handle_tool_alignment_command(
        tool_axis_world=tool_axis_world,
        outward_normal_world=outward,
        base_rotation=base_rotation,
        tolerance=float(tolerance),
        max_command=float(max_command),
    )


def horizontal_side_grasp_roll_command(
    *, opening_axis_world: Sequence[float], target_opening_axis_world: Sequence[float],
    tool_axis_world: Sequence[float], base_rotation: Sequence[float],
    tolerance: float = np.deg2rad(6.), max_command: float = .45,
) -> tuple[np.ndarray, float]:
    """Roll about the horizontal tool axis until opposing pads span the box."""
    tool = np.asarray(tool_axis_world, dtype=float)
    opening = np.asarray(opening_axis_world, dtype=float)
    target = np.asarray(target_opening_axis_world, dtype=float)
    tool /= max(float(np.linalg.norm(tool)), 1e-9)
    opening -= tool * float(np.dot(opening, tool))
    target -= tool * float(np.dot(target, tool))
    opening_norm = float(np.linalg.norm(opening))
    target_norm = float(np.linalg.norm(target))
    if opening_norm <= 1e-9 or target_norm <= 1e-9:
        return np.zeros(3, dtype=float), 0.0
    opening /= opening_norm
    target /= target_norm
    if float(np.dot(opening, target)) < 0.0:
        target = -target
    signed_angle = float(np.arctan2(
        np.dot(tool, np.cross(opening, target)),
        np.clip(np.dot(opening, target), -1.0, 1.0),
    ))
    error = abs(signed_angle)
    if error <= float(tolerance):
        return np.zeros(3, dtype=float), error
    rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3)
    local_tool = rotation.T @ tool
    magnitude = min(error / .5, float(max_command))
    return local_tool * np.sign(signed_angle) * magnitude, error


def horizontal_side_target_opening_axis(
    *, geom_rotation: Sequence[float], geom_size: Sequence[float],
) -> np.ndarray:
    """Return the box's narrow horizontal OBB axis for opposing-pad closure."""
    rotation = np.asarray(geom_rotation, dtype=float).reshape(3, 3)
    size = np.asarray(geom_size, dtype=float)
    axis = np.asarray(rotation[:, int(np.argmin(size[:2]))], dtype=float)
    axis[2] = 0.0
    return axis / max(float(np.linalg.norm(axis)), 1e-9)


def horizontal_side_orientation_transition(
    tool_alignment_error: float, opening_alignment_error: float,
    tolerance: float = np.deg2rad(6.),
) -> str:
    """Gate side entry until both tool direction and pad opening are aligned."""
    if float(tool_alignment_error) > float(tolerance):
        return "side_orient"
    if float(opening_alignment_error) > float(tolerance):
        return "side_axis_align"
    return "side_descend"


def pick_place_side_center_ready(
    selection_id: str, *, distance: float, exact_contact: bool, two_pad_contact: bool,
) -> bool:
    """Decide whether a side-entry grasp can advance to gripper closure."""
    if str(selection_id) in {"VLA82-046", "VLA82-052"}:
        # A flat folder is first touched by the palm if the wrist remains
        # outside its side-grasp pose. Only opposed pads prove that closure
        # will hold rather than shove the folder across the desk.
        return float(distance) <= .025 or bool(two_pad_contact)
    return float(distance) <= .025 or bool(exact_contact)


def drawer_front_orientation_transition(
    tool_alignment_error: float, opening_alignment_error: float,
    tolerance: float = np.deg2rad(6.),
) -> str:
    """Hold the safe orientation pose until both drawer-entry axes align."""
    return (
        "drawer_front_stage"
        if max(float(tool_alignment_error), float(opening_alignment_error)) <= float(tolerance)
        else "drawer_side_orient"
    )


def drawer_front_motion_transition(
    state: str, distance: float, fixture_collision: bool, threshold: float = .04,
) -> str:
    """Advance a drawer waypoint only after observed reach; collision aborts."""
    if bool(fixture_collision):
        return "failed"
    if float(distance) > float(threshold):
        return str(state)
    return "drawer_front_insert" if str(state) == "drawer_front_stage" else "side_center"


def drawer_front_waypoint_step_cap(selection_id: str) -> int:
    """Allow VLA82-037's deep drawer approach to finish its safe descent."""
    return 320 if str(selection_id) == "VLA82-039" else 240 if str(selection_id) == "VLA82-037" else 120


def drawer_over_door_height(
    selection_id: str, *, drawer_high_z: float, nominal_height: float,
) -> float:
    """Keep the long spoon above the open drawer front until insertion."""
    if str(selection_id) in {"VLA82-037", "VLA82-039"}:
        return max(float(nominal_height), float(drawer_high_z) + .08)
    return float(nominal_height)


def pick_place_close_target(
    selection_id: str, *, eef: Sequence[float], tracking_target: Sequence[float],
) -> np.ndarray:
    """Lock a side-entry wrist at first contact while the gripper closes."""
    return np.asarray(
        eef if str(selection_id) in {"VLA82-017", "VLA82-018", "VLA82-021", "VLA82-037", "VLA82-046"} else tracking_target,
        dtype=float,
    )


def pick_place_side_entry_point(
    *, object_center: Sequence[float], eef: Sequence[float], geom_rotation: Sequence[float], geom_size: Sequence[float],
    clearance: float = .05,
) -> np.ndarray:
    """Point outside an OBB along its long axis for collision-free side entry.

    The sign is selected from the live wrist location, so the controller never
    crosses the object merely to reach its chosen entry side.  ``geom_size``
    holds MuJoCo half-extents.
    """
    center = np.asarray(object_center, dtype=float)
    wrist = np.asarray(eef, dtype=float)
    rotation = np.asarray(geom_rotation, dtype=float).reshape(3, 3)
    size = np.asarray(geom_size, dtype=float)
    horizontal = np.asarray((size[0], size[1]), dtype=float)
    # Finger opening is aligned to the short axis. Enter perpendicular to that
    # closing direction so neither near-side finger pushes the free object.
    axis_index = int(np.argmax(horizontal))
    axis = np.asarray(rotation[:, axis_index], dtype=float)
    axis[2] = 0.
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    if float(np.dot(wrist - center, axis)) < 0.:
        axis *= -1.
    return center + axis * (float(size[axis_index]) + float(clearance))


def pick_place_side_entry_reference(
    selection_id: str, *, eef: Sequence[float], base: Sequence[float],
) -> np.ndarray:
    """Choose the physically open side of an edge-supported thin rectangle."""
    if str(selection_id) in {"VLA82-046", "VLA82-048"}:
        return np.asarray(base, dtype=float)
    return np.asarray(eef, dtype=float)


def drawer_front_side_waypoints(
    *, object_center: Sequence[float], robot_base: Sequence[float],
    drawer_low: Sequence[float], drawer_high: Sequence[float],
    slide_axis_world: Sequence[float], geom_rotation: Sequence[Sequence[float]],
    geom_size: Sequence[float], front_clearance: float = .160,
    side_clearance: float = .035,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a front-door stage and long-axis side grasp inside a live drawer."""
    center = np.asarray(object_center, dtype=float)
    base = np.asarray(robot_base, dtype=float)
    low = np.asarray(drawer_low, dtype=float)
    high = np.asarray(drawer_high, dtype=float)
    slide = np.asarray(slide_axis_world, dtype=float)
    rotation = np.asarray(geom_rotation, dtype=float).reshape(3, 3)
    size = np.asarray(geom_size, dtype=float)
    values = (center, base, low, high, slide, rotation, size)
    if any(not np.all(np.isfinite(value)) for value in values):
        raise EnvironmentValidationError("drawer front geometry must be finite")
    if np.any(high <= low) or np.any(size <= 0.0):
        raise EnvironmentValidationError("drawer front geometry has non-positive extents")

    slide_xy = slide[:2].copy()
    slide_norm = float(np.linalg.norm(slide_xy))
    if slide_norm <= 1e-9:
        raise EnvironmentValidationError("drawer slide axis has no horizontal component")
    slide_xy /= slide_norm
    projection = float(np.dot(base[:2] - center[:2], slide_xy))
    if abs(projection) <= 1e-9:
        raise EnvironmentValidationError("robot base does not identify drawer opening side")
    outward = slide_xy * np.sign(projection)

    horizontal_norms = np.linalg.norm(rotation[:2, :], axis=0)
    horizontal_extents = size * horizontal_norms
    long_index = int(np.argmax(horizontal_extents))
    long_axis = rotation[:2, long_index].copy()
    long_norm = float(np.linalg.norm(long_axis))
    if long_norm <= 1e-9:
        raise EnvironmentValidationError("object long axis has no horizontal component")
    long_axis /= long_norm
    reach = float(size[long_index]) + float(side_clearance)
    candidates = tuple(center[:2] + sign * long_axis * reach for sign in (-1.0, 1.0))
    valid = tuple(
        candidate for candidate in candidates
        if np.all(candidate >= low[:2] - 1e-9) and np.all(candidate <= high[:2] + 1e-9)
    )
    if not valid:
        raise EnvironmentValidationError(
            "drawer width cannot contain side grasp waypoint: "
            f"center={center[:2].round(5).tolist()} "
            f"low={low[:2].round(5).tolist()} high={high[:2].round(5).tolist()} "
            f"long_axis={long_axis.round(5).tolist()} reach={reach:.5f} "
            f"candidates={[candidate.round(5).tolist() for candidate in candidates]}"
        )
    side_xy = max(
        valid,
        key=lambda candidate: float(np.min(np.minimum(candidate - low[:2], high[:2] - candidate))),
    )

    object_top = float(center[2] + np.dot(np.abs(rotation[2, :]), size))
    height_low = max(float(low[2] + .035), object_top + .015)
    height_high = float(high[2] - .020)
    if height_low > height_high:
        raise EnvironmentValidationError("drawer height cannot contain front grasp waypoint")
    waypoint_height = float(np.clip(object_top + .035, height_low, height_high))

    inserted = np.array((side_xy[0], side_xy[1], waypoint_height), dtype=float)
    stage = inserted.copy()
    drawer_corners_xy = np.array(
        ((low[0], low[1]), (low[0], high[1]), (high[0], low[1]), (high[0], high[1])),
        dtype=float,
    )
    face_projections = drawer_corners_xy @ slide_xy
    outward_sign = float(np.sign(projection))
    face_projection = (
        float(np.min(face_projections)) if outward_sign < 0.0
        else float(np.max(face_projections))
    )
    desired_stage_projection = face_projection + outward_sign * float(front_clearance)
    stage[:2] += slide_xy * (
        desired_stage_projection - float(np.dot(stage[:2], slide_xy))
    )
    grasp = center.copy()
    return stage, inserted, grasp


def drawer_front_target_tool_axis(
    *, object_center: Sequence[float], inserted_side_point: Sequence[float],
) -> np.ndarray:
    """Aim horizontally from the staged long-axis end toward the object."""
    direction = np.asarray(object_center, dtype=float) - np.asarray(
        inserted_side_point, dtype=float,
    )
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9 or not np.all(np.isfinite(direction)):
        raise EnvironmentValidationError("drawer side tool axis is undefined")
    return direction / norm


def pick_place_close_transition(*, raw_grasp: bool, two_pad_contact: bool, broad_two_finger_contact: bool) -> str:
    """Only native pad contact (or RoboSuite's native grasp) may secure an object.

    ``broad_two_finger_contact`` is intentionally an input for auditability: a
    pair of finger-shell contacts is useful diagnostics but is not a grasp.
    """
    del broad_two_finger_contact
    return "secure" if bool(raw_grasp) or bool(two_pad_contact) else "close"


def pick_place_close_transition_for_selection(
    selection_id: str, *, raw_grasp: bool, two_pad_contact: bool,
    broad_two_finger_contact: bool,
) -> str:
    """Apply the stronger opposing-pad gate to calibrated box grasps."""
    if pick_place_grasp_strategy(selection_id) in {"horizontal_side", "top_extraction", "drawer_front_side"}:
        return "secure" if bool(raw_grasp) and bool(two_pad_contact) else "close"
    return pick_place_close_transition(
        raw_grasp=raw_grasp,
        two_pad_contact=two_pad_contact,
        broad_two_finger_contact=broad_two_finger_contact,
    )


def validate_knob_gate(
    expected_joint_id: str,
    observed_joint_id: str | None,
    initial_joint_value: float,
    final_joint_value: float,
    contacts: Sequence[Mapping[str, Any]],
    *,
    joint_range: Sequence[float] = (0.0, 0.0),
    collateral_initial_position: Sequence[float] = (),
    collateral_final_position: Sequence[float] = (),
    collateral_contact_free: bool = True,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not expected_joint_id or observed_joint_id != expected_joint_id:
        errors.append("knob_exact_joint_missing")
    if not any(
        item.get("joint_id") == expected_joint_id and bool(item.get("gripper_contact")) and bool(item.get("after_step"))
        for item in contacts
    ):
        errors.append("knob_contact_missing")
    if abs(float(final_joint_value) - float(initial_joint_value)) < knob_required_turn_delta(joint_range):
        errors.append("knob_joint_delta_not_visibly_open")
    if not knob_collateral_is_stable(
        initial_position=collateral_initial_position,
        final_position=collateral_final_position,
    ):
        errors.append("knob_collateral_cookware_displaced")
    if not bool(collateral_contact_free):
        errors.append("knob_collateral_cookware_contact")
    return tuple(errors)


def _knob_contacts(history: Sequence[Any], expected_joint: str) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for snapshot in history:
        for evidence in getattr(snapshot, "contact_evidence", ()):
            result.append({
                "joint_id": getattr(evidence, "joint_id", None),
                "gripper_contact": "gripper" in str(getattr(evidence, "actor", "")).lower() or "gripper" in str(getattr(evidence, "counterpart", "")).lower(),
                "after_step": int(getattr(snapshot, "step", 0)) > 0,
            })
    return tuple(result)


def _predicate_dict(predicate: Any) -> dict[str, Any]:
    if hasattr(predicate, "to_dict"):
        return dict(predicate.to_dict())
    return {"success": bool(getattr(predicate, "success", False))}


def _write_npz(path: Path, *, primary: Sequence[np.ndarray], wrist: Sequence[np.ndarray], proprio: Sequence[np.ndarray], actions: Sequence[np.ndarray], phases: Sequence[int], snapshots: Sequence[Any], predicate: Any, seed: int, spec_sha: str, scene_sha: str, asset_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(EXPERT_SCHEMA_VERSION, dtype=np.int64),
            primary=np.stack(primary), wrist=np.stack(wrist), proprio=np.stack(proprio),
            actions=np.stack(actions), phases=np.asarray(phases, dtype=np.int16),
            contacts_json=np.asarray(json.dumps([_snapshot_json(item).get("contacts", []) for item in snapshots], ensure_ascii=False)),
            physics_snapshots_json=np.asarray(json.dumps([_snapshot_json(item) for item in snapshots], ensure_ascii=False)),
            predicate_json=np.asarray(json.dumps(_predicate_dict(predicate), ensure_ascii=False)),
            training_expert=np.asarray(True), seed=np.asarray(seed, dtype=np.int64),
            spec_sha256=np.asarray(spec_sha), scene_sha256=np.asarray(scene_sha), asset_sha256=np.asarray(asset_sha),
        )
    temporary.replace(path)


def validated_resume_episode(path: Path, spec: OperationSpec, scene: Any, seed: int, *, scene_sha256: str | None = None) -> bool:
    """Only a complete, source-matched expert PASS is reusable."""
    required = {"schema_version", "primary", "wrist", "proprio", "actions", "phases", "contacts_json", "physics_snapshots_json", "predicate_json", "training_expert", "seed", "spec_sha256", "scene_sha256", "asset_sha256"}
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files) or int(data["schema_version"].item()) != EXPERT_SCHEMA_VERSION:
                return False
            if not bool(data["training_expert"].item()) or int(data["seed"].item()) != int(seed):
                return False
            expected_scene = scene_sha256 or scene_fingerprint(scene)
            if str(data["spec_sha256"].item()) != spec_fingerprint(spec) or str(data["scene_sha256"].item()) != expected_scene or str(data["asset_sha256"].item()) != asset_fingerprint(scene):
                return False
            if data["primary"].shape[0] == 0 or data["primary"].shape[0] != data["actions"].shape[0] or data["proprio"].shape != (data["actions"].shape[0], 8):
                return False
            return bool(json.loads(str(data["predicate_json"].item())).get("success", False))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _manifest_path(output_dir: Path, selection_id: str) -> Path:
    return output_dir / selection_id / "manifest.json"


def _record_manifest(output_dir: Path, report: EpisodeReport) -> None:
    path = _manifest_path(output_dir, report.selection_id)
    existing: dict[str, Any] = {"selection_id": report.selection_id, "episodes": {}}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    existing.setdefault("episodes", {})[str(report.seed)] = report.to_dict()
    existing["successful_episodes"] = sum(1 for row in existing["episodes"].values() if row.get("status") == "PASS")
    _atomic_json(path, existing)


def collect_expert_episode(
    spec: OperationSpec,
    scene: SceneRequest | Any,
    seed: int,
    output_dir: Path,
    *,
    camera_width: int = 256,
    camera_height: int = 256,
) -> EpisodeReport:
    """Run one source-phase expert through public steps and capture raw proof."""
    output_dir = Path(output_dir)
    episode_dir = output_dir / spec.selection_id
    npz_path = episode_dir / f"episode-{seed}.npz"
    diagnostic_path = episode_dir / f"episode-{seed}.json"
    spec_sha, scene_sha, asset_sha = spec_fingerprint(spec), scene_fingerprint(scene), asset_fingerprint(scene)
    environment = None
    errors: list[str] = []
    primary: list[np.ndarray] = []
    wrist: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phase_values: list[int] = []
    snapshots: list[Any] = []
    predicate: Any = None
    runtime_scene: dict[str, Any] = {}
    friction_evidence: dict[str, Any] = {}
    coverage_tracker: CleaningCoverageTracker | None = None
    try:
        environment = make_environment(
            scene,
            seed=seed,
            camera_width=int(camera_width),
            camera_height=int(camera_height),
        )
        friction_evidence = dict(getattr(environment, "friction_evidence", {}))
        observation, _ = environment.reset(seed=seed)
        runtime_scene = scene_runtime_fingerprint(environment, scene)
        scene_sha = str(runtime_scene["sha256"])
        if validated_resume_episode(npz_path, spec, scene, seed, scene_sha256=scene_sha):
            report = EpisodeReport(spec.selection_id, seed, "PASS", True, str(npz_path.resolve()), str(diagnostic_path.resolve()), True, (), spec_sha, scene_sha, asset_sha, 0)
            _record_manifest(output_dir, report)
            return report
        contract = build_physics_capture_contract(environment, scene)
        if set(spec.phases) & {"wipe", "scrub"}:
            # The tracker is a capture-side derivation from immutable raw
            # contact snapshots. It never writes a simulator or task metric.
            coverage_tracker = CleaningCoverageTracker(next(iter(contract.target_geometries)))
        initial = snapshot_from_environment(environment, 0, contract=contract)
        snapshots.append(initial)
        cameras = tuple(getattr(scene, "camera_names", ("agentview", "robot0_robotview")))
        primary_camera = cameras[0] if cameras else "agentview"
        wrist_camera = cameras[1] if len(cameras) > 1 else primary_camera
        expert_controller = CompositeExpert(spec.phases, contract)
        for step, (phase, action) in enumerate(expert_controller.run(environment), start=1):
            before = _raw_qpos(environment)
            observation, _, terminated, truncated, _ = environment.step(action)
            after = _raw_qpos(environment)
            if _direct_write_detected(before, after, action):
                errors.append("direct_state_write_detected")
                break
            snap = snapshot_from_environment(environment, step, contract=contract)
            raw = _raw_environment(environment)
            held_tool = False
            try:
                raw_grasp = bool(raw._check_grasp(raw.robots[0].gripper["right"], raw.objects["obj"]))
                held_tool = held_for_cleaning_credit(
                    raw_grasp=raw_grasp,
                    two_pad_contact=_selected_geom_has_two_finger_contacts(raw, tuple(getattr(contract, "object_geom_names", {}).get("obj", ()))),
                )
            except (AttributeError, KeyError, TypeError):
                held_tool = False
            if coverage_tracker is not None and phase in {"wipe", "scrub"}:
                # Preserve the immutable derived dirt baseline after a lost
                # grasp, while granting new cells only through the dual gate.
                snap = coverage_tracker.derive(snap, credit=held_tool)
            snapshots.append(snap)
            primary.append(_frame(observation, primary_camera))
            wrist.append(_frame(observation, wrist_camera, primary[-1]))
            proprio.append(_proprio(snap))
            actions.append(np.asarray(action, dtype=np.float32))
            phase_values.append(PHASE_INDEX.get(phase, -1))
            if terminated or truncated:
                errors.append("environment_terminated_before_source_phases_complete")
                break
        if not actions:
            errors.append("no_env_step_actions")
        final = snapshots[-1] if snapshots else initial
        evaluation_spec = pick_place_evaluation_spec_for_scene(spec, scene)
        predicate = evaluate_operation(evaluation_spec, initial, snapshots[1:], final)
        if not bool(getattr(predicate, "success", False)):
            errors.extend(f"predicate:{item}" for item in getattr(predicate, "errors", ("failed",)))
        if not fixture_close_handle_grasp_observed(
            spec.selection_id, expert_controller.controller_trace,
        ):
            errors.append("fixture_close:two_pad_handle_grasp_not_observed")
        errors.extend(pick_place_strict_trace_errors(
            spec.phases, expert_controller.controller_trace,
        ))
        if spec.selection_id == "VLA82-007":
            expected_joint = getattr(contract, "fixture_joint_ids", {}).get("turn_knob", "")
            initial_joint = float(getattr(initial, "joint_positions", {}).get(expected_joint, 0.0))
            final_joint = float(getattr(final, "joint_positions", {}).get(expected_joint, initial_joint))
            try:
                raw = _raw_environment(environment)
                model_joint_id = int(raw.sim.model.joint_name2id(expected_joint))
                joint_range = np.asarray(raw.sim.model.jnt_range[model_joint_id], dtype=float)
            except (AttributeError, KeyError, TypeError, ValueError):
                joint_range = np.asarray((0.0, 0.0), dtype=float)
            errors.extend(validate_knob_gate(
                expected_joint, expected_joint if expected_joint else None,
                initial_joint, final_joint, _knob_contacts(snapshots, expected_joint),
                joint_range=joint_range,
                collateral_initial_position=getattr(initial, "body_poses", {}).get("cookware", ()),
                collateral_final_position=getattr(final, "body_poses", {}).get("cookware", ()),
                collateral_contact_free=knob_collateral_contact_free(snapshots),
            ))
    except Exception as error:  # Diagnostics are retained; failure never counts as a demo.
        errors.append(f"collector_exception:{type(error).__name__}:{error}")
        errors.append(f"collector_traceback:{traceback.format_exc()}")
    finally:
        if environment is not None:
            environment.close()
    success = not errors and bool(getattr(predicate, "success", False))
    if success:
        _write_npz(npz_path, primary=primary, wrist=wrist, proprio=proprio, actions=actions, phases=phase_values, snapshots=snapshots, predicate=predicate, seed=seed, spec_sha=spec_sha, scene_sha=scene_sha, asset_sha=asset_sha)
    report = EpisodeReport(spec.selection_id, seed, "PASS" if success else "FAIL", True, str(npz_path.resolve()), str(diagnostic_path.resolve()), bool(getattr(predicate, "success", False)), tuple(dict.fromkeys(errors)), spec_sha, scene_sha, asset_sha, len(actions))
    _atomic_json(diagnostic_path, {**report.to_dict(), "runtime_scene": runtime_scene, "friction_evidence": friction_evidence, "controller_trace": expert_controller.controller_trace if 'expert_controller' in locals() else [], "derived_cleaning_coverage": coverage_tracker.evidence if coverage_tracker is not None else [], "predicate": _predicate_dict(predicate) if predicate is not None else {"success": False}, "snapshots": [_snapshot_json(item) for item in snapshots]})
    _record_manifest(output_dir, report)
    return report
