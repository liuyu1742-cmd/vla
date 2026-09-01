"""Source-phase-specific physical success predicates for VLA82 rollouts.

The predicates in this module deliberately accept only read-only physics
snapshots.  They require an interaction history; a terminal pose, a task label,
or a controller completion flag is never a success signal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .annotations import OperationSpec
from .environment import ContactEvidence, PhysicsSnapshot, TargetGeometry


def surface_coverage_cell(point: Sequence[float], *, cell_size: float = 0.02) -> tuple[int, int]:
    """Map a world point to the same fixed 2-cm grid used by live capture."""
    return tuple(np.floor(np.asarray(point, dtype=float)[:2] / float(cell_size)).astype(int))


@dataclass(frozen=True)
class PredicateResult:
    """Immutable, auditable outcome for one physical phase or whole operation."""

    success: bool
    phase_results: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, float | bool | str]
    contact_verified: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        def freeze(value: Any) -> Any:
            if isinstance(value, Mapping):
                return MappingProxyType({key: freeze(item) for key, item in value.items()})
            if isinstance(value, list):
                return tuple(freeze(item) for item in value)
            if isinstance(value, tuple):
                return tuple(freeze(item) for item in value)
            return value

        object.__setattr__(self, "phase_results", tuple(freeze(item) for item in self.phase_results))
        object.__setattr__(self, "metrics", freeze(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        """Return a stable ordinary-dict payload suitable for JSON evidence."""
        def thaw(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): thaw(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return value

        return {
            "success": self.success,
            "phase_results": thaw(self.phase_results),
            "metrics": thaw(self.metrics),
            "contact_verified": self.contact_verified,
            "errors": list(self.errors),
        }


# These contracts are intentionally explicit.  Reporting can expose them along
# with the raw metric values without inventing a generic task-complete score.
PHASE_CONTRACTS: dict[str, dict[str, tuple[str, ...] | dict[str, float]]] = {
    "grasp": {"metrics": ("gripper_closure", "gripper_object_contact", "lift_distance"), "thresholds": {"closure": 0.010, "lift": 0.025}},
    "place": {"metrics": ("stable_release", "target_support_contact", "transit_distance"), "thresholds": {"stable_steps": 2.0, "transit": 0.025}},
    "return": {"metrics": ("stable_release", "target_support_contact", "transit_distance"), "thresholds": {"stable_steps": 2.0, "transit": 0.025}},
    "deposit": {"metrics": ("inside_target_volume", "stable_release", "transit_distance"), "thresholds": {"stable_steps": 2.0, "transit": 0.025}},
    "wipe": {"metrics": ("dirt_reduction", "tool_surface_contact_coverage", "spatial_coverage_cells"), "thresholds": {"dirt_reduction": 0.30, "contact_steps": 3.0, "spatial_cells": 3.0}},
    "scrub": {"metrics": ("dirt_reduction", "tool_surface_contact_coverage", "spatial_coverage_cells"), "thresholds": {"dirt_reduction": 0.30, "contact_steps": 3.0, "spatial_cells": 3.0}},
    "spray": {"metrics": ("spray_coverage", "trigger_contact"), "thresholds": {"coverage": 0.20}},
    "insert": {"metrics": ("containment", "stable_release", "transit_distance"), "thresholds": {"stable_steps": 2.0, "transit": 0.025}},
    "stack": {"metrics": ("stable_support", "stable_release", "transit_distance"), "thresholds": {"stable_steps": 2.0, "transit": 0.025}},
    "arrange": {"metrics": ("spatial_relation", "stable_release", "transit_distance"), "thresholds": {"stable_steps": 2.0, "transit": 0.025}},
    "open": {"metrics": ("joint_target", "joint_motion", "fixture_contact"), "thresholds": {"joint_motion": 0.20, "open_target": 0.35}},
    "close": {"metrics": ("joint_target", "joint_motion", "fixture_contact"), "thresholds": {"joint_motion": 0.20, "closed_target": 0.15}},
    "close_lid": {"metrics": ("joint_target", "joint_motion", "fixture_contact"), "thresholds": {"joint_motion": 0.20, "closed_target": 0.15}},
    "pull": {"metrics": ("joint_target", "joint_motion", "fixture_contact"), "thresholds": {"joint_motion": 0.20}},
    "push": {"metrics": ("joint_target", "joint_motion", "fixture_contact"), "thresholds": {"joint_motion": 0.20}},
    "turn_knob": {"metrics": ("joint_target", "joint_motion", "fixture_contact"), "thresholds": {"joint_motion": 0.20}},
    "press": {"metrics": ("dispensed_amount", "trigger_contact"), "thresholds": {"dispensed_amount": 0.01}},
    # Source annotations with no parseable action phrase remain a strict
    # pick-transit-release contract, rather than a metadata-only pass.
    "composite": {"metrics": ("physical_transition_count", "target_support_contact", "transit_distance"), "thresholds": {"transitions": 2.0, "transit": 0.025}},
}


def fixture_motion_threshold(phase: str, initial_joint_position: float) -> float:
    """Return a strict but physically attainable joint-motion requirement."""
    generic = float(PHASE_CONTRACTS[str(phase)]["thresholds"]["joint_motion"])  # type: ignore[index,union-attr]
    initial = abs(float(initial_joint_position))
    if str(phase) == "push" and initial >= .02:
        return min(generic, .75 * initial)
    return generic


def fixture_target_reached(phase: str, joint_positions: Sequence[float]) -> bool:
    """Check a fixture's terminal target without assuming all slides are long."""
    values = tuple(float(value) for value in joint_positions)
    if not values:
        return False
    if str(phase) == "open":
        return values[-1] >= float(PHASE_CONTRACTS["open"]["thresholds"]["open_target"])  # type: ignore[index,union-attr]
    if str(phase) in {"close", "close_lid"}:
        return abs(values[-1]) <= float(PHASE_CONTRACTS[str(phase)]["thresholds"]["closed_target"])  # type: ignore[index,union-attr]
    if str(phase) == "push":
        initial = abs(values[0])
        return initial >= .02 and abs(values[-1]) <= max(.004, .10 * initial)
    return True


_FIXTURE_TOKENS = {
    "VLA82-007": "knob", "VLA82-008": "door", "VLA82-009": "rack", "VLA82-010": "rack",
    "VLA82-026": "shelf", "VLA82-027": "drawer", "VLA82-028": "door", "VLA82-029": "microwave",
    "VLA82-030": "slidejoint", "VLA82-031": "door", "VLA82-032": "rack", "VLA82-033": "door",
    "VLA82-034": "door", "VLA82-035": "door", "VLA82-049": "lid",
}


def _timeline(initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> tuple[PhysicsSnapshot, ...]:
    """Sort and de-duplicate snapshots by physical step, retaining final state."""
    items = {snapshot.step: snapshot for snapshot in (initial, *history, final)}
    return tuple(items[step] for step in sorted(items))


def _names(contact: tuple[str, str, float]) -> tuple[str, str]:
    return str(contact[0]).lower(), str(contact[1]).lower()


def _is_gripper(name: str) -> bool:
    return any(token in name for token in ("gripper", "finger", "robot0"))


def _contacts_between(snapshot: PhysicsSnapshot, left: Callable[[str], bool], right: Callable[[str], bool]) -> bool:
    for contact in snapshot.contacts:
        first, second = _names(contact)
        if (left(first) and right(second)) or (left(second) and right(first)):
            return True
    return False


def _has_any_gripper_contact(snapshot: PhysicsSnapshot) -> bool:
    return _contacts_between(snapshot, _is_gripper, lambda name: not _is_gripper(name))


def _object_key(snapshot: PhysicsSnapshot, index: int = 0) -> str | None:
    preferred = "obj" if index == 0 else f"annotated_{index:02d}"
    if preferred in snapshot.body_poses:
        return preferred
    return next(iter(snapshot.body_poses), None)


def _z(snapshot: PhysicsSnapshot, key: str | None) -> float:
    if key is None or key not in snapshot.body_poses or len(snapshot.body_poses[key]) < 3:
        return 0.0
    return float(np.asarray(snapshot.body_poses[key], dtype=float)[2])


def _object_vertical_cosine(snapshot: PhysicsSnapshot, key: str = "obj") -> float:
    """Return the world-Z component of an object's local upright axis."""
    pose = np.asarray(snapshot.body_poses.get(key, ()), dtype=float)
    if pose.size < 7:
        return -1.0
    quaternion = pose[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        return -1.0
    _, x, y, _ = quaternion / norm
    return float(1.0 - 2.0 * (x * x + y * y))


def requires_upright_final_pose(selection_id: str) -> bool:
    """Return whether a tall bottle must finish standing on its support."""
    return str(selection_id) in {"VLA82-004", "VLA82-017"}


def _gripper_width(snapshot: PhysicsSnapshot) -> float:
    values = np.asarray(snapshot.gripper_qpos, dtype=float)
    # PandaOmron exposes one signed joint coordinate for each opposing finger.
    # The physical jaw aperture is their total separation, not their mean.
    return float(np.sum(np.abs(values))) if values.size else 0.0


def _result(success: bool, metrics: dict[str, float | bool | str], *, contact: bool, errors: Iterable[str] = (), phase: str = "", step: int | None = None) -> PredicateResult:
    phase_row: dict[str, Any] = {"phase": phase, "success": success, "transition_step": step}
    return PredicateResult(success, (phase_row,), metrics, contact, tuple(dict.fromkeys(errors)))


def _grasp_event(timeline: Sequence[PhysicsSnapshot]) -> tuple[int | None, float, float]:
    if not timeline:
        return None, 0.0, 0.0
    initial_width = _gripper_width(timeline[0])
    initial_z = _z(timeline[0], _object_key(timeline[0]))
    for snapshot in timeline[1:]:
        if _has_any_gripper_contact(snapshot) and initial_width - _gripper_width(snapshot) >= 0.015:
            return snapshot.step, initial_width - _gripper_width(snapshot), max(0.0, _z(snapshot, _object_key(snapshot)) - initial_z)
    return None, 0.0, 0.0


def _transit_after_grasp(timeline: Sequence[PhysicsSnapshot]) -> tuple[int | None, float]:
    grasp_step, _, _ = _grasp_event(timeline)
    if grasp_step is None:
        return None, 0.0
    initial_z = _z(timeline[0], _object_key(timeline[0]))
    for snapshot in timeline:
        if snapshot.step > grasp_step:
            distance = max(0.0, _z(snapshot, _object_key(snapshot)) - initial_z)
            if distance >= 0.025:
                return snapshot.step, distance
    return None, 0.0


def _is_target_name(name: str) -> bool:
    return any(token in name for token in ("target", "surface", "bin", "receptacle", "support", "reference", "container", "shelf", "cabinet", "drawer", "cup"))


def _is_manipulated_geom(name: str) -> bool:
    """Recognize both named scene objects and RoboCasa/ObjectPlay geom names."""
    return name == "obj" or name.startswith(("obj_", "annotated_", "mcjfobj_"))


def _target_contacts(snapshot: PhysicsSnapshot) -> bool:
    return _contacts_between(snapshot, _is_manipulated_geom, _is_target_name)


def _release_event(timeline: Sequence[PhysicsSnapshot]) -> tuple[int | None, int]:
    transit_step, _ = _transit_after_grasp(timeline)
    if transit_step is None:
        return None, 0
    min_width = min(_gripper_width(snapshot) for snapshot in timeline)
    sustained = 0
    start: int | None = None
    for snapshot in timeline:
        if snapshot.step <= transit_step:
            continue
        released = _gripper_width(snapshot) - min_width >= 0.015
        if released and _target_contacts(snapshot):
            start = snapshot.step if start is None else start
            sustained += 1
        elif start is not None:
            break
    return start, sustained


def _generic_release(phase: str, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    transit_step, transit = _transit_after_grasp(timeline)
    release_step, sustained = _release_event(timeline)
    contact = any(_target_contacts(snapshot) for snapshot in timeline) and any(_has_any_gripper_contact(snapshot) for snapshot in timeline)
    metrics: dict[str, float | bool | str] = {
        "transit_distance": transit,
        "target_support_contact": any(_target_contacts(snapshot) for snapshot in timeline),
        "stable_release": sustained >= 2,
        "transition_step": float(release_step or 0),
    }
    if phase == "deposit":
        metrics["inside_target_volume"] = bool(metrics["target_support_contact"] and sustained >= 2)
    elif phase == "insert":
        metrics["containment"] = bool(metrics["target_support_contact"] and sustained >= 2)
    elif phase == "stack":
        metrics["stable_support"] = bool(metrics["target_support_contact"] and sustained >= 2)
    elif phase == "arrange":
        metrics["spatial_relation"] = bool(metrics["target_support_contact"] and sustained >= 2)
    errors: list[str] = []
    if not contact:
        errors.append("contact_missing")
    if transit_step is None:
        errors.append("transit_or_grasp_evidence_missing")
    if release_step is None:
        errors.append("release_evidence_missing")
    if sustained < 2:
        errors.append("release_not_stable")
    return _result(not errors, metrics, contact=contact, errors=errors, phase=phase, step=release_step)


def evaluate_grasp(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    step, closure, contact_lift = _grasp_event(timeline)
    final_lift = max(0.0, _z(final, _object_key(final)) - _z(initial, _object_key(initial)))
    contact = step is not None
    errors = []
    if not contact:
        errors.append("contact_missing")
    if closure < 0.015:
        errors.append("gripper_closure_insufficient")
    if max(contact_lift, final_lift) < 0.025:
        errors.append("lift_insufficient")
    return _result(not errors, {"gripper_closure": closure, "gripper_object_contact": contact, "lift_distance": max(contact_lift, final_lift), "transition_step": float(step or 0)}, contact=contact, errors=errors, phase="grasp", step=step)


def evaluate_place(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _generic_release("place", initial, history, final)


def evaluate_deposit(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _generic_release("deposit", initial, history, final)


def evaluate_wipe(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    contact_steps = [snapshot.step for snapshot in timeline if _contacts_between(snapshot, lambda name: name == "obj", lambda name: "surface" in name)]
    reduction = max(0.0, float(initial.dirt_fraction) - float(final.dirt_fraction))
    contact = len(contact_steps) >= 3
    errors = []
    if not contact:
        errors.append("contact_missing")
    if reduction < 0.30:
        errors.append("dirt_reduction_insufficient")
    return _result(not errors, {"dirt_reduction": reduction, "tool_surface_contact_coverage": float(len(contact_steps)), "transition_step": float(contact_steps[0] if contact_steps else 0)}, contact=contact, errors=errors, phase="wipe", step=contact_steps[0] if contact_steps else None)


def evaluate_spray(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    trigger_steps = [snapshot.step for snapshot in timeline if _contacts_between(snapshot, _is_gripper, lambda name: "trigger" in name or "button" in name)]
    coverage = max(0.0, float(final.spray_coverage) - float(initial.spray_coverage))
    contact = bool(trigger_steps)
    errors = []
    if not contact:
        errors.append("contact_missing")
    if coverage < 0.20:
        errors.append("spray_coverage_insufficient")
    return _result(not errors, {"spray_coverage": coverage, "trigger_contact": contact, "transition_step": float(trigger_steps[0] if trigger_steps else 0)}, contact=contact, errors=errors, phase="spray", step=trigger_steps[0] if trigger_steps else None)


def _expected_fixture_token(spec: OperationSpec | None, phase: str) -> str:
    if spec and spec.selection_id in _FIXTURE_TOKENS:
        return _FIXTURE_TOKENS[spec.selection_id]
    return {"turn_knob": "knob", "pull": "slider", "push": "slider", "close_lid": "lid"}.get(phase, "door")


def _joint_predicate(phase: str, spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    token = _expected_fixture_token(spec, phase)
    candidates = [name for name in set(initial.joint_positions) | set(final.joint_positions) if token in name.lower()]
    if not candidates:
        return _result(False, {"joint_target": False, "joint_motion": 0.0, "fixture_contact": False, "transition_step": 0.0}, contact=False, errors=("fixture_joint_mismatch",), phase=phase)
    name = sorted(candidates)[0]
    start = float(initial.joint_positions.get(name, 0.0))
    end = float(final.joint_positions.get(name, start))
    motion = abs(end - start)
    contact_steps = [snapshot.step for snapshot in timeline if _contacts_between(snapshot, _is_gripper, lambda value: token in value)]
    contact = bool(contact_steps)
    target = motion >= 0.20
    if phase == "open":
        target = target and end >= 0.35
    elif phase in {"close", "close_lid"}:
        target = target and abs(end) <= 0.15
    errors = []
    if not contact:
        errors.append("contact_missing")
    if not target:
        errors.append("joint_target_not_reached")
    return _result(not errors, {"joint_target": target, "joint_motion": motion, "fixture_contact": contact, "transition_step": float(contact_steps[0] if contact_steps else 0)}, contact=contact, errors=errors, phase=phase, step=contact_steps[0] if contact_steps else None)


def evaluate_hinge(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot, *, phase: str = "open") -> PredicateResult:
    return _joint_predicate(phase, spec, initial, history, final)


def evaluate_slider(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot, *, phase: str = "pull") -> PredicateResult:
    return _joint_predicate(phase, spec, initial, history, final)


def evaluate_knob(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _joint_predicate("turn_knob", spec, initial, history, final)


def evaluate_press(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    trigger_steps = [snapshot.step for snapshot in timeline if _contacts_between(snapshot, _is_gripper, lambda name: "button" in name or "trigger" in name or "dispenser" in name)]
    amount = max(0.0, float(final.dispensed_amount) - float(initial.dispensed_amount))
    contact = bool(trigger_steps)
    errors = []
    if not contact:
        errors.append("contact_missing")
    if amount < 0.01:
        errors.append("dispensed_amount_insufficient")
    return _result(not errors, {"dispensed_amount": amount, "trigger_contact": contact, "transition_step": float(trigger_steps[0] if trigger_steps else 0)}, contact=contact, errors=errors, phase="press", step=trigger_steps[0] if trigger_steps else None)


def evaluate_composite(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    grasp_step, _, _ = _grasp_event(timeline)
    transit_step, transit = _transit_after_grasp(timeline)
    release_step, stable = _release_event(timeline)
    transitions = sum(step is not None for step in (grasp_step, transit_step, release_step))
    contact = any(_has_any_gripper_contact(item) for item in timeline) and any(_target_contacts(item) for item in timeline)
    errors = []
    if not contact:
        errors.append("contact_missing")
    if transitions < 3 or stable < 2:
        errors.append("composite_physical_transition_missing")
    return _result(not errors, {"physical_transition_count": float(transitions), "target_support_contact": any(_target_contacts(item) for item in timeline), "transit_distance": transit, "transition_step": float(release_step or 0)}, contact=contact, errors=errors, phase="composite", step=release_step)


PREDICATES: dict[str, Callable[[OperationSpec | None, PhysicsSnapshot, Sequence[PhysicsSnapshot], PhysicsSnapshot], PredicateResult]] = {
    "grasp": evaluate_grasp,
    "place": evaluate_place,
    "return": evaluate_place,
    "deposit": evaluate_deposit,
    "wipe": evaluate_wipe,
    "scrub": evaluate_wipe,
    "spray": evaluate_spray,
    "insert": lambda spec, initial, history, final: _generic_release("insert", initial, history, final),
    "stack": lambda spec, initial, history, final: _generic_release("stack", initial, history, final),
    "arrange": lambda spec, initial, history, final: _generic_release("arrange", initial, history, final),
    "open": lambda spec, initial, history, final: evaluate_hinge(spec, initial, history, final, phase="open"),
    "close": lambda spec, initial, history, final: evaluate_hinge(spec, initial, history, final, phase="close"),
    "close_lid": lambda spec, initial, history, final: evaluate_hinge(spec, initial, history, final, phase="close_lid"),
    "pull": lambda spec, initial, history, final: evaluate_slider(spec, initial, history, final, phase="pull"),
    "push": lambda spec, initial, history, final: evaluate_slider(spec, initial, history, final, phase="push"),
    "turn_knob": evaluate_knob,
    "press": evaluate_press,
    "composite": evaluate_composite,
}


@dataclass(frozen=True)
class PhaseWindow:
    """One uniquely assigned, strictly ordered physics-event window."""

    phase: str
    start_step: int
    end_step: int
    snapshots: tuple[PhysicsSnapshot, ...]


def _entity_for(index: int) -> str:
    return "obj" if index == 0 else f"annotated_{index:02d}"


def _named_contact(snapshot: PhysicsSnapshot, *, object_id: str, target_id: str | None = None, joint_id: str | None = None, gripper: bool = False) -> bool:
    """Match only exact scene identities supplied by ``ContactEvidence``."""
    for contact in snapshot.contact_evidence:
        if contact.object_id != object_id:
            continue
        if target_id is not None and contact.target_id != target_id:
            continue
        if joint_id is not None and contact.joint_id != joint_id:
            continue
        if gripper and "gripper" not in contact.actor.lower() and "gripper" not in contact.counterpart.lower():
            continue
        return True
    return False


def opposing_finger_grasp_sufficient(
    snapshot: PhysicsSnapshot | Any, *, object_id: str, closure: float,
    nominal_closure: float = .010, constrained_closure: float = .0005,
) -> bool:
    """Accept a wide-object pinch only with exact contacts on both fingers."""
    if float(closure) >= float(nominal_closure):
        return True
    if float(closure) < float(constrained_closure):
        return False
    fingers: set[str] = set()
    for contact in getattr(snapshot, "contact_evidence", ()):
        if contact.object_id != object_id:
            continue
        names = f"{contact.actor} {contact.counterpart}".lower()
        if "finger1" in names:
            fingers.add("finger1")
        if "finger2" in names:
            fingers.add("finger2")
    return fingers == {"finger1", "finger2"}


def _target_for(snapshot: PhysicsSnapshot) -> TargetGeometry | None:
    return next(iter(snapshot.target_geometries.values()), None)


def _target_for_object_contact(snapshot: PhysicsSnapshot, object_id: str) -> TargetGeometry | None:
    for contact in snapshot.contact_evidence:
        if contact.object_id == object_id and contact.target_id in snapshot.target_geometries:
            return snapshot.target_geometries[contact.target_id]
    return None


def _inside_target(snapshot: PhysicsSnapshot, object_id: str, target: TargetGeometry) -> bool:
    pose = snapshot.body_poses.get(object_id)
    if pose is None:
        return False
    position = np.asarray(pose, dtype=float)[:3]
    return bool(np.all(position >= np.asarray(target.min_corner, dtype=float)) and np.all(position <= np.asarray(target.max_corner, dtype=float)))


def _release_relation(snapshot: PhysicsSnapshot, object_id: str, target: TargetGeometry, phase: str) -> bool:
    """Check the source-declared target relation; placement is not always containment."""
    pose = snapshot.body_poses.get(object_id)
    if pose is None:
        return False
    point = np.asarray(pose, dtype=float)[:3]
    low = np.asarray(target.min_corner, dtype=float)
    high = np.asarray(target.max_corner, dtype=float)
    relation = target.spatial_relation
    if phase in {"deposit", "insert"} or relation == "inside":
        return _inside_target(snapshot, object_id, target)
    if relation in {"on", "above"}:
        lateral = bool(np.all(point[:2] >= low[:2] - 0.03) and np.all(point[:2] <= high[:2] + 0.03))
        return lateral and bool(point[2] >= high[2] - 0.03)
    if relation in {"near", "beside"}:
        return float(np.linalg.norm(point[:2] - ((low[:2] + high[:2]) / 2.0))) <= 0.15
    # A task fixture can declare a custom support relation. Contact is still
    # required by the caller; no containment is invented for such a surface.
    return True


def stable_release_end_index(
    timeline: Sequence[PhysicsSnapshot], *, start_index: int, object_id: str,
    target: TargetGeometry, phase: str,
) -> int:
    """Find the first delayed two-frame stable release after an attempted placement."""
    start = max(0, int(start_index))
    for end in range(start + 1, len(timeline)):
        pair = timeline[end - 1:end + 1]
        if all(
            _named_contact(snapshot, object_id=object_id, target_id=target.target_id)
            and not _named_contact(snapshot, object_id=object_id, gripper=True)
            and _release_relation(snapshot, object_id, target, phase)
            for snapshot in pair
        ):
            return end
    return min(start + 1, len(timeline) - 1)


def controlled_release_is_recently_held(
    timeline: Sequence[PhysicsSnapshot], *, release_index: int,
    object_id: str, max_gap: int = 6,
) -> bool:
    """Reject an object that free-falls long before reaching its target."""
    end = max(0, int(release_index))
    start = max(0, end - int(max_gap))
    return any(
        _named_contact(timeline[index], object_id=object_id, gripper=True)
        for index in range(start, end)
    )


def controlled_release_from_target_contact(
    timeline: Sequence[PhysicsSnapshot],
    *,
    release_index: int,
    object_id: str,
    target_id: str,
) -> bool:
    """Accept a release initiated while the held object already touches its target."""
    end = min(max(0, int(release_index)), len(timeline) - 1)
    held_on_target = any(
        _named_contact(snapshot, object_id=object_id, gripper=True)
        and _named_contact(snapshot, object_id=object_id, target_id=target_id)
        for snapshot in timeline[:end]
    )
    released_on_target = any(
        not _named_contact(snapshot, object_id=object_id, gripper=True)
        and _named_contact(snapshot, object_id=object_id, target_id=target_id)
        for snapshot in timeline[:end + 1]
    )
    return bool(held_on_target and released_on_target)


def _joint_delta(previous: PhysicsSnapshot, current: PhysicsSnapshot, joint_id: str) -> float:
    return float(current.joint_positions.get(joint_id, 0.0) - previous.joint_positions.get(joint_id, 0.0))


def _phase_event(phase: str, snapshots: Sequence[PhysicsSnapshot], index: int, object_id: str) -> bool:
    current = snapshots[index]
    previous = snapshots[index - 1] if index else current
    target = _target_for(current)
    if phase == "grasp":
        # MuJoCo's finger controller often closes in several sub-threshold
        # solver steps.  Permit their accumulated closure only while the same
        # exact gripper/object contact persists; this cannot be satisfied by
        # a visual label or an unrelated gripper motion.
        if not _named_contact(current, object_id=object_id, gripper=True):
            return False
        closure = max(_gripper_width(snapshot) for snapshot in snapshots[:index + 1]) - _gripper_width(current)
        threshold = float(PHASE_CONTRACTS["grasp"]["thresholds"]["closure"])  # type: ignore[index]
        return opposing_finger_grasp_sufficient(
            current, object_id=object_id, closure=closure,
            nominal_closure=threshold,
        )
    if phase in {"wipe", "scrub"}:
        target = _target_for_object_contact(current, object_id)
        return target is not None and _named_contact(current, object_id=object_id, target_id=target.target_id, gripper=False)
    if phase == "spray":
        return any(contact.object_id == object_id and ("trigger" in contact.counterpart.lower() or "trigger" in contact.actor.lower()) for contact in current.contact_evidence)
    if phase == "press":
        return any(contact.object_id == object_id and ("button" in contact.counterpart.lower() or "trigger" in contact.counterpart.lower() or "dispenser" in contact.counterpart.lower()) for contact in current.contact_evidence)
    if phase in {"place", "return", "deposit", "insert", "stack", "arrange"}:
        # Segmentation finds the next attempted release event. Exact requested
        # target/volume validation happens after allocation, so a wrong
        # same-class fixture becomes an explicit predicate error rather than
        # disappearing as an uninformative missing phase.
        target = _target_for_object_contact(current, object_id)
        prior_held = controlled_release_is_recently_held(
            snapshots, release_index=index, object_id=object_id,
        )
        stable_support = (
            target is not None
            and index + 1 < len(snapshots)
            and all(
                _named_contact(snapshot, object_id=object_id, target_id=target.target_id)
                and not _named_contact(snapshot, object_id=object_id, gripper=True)
                for snapshot in snapshots[index:index + 2]
            )
        )
        return bool(prior_held and stable_support)
    if phase in {"open", "close", "close_lid", "pull", "push", "turn_knob"}:
        if index == 0:
            return False
        expected_joint = current.fixture_joint_ids.get(phase)
        joint_ids = {expected_joint} if expected_joint else {contact.joint_id for contact in current.contact_evidence if contact.joint_id}
        # A real OSC controller commonly turns a fixture over several small
        # integration ticks. Require a *preceding exact contact* and the same
        # exact contact at this tick, then evaluate accumulated post-contact
        # joint movement rather than demanding one unsafe >=0.15 jump.
        for joint_id in joint_ids:
            motion_threshold = fixture_motion_threshold(
                phase, float(snapshots[0].joint_positions.get(joint_id, 0.0)),
            )
            if (
                abs(_joint_delta(previous, current, joint_id)) >= motion_threshold
                and _named_contact(previous, object_id=object_id, joint_id=joint_id, gripper=True)
                and _named_contact(current, object_id=object_id, joint_id=joint_id, gripper=True)
            ):
                return True
            contact_steps = [
                step for step in range(index)
                if _named_contact(snapshots[step], object_id=object_id, joint_id=joint_id, gripper=True)
            ]
            if not contact_steps or not _named_contact(current, object_id=object_id, joint_id=joint_id, gripper=True):
                continue
            contact_step = contact_steps[0]
            accumulated = abs(float(current.joint_positions.get(joint_id, 0.0)) - float(snapshots[contact_step].joint_positions.get(joint_id, 0.0)))
            if accumulated >= motion_threshold:
                return True
        return False
    return False


def segment_history(spec: OperationSpec, history: Sequence[PhysicsSnapshot]) -> tuple[PhaseWindow, ...]:
    """Scan once and allocate one non-overlapping increasing event to each phase.

    Allocation deliberately advances past the selected event, so repeated phases
    cannot reuse the same earliest contact and a pull→close action can be proved
    even when its terminal joint position equals its initial position.
    """
    ordered = tuple(sorted(history, key=lambda item: item.step))
    windows: list[PhaseWindow] = []
    cursor = 1
    for phase in spec.phases:
        object_id = _entity_for(0)
        selected: int | None = None
        for index in range(cursor, len(ordered)):
            if _phase_event(phase, ordered, index, object_id):
                selected = index
                break
        if selected is None:
            continue
        window_end = selected
        if phase == "grasp":
            # A grasp may only consume the first lift causally following its
            # exact gripper-contact transition.  Comparing against the phase
            # start can accidentally absorb an earlier motion and blur the
            # following wipe segment.
            contact_baseline = _z(ordered[max(cursor - 1, selected - 1)], object_id)
            while window_end + 1 < len(ordered) and max(_z(snapshot, object_id) for snapshot in ordered[selected:window_end + 1]) - contact_baseline < float(PHASE_CONTRACTS["grasp"]["thresholds"]["lift"]):  # type: ignore[index]
                window_end += 1
        if phase == "spray":
            start_coverage = ordered[cursor - 1].spray_coverage
            while window_end + 1 < len(ordered) and ordered[window_end].spray_coverage - start_coverage < float(PHASE_CONTRACTS["spray"]["thresholds"]["coverage"]):  # type: ignore[index]
                window_end += 1
        if phase == "press":
            start_amount = ordered[cursor - 1].dispensed_amount
            while window_end + 1 < len(ordered) and ordered[window_end].dispensed_amount - start_amount < float(PHASE_CONTRACTS["press"]["thresholds"]["dispensed_amount"]):  # type: ignore[index]
                window_end += 1
        if phase in {"open", "close", "close_lid", "pull", "push", "turn_knob"}:
            joint_id = ordered[selected].fixture_joint_ids.get(phase)
            while joint_id and window_end + 1 < len(ordered):
                values = [float(snapshot.joint_positions.get(joint_id, 0.0)) for snapshot in ordered[cursor - 1:window_end + 1]]
                motion_threshold = fixture_motion_threshold(phase, values[0])
                target_reached = fixture_target_reached(phase, values)
                if max(values) - min(values) >= motion_threshold and target_reached:
                    break
                window_end += 1
        if phase in {"place", "return", "deposit", "insert", "stack", "arrange", "wipe", "scrub"} and selected + 1 < len(ordered):
            target = _target_for_object_contact(ordered[selected], object_id)
            while target is not None and window_end + 1 < len(ordered) and _named_contact(ordered[window_end + 1], object_id=object_id, target_id=target.target_id):
                window_end += 1
        windows.append(PhaseWindow(phase, ordered[cursor - 1].step, ordered[window_end].step, tuple(ordered[cursor - 1:window_end + 1])))
        cursor = window_end + 1
    return tuple(windows)


def _segment_phase_object_windows(spec: OperationSpec, timeline: Sequence[PhysicsSnapshot]) -> tuple[tuple[str, tuple[PhaseWindow, ...]], ...]:
    """Allocate a distinct increasing evidence window for every phase/object."""
    cursor = 1
    rows: list[tuple[str, tuple[PhaseWindow, ...]]] = []
    for phase_index, phase in enumerate(spec.phases):
        object_windows: list[PhaseWindow] = []
        for object_index, _ in enumerate(spec.manipulated_objects):
            object_id = _entity_for(object_index)
            selected = next((index for index in range(cursor, len(timeline)) if _phase_event(phase, timeline, index, object_id)), None)
            if selected is None:
                continue
            # One following snapshot is reserved for release/support stability;
            # the global cursor then advances past it, preventing reuse.
            end = selected
            if phase == "grasp":
                # Keep the grasp proof to the first lift following the exact
                # finger contact selected above.  This establishes a strict
                # grasp -> wipe boundary without attributing cleaning motion
                # to the grasp phase.
                contact_baseline = _z(timeline[max(cursor - 1, selected - 1)], object_id)
                while end + 1 < len(timeline) and max(_z(snapshot, object_id) for snapshot in timeline[selected:end + 1]) - contact_baseline < float(PHASE_CONTRACTS["grasp"]["thresholds"]["lift"]):  # type: ignore[index]
                    end += 1
            if phase == "spray":
                while end + 1 < len(timeline) and timeline[end].spray_coverage - timeline[cursor - 1].spray_coverage < float(PHASE_CONTRACTS["spray"]["thresholds"]["coverage"]):  # type: ignore[index]
                    end += 1
            if phase == "press":
                while end + 1 < len(timeline) and timeline[end].dispensed_amount - timeline[cursor - 1].dispensed_amount < float(PHASE_CONTRACTS["press"]["thresholds"]["dispensed_amount"]):  # type: ignore[index]
                    end += 1
            if phase in {"wipe", "scrub"}:
                target = _target_for_object_contact(timeline[selected], object_id)
                while target is not None and end + 1 < len(timeline) and _named_contact(timeline[end + 1], object_id=object_id, target_id=target.target_id):
                    end += 1
            if phase in {"open", "close", "close_lid", "pull", "push", "turn_knob"}:
                joint_id = timeline[selected].fixture_joint_ids.get(phase)
                while joint_id and end + 1 < len(timeline):
                    values = [float(snapshot.joint_positions.get(joint_id, 0.0)) for snapshot in timeline[cursor - 1:end + 1]]
                    motion_threshold = fixture_motion_threshold(phase, values[0])
                    contact_indexes = [
                        index for index, snapshot in enumerate(timeline[cursor - 1:end + 1])
                        if _named_contact(snapshot, object_id=object_id, joint_id=joint_id, gripper=True)
                    ]
                    target_reached = fixture_target_reached(phase, values)
                    contact_coupled_motion = any(
                        abs(values[-1] - values[contact_index]) >= motion_threshold
                        for contact_index in contact_indexes
                    )
                    if (
                        max(values) - min(values) >= motion_threshold
                        and contact_coupled_motion
                        and target_reached
                    ):
                        break
                    end += 1
            if phase in {"place", "return", "deposit", "insert", "stack", "arrange"}:
                target = _target_for_object_contact(timeline[selected], object_id)
                if target is not None:
                    end = stable_release_end_index(
                        timeline,
                        start_index=selected,
                        object_id=object_id,
                        target=target,
                        phase=phase,
                    )
            # Preserve the immediately preceding raw snapshot as the dirt
            # baseline.  Coverage itself still begins at ``selected``; the
            # baseline cannot be taken after derived cleaning has occurred.
            start = cursor - 1
            object_windows.append(PhaseWindow(phase, timeline[start].step, timeline[end].step, tuple(timeline[start:end + 1])))
            cursor = end + 1
        rows.append((phase, tuple(object_windows)))
    return tuple(rows)


def _strict_phase_object_result(
    phase: str,
    object_id: str,
    window: PhaseWindow,
    target: TargetGeometry | None,
    *,
    prior_grasp_evidence: bool = False,
) -> tuple[bool, dict[str, float | bool | str], tuple[str, ...]]:
    states = window.snapshots
    end = states[-1]
    start = states[0]
    transition = any(_named_contact(snapshot, object_id=object_id) for snapshot in states)
    errors: list[str] = []
    if not transition:
        errors.extend((f"object_contact_missing:{object_id}", f"object_evidence_missing:{object_id}"))
    if phase in {"grasp", "place", "return", "deposit", "insert", "stack", "arrange"}:
        heights = [_z(snapshot, object_id) for snapshot in states]
        movement = max(heights) - min(heights)
        minimum = PHASE_CONTRACTS[phase]["thresholds"].get("lift", PHASE_CONTRACTS[phase]["thresholds"].get("transit", 0.0))  # type: ignore[index,union-attr]
        if movement < float(minimum):
            errors.append(f"object_state_transition_missing:{object_id}")
    else:
        movement = 0.0
    stable = False
    metrics: dict[str, float | bool | str] = {"object_id": object_id, "state_transition": movement, "stable": stable}
    if phase == "grasp":
        closure = max(_gripper_width(snapshot) for snapshot in states) - min(_gripper_width(snapshot) for snapshot in states)
        contact = any(_named_contact(snapshot, object_id=object_id, gripper=True) for snapshot in states)
        metrics.update({"gripper_closure": closure, "gripper_object_contact": contact, "lift_distance": movement})
        nominal_closure = float(PHASE_CONTRACTS[phase]["thresholds"]["closure"])  # type: ignore[index]
        opposing = any(
            opposing_finger_grasp_sufficient(
                snapshot, object_id=object_id, closure=closure,
                nominal_closure=nominal_closure,
            )
            for snapshot in states
        )
        metrics["opposing_finger_contact"] = opposing
        if not opposing:
            errors.append("gripper_closure_insufficient")
    if target is not None and phase in {"place", "return", "deposit", "insert", "stack", "arrange"}:
        if not any(_named_contact(snapshot, object_id=object_id, gripper=True) for snapshot in states):
            errors.append("transit_or_grasp_evidence_missing")
        release_index = max(0, len(states) - 2)
        controlled_release = controlled_release_is_recently_held(
            states, release_index=release_index, object_id=object_id,
        ) or controlled_release_from_target_contact(
            states,
            release_index=release_index,
            object_id=object_id,
            target_id=target.target_id,
        )
        if not controlled_release:
            errors.append(f"controlled_release_missing:{target.target_id}")
        stable = len(states) >= 2 and all(
            _named_contact(snapshot, object_id=object_id, target_id=target.target_id)
            and _release_relation(snapshot, object_id, target, phase)
            for snapshot in states[-2:]
        )
        if not stable:
            errors.append(f"target_stability_missing:{target.target_id}")
        metrics.update({
            "stable_release": stable,
            "target_support_contact": stable,
            "controlled_release": controlled_release,
            "transit_distance": movement,
        })
        if phase == "deposit":
            metrics["inside_target_volume"] = stable
        if phase == "insert":
            metrics["containment"] = stable
        if phase == "stack":
            support_contact = all(_named_contact(snapshot, object_id=object_id, target_id=target.target_id) for snapshot in states[-2:])
            # The source annotations use both natural-language-equivalent
            # relations: ``on`` means the object is supported by the target,
            # while older compiled tasks used ``above``. Exact sustained
            # object-target contact is still required separately below.
            above_support = target.spatial_relation in {"on", "above"} and _release_relation(end, object_id, target, "stack")
            supported = stable and support_contact and bool(target.support_id) and above_support
            metrics.update({"stable_support": supported, "exact_support_contact": support_contact, "above_support": above_support})
            if not supported:
                errors.append("correct_support_missing")
        if phase == "arrange":
            center = (np.asarray(target.min_corner, dtype=float) + np.asarray(target.max_corner, dtype=float)) / 2.0
            point = np.asarray(end.body_poses.get(object_id, np.zeros(3)), dtype=float)[:3]
            relation = target.spatial_relation
            if relation in {"near", "beside"}:
                relation_ok = float(np.linalg.norm(point[:2] - center[:2])) <= 0.15
            elif relation == "above":
                relation_ok = point[2] > center[2]
            elif relation == "on":
                relation_ok = point[2] >= float(np.asarray(target.max_corner)[2]) - 0.03
            elif relation == "clean-region":
                relation_ok = _inside_target(end, object_id, target)
            else:  # inside / supported regions use containment as the relation.
                relation_ok = _inside_target(end, object_id, target)
            metrics["spatial_relation"] = relation_ok
            if not relation_ok:
                errors.append("spatial_relation_not_satisfied")
    if phase in {"wipe", "scrub"}:
        if target is None:
            errors.append("target_geometry_missing")
        else:
            contact_states = [
                snapshot for snapshot in states
                if _named_contact(snapshot, object_id=object_id, target_id=target.target_id)
            ]
            coverage = len(contact_states)
            cells = {
                surface_coverage_cell(snapshot.body_poses[object_id])
                for snapshot in contact_states
                if object_id in snapshot.body_poses and _inside_target(snapshot, object_id, target)
            }
            # A wipe begins at its first surface-contact frame.  Finger/tool
            # contact is therefore proven by the immediately preceding grasp
            # window, without reusing that frame as wipe coverage evidence.
            held = prior_grasp_evidence or any(_named_contact(snapshot, object_id=object_id, gripper=True) for snapshot in states)
            reduction = float(start.dirt_fraction) - float(end.dirt_fraction)
            if not held:
                errors.append("tool_grasp_evidence_missing")
            if coverage < 3:
                errors.append("surface_contact_coverage_insufficient")
            if len(cells) < int(PHASE_CONTRACTS[phase]["thresholds"]["spatial_cells"]):  # type: ignore[index]
                errors.append("surface_spatial_coverage_insufficient")
            if reduction < 0.30:
                errors.append("dirt_reduction_insufficient")
            return not errors, {
                **metrics,
                "tool_surface_contact_coverage": float(coverage),
                "spatial_coverage_cells": float(len(cells)),
                "dirt_reduction": reduction,
            }, tuple(errors)
    if phase == "spray":
        coverage = float(end.spray_coverage) - float(start.spray_coverage)
        trigger = any(
            contact.object_id == object_id
            and "gripper" in f"{contact.actor} {contact.counterpart}".lower()
            and ("trigger" in f"{contact.actor} {contact.counterpart}".lower() or "button" in f"{contact.actor} {contact.counterpart}".lower())
            for snapshot in states for contact in snapshot.contact_evidence
        )
        metrics.update({"spray_coverage": coverage, "trigger_contact": trigger})
        if not trigger:
            errors.append("trigger_contact_missing")
        if coverage < float(PHASE_CONTRACTS[phase]["thresholds"]["coverage"]):  # type: ignore[index]
            errors.append("spray_coverage_insufficient")
    if phase == "press":
        amount = float(end.dispensed_amount) - float(start.dispensed_amount)
        trigger = any(
            contact.object_id == object_id
            and "gripper" in f"{contact.actor} {contact.counterpart}".lower()
            and ("trigger" in f"{contact.actor} {contact.counterpart}".lower() or "button" in f"{contact.actor} {contact.counterpart}".lower() or "dispenser" in f"{contact.actor} {contact.counterpart}".lower())
            for snapshot in states for contact in snapshot.contact_evidence
        )
        metrics.update({"dispensed_amount": amount, "trigger_contact": trigger})
        if not trigger:
            errors.append("trigger_contact_missing")
        if amount < float(PHASE_CONTRACTS[phase]["thresholds"]["dispensed_amount"]):  # type: ignore[index]
            errors.append("dispensed_amount_insufficient")
    if phase in {"open", "close", "close_lid", "pull", "push", "turn_knob"}:
        joint_id = end.fixture_joint_ids.get(phase)
        if not joint_id:
            errors.append("fixture_joint_contract_missing")
        else:
            values = [float(snapshot.joint_positions.get(joint_id, 0.0)) for snapshot in states]
            motion = max(values) - min(values)
            motion_threshold = fixture_motion_threshold(phase, values[0])
            contact_indices = [index for index, snapshot in enumerate(states) if _named_contact(snapshot, object_id=object_id, joint_id=joint_id, gripper=True)]
            # The fixture can begin moving on the same integration tick that
            # creates the first physical contact.  Require that contact to be
            # established *before the authoritative accumulated 0.20 motion*,
            # rather than rejecting legitimate contact-coupled dynamics due to
            # a small initial seating displacement.
            contact_before_motion = any(
                any(
                    later > contact_index
                    and abs(values[later] - values[contact_index]) >= motion_threshold
                    for later in range(contact_index + 1, len(values))
                )
                for contact_index in contact_indices
            )
            target_ok = motion >= motion_threshold and fixture_target_reached(phase, values)
            metrics.update({"joint_target": target_ok, "joint_motion": motion, "fixture_contact": contact_before_motion})
            if not contact_before_motion:
                errors.append("fixture_contact_before_motion_missing")
            if not target_ok:
                errors.append("joint_target_not_reached")
    return not errors, metrics, tuple(errors)


def _strict_evaluate_operation(spec: OperationSpec, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    timeline = _timeline(initial, history, final)
    phase_object_windows = _segment_phase_object_windows(spec, timeline)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if any(not windows for _, windows in phase_object_windows):
        errors.append("phase_event_missing")
        errors.append("contact_missing")
        for phase in spec.phases:
            expected_joint = next((snapshot.fixture_joint_ids.get(phase) for snapshot in timeline if snapshot.fixture_joint_ids.get(phase)), None)
            if expected_joint and any(contact.joint_id and contact.joint_id != expected_joint for snapshot in timeline for contact in snapshot.contact_evidence):
                errors.append(f"fixture_joint_mismatch:{expected_joint}")
    # Diagnose reversed physical events explicitly instead of concealing them
    # behind a generic missing-phase result.
    # VLA82-002 begins with the sponge already resting on its declared counter
    # target.  Its initial physical support contact is not a wipe event; the
    # ordered per-phase windows below retain the strict grasp -> wipe -> return
    # proof.  Other tasks continue to receive the raw reversed-event audit.
    if spec.selection_id != "VLA82-002":
        earliest: dict[str, int] = {}
        for phase in dict.fromkeys(spec.phases):
            for index in range(1, len(timeline)):
                if _phase_event(phase, timeline, index, "obj"):
                    earliest[phase] = timeline[index].step
                    break
        for previous_phase, next_phase in zip(spec.phases, spec.phases[1:]):
            if previous_phase in earliest and next_phase in earliest and earliest[next_phase] < earliest[previous_phase]:
                errors.append("phase_order_violation")
    prior = -1
    order_complete = True
    grasp_evidence_by_object: dict[str, bool] = {}
    for phase_index, phase in enumerate(spec.phases):
        object_windows = phase_object_windows[phase_index][1]
        if not object_windows:
            errors.append(f"phase_failed:{phase}")
            order_complete = False
            continue
        window = object_windows[0]
        if order_complete and window.end_step <= prior:
            errors.append("phase_order_violation")
        prior = window.end_step
        target = _target_for_object_contact(window.snapshots[-1], "obj") or _target_for(window.snapshots[-1])
        expected_token = _FIXTURE_TOKENS.get(spec.selection_id)
        declared_joint = window.snapshots[-1].fixture_joint_ids.get(phase)
        if phase in {"open", "close", "close_lid", "pull", "push", "turn_knob"} and expected_token and (not declared_joint or expected_token not in declared_joint.lower()):
            errors.extend(("fixture_joint_mismatch", f"fixture_joint_mismatch:{declared_joint or 'missing'}"))
        object_rows = []
        for index, _ in enumerate(spec.manipulated_objects):
            object_id = _entity_for(index)
            if index >= len(object_windows):
                errors.extend((f"object_contact_missing:{object_id}", f"object_evidence_missing:{object_id}"))
                object_rows.append({"object": object_id, "evidence": False, "transition_step": None, "object_id": object_id, "state_transition": 0.0, "stable": False})
                continue
            object_window = object_windows[index]
            object_target = _target_for_object_contact(object_window.snapshots[-1], object_id) or _target_for(object_window.snapshots[-1])
            ok, metrics, object_errors = _strict_phase_object_result(
                phase,
                object_id,
                object_window,
                object_target,
                prior_grasp_evidence=grasp_evidence_by_object.get(object_id, False),
            )
            if phase == "grasp":
                grasp_evidence_by_object[object_id] = any(
                    _named_contact(snapshot, object_id=object_id, gripper=True)
                    for snapshot in object_window.snapshots
                )
            object_rows.append({"object": object_id, "evidence": ok, "transition_step": object_window.end_step, **metrics})
            errors.extend(object_errors)
        if target is None and phase in {"place", "return", "deposit", "insert", "stack", "arrange", "wipe", "scrub"}:
            errors.append("target_geometry_missing")
        if target is not None and phase in {"place", "return", "deposit", "insert", "stack", "arrange", "wipe", "scrub"} and not any(_named_contact(snapshot, object_id="obj", target_id=target.target_id) for snapshot in window.snapshots):
            errors.append(f"target_contact_missing:{target.target_id}")
        rows.append({"phase": phase, "transition_step": window.end_step, "object_results": tuple(object_rows)})
    contact = bool(rows) and not any(error.startswith("object_contact_missing") for error in errors)
    metrics: dict[str, float | bool | str] = {"phase_count": float(len(rows)), "required_phase_count": float(len(spec.phases)), "strict_physics_contract": True}
    for row in rows:
        phase = str(row["phase"])
        for object_result in row["object_results"]:
            for name in PHASE_CONTRACTS[phase]["metrics"]:  # type: ignore[index]
                value = object_result.get(name, False)
                metrics[f"{phase}.{object_result['object']}.{name}"] = value
                if object_result["object"] == "obj":
                    metrics[str(name)] = value
    if requires_upright_final_pose(spec.selection_id):
        upright_cosine = _object_vertical_cosine(final)
        upright = upright_cosine >= float(np.cos(np.deg2rad(12.0)))
        object_label = "spray_bottle" if spec.selection_id == "VLA82-004" else "water_bottle"
        metrics[f"{object_label}_upright_cosine"] = upright_cosine
        metrics[f"{object_label}_upright"] = upright
        if not upright:
            errors.append(f"{object_label}_not_upright")
    return PredicateResult(not errors, tuple(rows), metrics, contact, tuple(dict.fromkeys(errors)))


def evaluate_phase(phase: str, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot, *, spec: OperationSpec | None = None) -> PredicateResult:
    """Evaluate one source phase from a continuous, read-only physics history."""
    if phase not in PREDICATES:
        return _result(False, {"transition_step": 0.0}, contact=False, errors=(f"phase_unregistered:{phase}",), phase=phase)
    if spec is None:
        return _result(False, {"transition_step": 0.0}, contact=False, errors=("operation_spec_missing",), phase=phase)
    one_phase_spec = replace(spec, phases=(phase,), predicate_names=(f"{phase}_completed",))
    return _strict_evaluate_operation(one_phase_spec, initial, history, final)


def _object_evidence_entries(spec: OperationSpec, history: Sequence[PhysicsSnapshot]) -> tuple[dict[str, bool | str], ...]:
    """Return one independently auditable physics-evidence bit per object."""
    entries: list[dict[str, bool | str]] = []
    for index, name in enumerate(spec.manipulated_objects):
        key = "obj" if index == 0 else f"annotated_{index:02d}"
        if index == 0:
            found = any(snapshot.contacts for snapshot in history)
        else:
            lowered = key.lower()
            found = any(key in snapshot.body_poses or any(lowered in item for contact in snapshot.contacts for item in _names(contact)) for snapshot in history)
        entries.append({"object": name if index == 0 else key, "evidence": found})
    return tuple(entries)


def _object_evidence(spec: OperationSpec, history: Sequence[PhysicsSnapshot]) -> list[str]:
    return [
        f"object_evidence_missing:{entry['object']}"
        for entry in _object_evidence_entries(spec, history)
        if entry["evidence"] is not True
    ]


def combine_ordered_phase_results(spec: OperationSpec, results: Sequence[PredicateResult], history: Sequence[PhysicsSnapshot]) -> PredicateResult:
    """Combine phase outcomes while proving source order and object coverage."""
    errors: list[str] = []
    phase_rows: list[dict[str, Any]] = []
    prior_step = -1
    for phase, result in zip(spec.phases, results):
        row = dict(result.phase_results[0]) if result.phase_results else {"phase": phase}
        row["object_results"] = _object_evidence_entries(spec, history)
        phase_rows.append(row)
        if not result.success:
            errors.append(f"phase_failed:{phase}")
        step = row.get("transition_step")
        if not isinstance(step, int) or step <= 0:
            errors.append(f"phase_transition_missing:{phase}")
        elif step < prior_step:
            errors.append("phase_order_violation")
        else:
            prior_step = step
        errors.extend(result.errors)
    if len(results) != len(spec.phases):
        errors.append("phase_result_count_mismatch")
    errors.extend(_object_evidence(spec, history))
    contact = bool(results) and all(result.contact_verified for result in results)
    metrics: dict[str, float | bool | str] = {
        "phase_count": float(len(results)),
        "required_phase_count": float(len(spec.phases)),
        "object_count": float(len(spec.manipulated_objects)),
        "ordered": "phase_order_violation" not in errors,
    }
    return PredicateResult(not errors, tuple(phase_rows), metrics, contact, tuple(dict.fromkeys(errors)))


def _evaluate_vla001_three_can_deposit(
    spec: OperationSpec,
    initial: PhysicsSnapshot,
    history: Sequence[PhysicsSnapshot],
    final: PhysicsSnapshot,
) -> PredicateResult:
    """Prove the source video's three-can deposit as three physical loops.

    The source registry stores the object category once (trash can), while its
    verbatim instruction requires *three cans*.  A valid episode must therefore
    show an independent grasp, held lift/transit, target arrival, open-gripper
    release, and two-frame stable support for each live can body.  ``deposit``
    and ``place`` are reported as the transport/arrival and stable-release
    halves of those same three operations; no duplicate motion is invented.
    """
    timeline = _timeline(initial, history, final)
    object_ids = ("obj", "annotated_01", "annotated_02")
    target = next((item for snapshot in timeline for item in snapshot.target_geometries.values()), None)
    errors: list[str] = []
    deposit_rows: list[dict[str, Any]] = []
    place_rows: list[dict[str, Any]] = []
    prior_release_step = -1
    completed = 0

    if target is None:
        return PredicateResult(
            False, (), {"required_can_count": 3.0, "completed_can_count": 0.0, "strict_physics_contract": True},
            False, ("target_geometry_missing",),
        )

    for object_id in object_ids:
        initial_pose = initial.body_poses.get(object_id)
        if initial_pose is None:
            errors.append(f"object_evidence_missing:{object_id}")
            deposit_rows.append({"object": object_id, "evidence": False, "transition_step": None})
            place_rows.append({"object": object_id, "evidence": False, "transition_step": None})
            continue
        start = np.asarray(initial_pose, dtype=float)[:3]
        if _inside_target(initial, object_id, target):
            errors.append(f"object_initially_inside_target:{object_id}")

        held_indices = [
            index for index, snapshot in enumerate(timeline)
            if index > 0 and _named_contact(snapshot, object_id=object_id, gripper=True)
        ]
        if not held_indices:
            errors.extend((f"object_contact_missing:{object_id}", f"object_evidence_missing:{object_id}"))
            deposit_rows.append({"object": object_id, "evidence": False, "transition_step": None})
            place_rows.append({"object": object_id, "evidence": False, "transition_step": None})
            continue

        first_held = held_indices[0]
        # The simulator may reset with closed finger joints.  Measure closure
        # from the widest physically observed pre-contact approach state for
        # this grasp, never from reset metadata or a commanded target.
        precontact_open = max(_gripper_width(snapshot) for snapshot in timeline[:first_held + 1])
        closure = precontact_open - min(_gripper_width(timeline[index]) for index in held_indices)
        lift = max(float(np.asarray(timeline[index].body_poses[object_id], dtype=float)[2] - start[2]) for index in held_indices)
        transit = max(float(np.linalg.norm(np.asarray(timeline[index].body_poses[object_id], dtype=float)[:3] - start)) for index in held_indices)
        arrivals = [
            index for index in range(first_held + 1, len(timeline))
            if _inside_target(timeline[index], object_id, target)
            and _named_contact(timeline[index], object_id=object_id, target_id=target.target_id)
        ]
        arrival_index = arrivals[0] if arrivals else None
        release_index: int | None = None
        if arrival_index is not None:
            for index in range(arrival_index, len(timeline) - 1):
                pair = timeline[index:index + 2]
                if not all(
                    _inside_target(snapshot, object_id, target)
                    and _named_contact(snapshot, object_id=object_id, target_id=target.target_id)
                    and not _named_contact(snapshot, object_id=object_id, gripper=True)
                    for snapshot in pair
                ):
                    continue
                first = np.asarray(pair[0].body_poses[object_id], dtype=float)[:3]
                second = np.asarray(pair[1].body_poses[object_id], dtype=float)[:3]
                if float(np.linalg.norm(second - first)) <= .001:
                    release_index = index + 1
                    break

        object_errors: list[str] = []
        if closure < .010:
            object_errors.append(f"gripper_closure_insufficient:{object_id}")
        if lift < .045:
            object_errors.append(f"lift_insufficient:{object_id}")
        if transit < .025:
            object_errors.append(f"transit_insufficient:{object_id}")
        if arrival_index is None:
            object_errors.append(f"inside_target_transition_missing:{object_id}")
        if release_index is None:
            object_errors.append(f"stable_release_missing:{object_id}")
        elif timeline[release_index].step <= prior_release_step:
            object_errors.append("object_operation_order_violation")
        else:
            prior_release_step = timeline[release_index].step
        if not _inside_target(final, object_id, target):
            object_errors.append(f"final_containment_missing:{object_id}")

        errors.extend(object_errors)
        deposit_ok = not any(
            error.startswith(("gripper_closure_insufficient", "lift_insufficient", "transit_insufficient", "inside_target_transition_missing"))
            for error in object_errors
        )
        place_ok = release_index is not None and not any(
            error.startswith(("stable_release_missing", "final_containment_missing")) or error == "object_operation_order_violation"
            for error in object_errors
        )
        if deposit_ok and place_ok:
            completed += 1
        deposit_rows.append({
            "object": object_id, "object_id": object_id, "evidence": deposit_ok,
            "transition_step": None if arrival_index is None else timeline[arrival_index].step,
            "gripper_closure": closure, "lift_distance": lift, "transit_distance": transit,
            "inside_target_volume": arrival_index is not None,
        })
        place_rows.append({
            "object": object_id, "object_id": object_id, "evidence": place_ok,
            "transition_step": None if release_index is None else timeline[release_index].step,
            "stable_release": release_index is not None, "target_support_contact": release_index is not None,
        })

    deposit_step = max((int(row["transition_step"]) for row in deposit_rows if row.get("transition_step") is not None), default=None)
    place_step = max((int(row["transition_step"]) for row in place_rows if row.get("transition_step") is not None), default=None)
    rows = (
        {"phase": "deposit", "transition_step": deposit_step, "object_results": tuple(deposit_rows)},
        {"phase": "place", "transition_step": place_step, "object_results": tuple(place_rows)},
    )
    metrics: dict[str, float | bool | str] = {
        "required_can_count": 3.0, "completed_can_count": float(completed),
        "phase_count": 2.0, "required_phase_count": 2.0, "strict_physics_contract": True,
        "all_finally_contained": all(_inside_target(final, object_id, target) for object_id in object_ids),
    }
    return PredicateResult(not errors and completed == 3, rows, metrics, completed == 3, tuple(dict.fromkeys(errors)))


def evaluate_operation(spec: OperationSpec, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    """Evaluate every authoritative phase and require ordered object evidence."""
    if spec.selection_id == "VLA82-001" and tuple(spec.phases) == ("deposit", "place"):
        return _evaluate_vla001_three_can_deposit(spec, initial, history, final)
    return _strict_evaluate_operation(spec, initial, history, final)


# Public compatibility entry points all delegate to the strict segmented path.
# The older helpers above are retained only as private migration reference and
# are never registered or callable through the public predicate registry.
def _strict_public(phase: str, spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    if spec is None:
        return _result(False, {"transition_step": 0.0}, contact=False, errors=("operation_spec_missing",), phase=phase)
    return evaluate_phase(phase, initial, history, final, spec=spec)


def evaluate_grasp(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("grasp", spec, initial, history, final)


def evaluate_place(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("place", spec, initial, history, final)


def evaluate_deposit(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("deposit", spec, initial, history, final)


def evaluate_wipe(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("wipe", spec, initial, history, final)


def evaluate_spray(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("spray", spec, initial, history, final)


def evaluate_hinge(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot, *, phase: str = "open") -> PredicateResult:
    return _strict_public(phase, spec, initial, history, final)


def evaluate_slider(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot, *, phase: str = "pull") -> PredicateResult:
    return _strict_public(phase, spec, initial, history, final)


def evaluate_knob(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("turn_knob", spec, initial, history, final)


def evaluate_press(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("press", spec, initial, history, final)


def evaluate_composite(spec: OperationSpec | None, initial: PhysicsSnapshot, history: Sequence[PhysicsSnapshot], final: PhysicsSnapshot) -> PredicateResult:
    return _strict_public("composite", spec, initial, history, final)


PREDICATES = {
    phase: (lambda spec, initial, history, final, _phase=phase: _strict_public(_phase, spec, initial, history, final))
    for phase in PHASE_CONTRACTS
}


def combine_ordered_phase_results(spec: OperationSpec, results: Sequence[PredicateResult], history: Sequence[PhysicsSnapshot]) -> PredicateResult:
    """Compatibility API that refuses to combine precomputed loose outcomes."""
    if len(history) < 2:
        return _result(False, {"phase_count": 0.0}, contact=False, errors=("strict_history_missing",), phase="")
    return _strict_evaluate_operation(spec, history[0], history[1:-1], history[-1])
