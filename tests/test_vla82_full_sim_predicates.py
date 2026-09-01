"""Physical-evidence contracts for VLA82 operation predicates."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tools.vla82_full_sim.annotations import OperationSpec, compile_all
from tools.vla82_full_sim.environment import ContactEvidence, PhysicsSnapshot, TargetGeometry
from tools.vla82_full_sim.predicates import (
    PHASE_CONTRACTS,
    PREDICATES,
    combine_ordered_phase_results,
    evaluate_operation,
    evaluate_phase,
    segment_history,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs" / "midterm_testing_vla82" / "vla82_midterm_registry.json"


def snapshot(
    step: int,
    *,
    object_z: float = 0.02,
    gripper: float = 0.08,
    contacts: tuple[tuple[str, str, float], ...] = (),
    joint: dict[str, float] | None = None,
    dirt: float = 1.0,
    spray: float = 0.0,
    dispensed: float = 0.0,
    extra_bodies: dict[str, np.ndarray] | None = None,
) -> PhysicsSnapshot:
    bodies = {"obj": np.array([0.0, 0.0, object_z, 1.0, 0.0, 0.0, 0.0])}
    bodies.update(extra_bodies or {})
    evidence = []
    targets: dict[str, TargetGeometry] = {}
    joint_id = next(iter((joint or {}).keys()), None)
    for first, second, distance in contacts:
        names = (first.lower(), second.lower())
        # This fixture states the object identity explicitly for every raw
        # physical contact; production snapshots must supply the same binding.
        object_id = "obj"
        target_id = next((name for name in names if any(token in name for token in ("target", "surface", "bin", "receptacle", "support", "reference", "home"))), None)
        evidence.append(ContactEvidence(first, second, object_id, target_id, "fixture" if joint_id else None, joint_id, distance))
        if target_id:
            is_support = "support" in target_id
            targets[target_id] = TargetGeometry(
                target_id, "fixture", (-1.0, -1.0, -0.05), (1.0, 1.0, 0.025) if is_support else (1.0, 1.0, 1.0),
                "support", "above" if is_support else "near",
            )
    return PhysicsSnapshot(
        step=step,
        robot_qpos=np.zeros(7),
        gripper_qpos=np.array([gripper]),
        body_poses=bodies,
        joint_positions=joint or {},
        contacts=contacts,
        dirt_fraction=dirt,
        spray_coverage=spray,
        dispensed_amount=dispensed,
        contact_evidence=tuple(evidence),
        target_geometries=targets,
        fixture_joint_ids={phase: joint_id for phase in ("open", "close", "close_lid", "pull", "push", "turn_knob") if joint_id},
    )


def spec(phases: tuple[str, ...], *, objects: tuple[str, ...] = ("obj",), selection_id: str = "VLA82-001") -> OperationSpec:
    return OperationSpec(
        selection_id=selection_id,
        task="test",
        object_name="obj",
        operation_text="source-backed test operation",
        source_kind="test",
        source_path=str(REGISTRY),
        phases=phases,
        manipulated_objects=objects,
        predicate_names=tuple(f"{phase}_completed" for phase in phases),
        source_sha256="test",
    )


def vla001_three_can_history(*, omit: str | None = None) -> tuple[PhysicsSnapshot, ...]:
    """Synthetic read-only trace matching three sequential physical deposits."""
    object_ids = ("obj", "annotated_01", "annotated_02")
    starts = {
        "obj": np.array((-.10, .00, .855)),
        "annotated_01": np.array((-.10, .10, .855)),
        "annotated_02": np.array((-.10, .20, .855)),
    }
    slots = {
        "obj": np.array((.22, .37, .873)),
        "annotated_01": np.array((.32, .37, .873)),
        "annotated_02": np.array((.27, .46, .873)),
    }
    target = TargetGeometry("trash_can:inner_cavity", "trash_can", (.163, .313, .81), (.377, .527, 1.11), "bottom", "inside")
    current = {key: value.copy() for key, value in starts.items()}

    def state(step: int, *, active: str | None = None, gripper: bool = False, target_contact: bool = False) -> PhysicsSnapshot:
        evidence: list[ContactEvidence] = []
        contacts: list[tuple[str, str, float]] = []
        if active is not None and active != omit:
            if gripper:
                evidence.append(ContactEvidence("gripper0_right_pad", f"{active}_collision", active, None, None, None, -.001))
                contacts.append(("gripper0_right_pad", f"{active}_collision", -.001))
            if target_contact:
                evidence.append(ContactEvidence(f"{active}_collision", "trash_can_bottom", active, target.target_id, "trash_can", None, -.001))
                contacts.append((f"{active}_collision", "trash_can_bottom", -.001))
        bodies = {key: np.r_[value, (1., 0., 0., 0.)] for key, value in current.items()}
        return PhysicsSnapshot(
            step=step, robot_qpos=np.zeros(7),
            gripper_qpos=np.array((.012, -.012)) if gripper else np.array((.04, -.04)),
            body_poses=bodies, joint_positions={}, contacts=tuple(contacts),
            dirt_fraction=1., spray_coverage=0., dispensed_amount=0.,
            contact_evidence=tuple(evidence), target_geometries={target.target_id: target},
        )

    rows = [state(0)]
    step = 1
    for object_id in object_ids:
        if object_id == omit:
            continue
        rows.append(state(step, active=object_id, gripper=True)); step += 1
        current[object_id] = starts[object_id] + np.array((0., 0., .07))
        rows.append(state(step, active=object_id, gripper=True)); step += 1
        current[object_id] = slots[object_id] + np.array((0., 0., .20))
        rows.append(state(step, active=object_id, gripper=True)); step += 1
        current[object_id] = slots[object_id].copy()
        rows.append(state(step, active=object_id, gripper=True, target_contact=True)); step += 1
        rows.append(state(step, active=object_id, target_contact=True)); step += 1
        current[object_id] = current[object_id] + np.array((.0002, -.0001, 0.))
        rows.append(state(step, active=object_id, target_contact=True)); step += 1
    return tuple(rows)


def test_vla001_full_operation_requires_and_accepts_three_independent_can_deposits() -> None:
    history = vla001_three_can_history()
    result = evaluate_operation(spec(("deposit", "place")), history[0], history[1:], history[-1])

    assert result.success is True
    for phase in ("deposit", "place"):
        row = next(item for item in result.phase_results if item["phase"] == phase)
        assert [item["object"] for item in row["object_results"]] == ["obj", "annotated_01", "annotated_02"]
        assert all(item["evidence"] for item in row["object_results"])
    assert result.metrics["required_can_count"] == 3.0
    assert result.metrics["completed_can_count"] == 3.0


def test_vla001_full_operation_rejects_missing_third_can_even_when_two_are_valid() -> None:
    history = vla001_three_can_history(omit="annotated_02")
    result = evaluate_operation(spec(("deposit", "place")), history[0], history[1:], history[-1])

    assert result.success is False
    assert "object_evidence_missing:annotated_02" in result.errors


def test_vla001_gripper_closure_uses_each_grasp_precontact_open_width_not_reset_width() -> None:
    history = vla001_three_can_history()
    # Panda can reset with closed joints, then physically open during approach.
    shifted = [replace(history[0], gripper_qpos=np.zeros(2))]
    shifted.append(replace(history[0], step=1, gripper_qpos=np.array((.04, -.04))))
    shifted.extend(replace(item, step=item.step + 1) for item in history[1:])

    result = evaluate_operation(spec(("deposit", "place")), shifted[0], shifted[1:], shifted[-1])

    assert result.success is True
    assert not any("gripper_closure_insufficient" in error for error in result.errors)


def grasp_history() -> tuple[PhysicsSnapshot, PhysicsSnapshot, PhysicsSnapshot]:
    initial = snapshot(0)
    contact = snapshot(4, object_z=0.03, gripper=0.01, contacts=(("robot0_gripper", "obj", -0.001),))
    final = snapshot(8, object_z=0.10, gripper=0.01, contacts=(("robot0_gripper", "obj", -0.001),))
    return initial, contact, final


def release_history(*, target: str = "target_bin") -> tuple[PhysicsSnapshot, ...]:
    initial, grasped, lifted = grasp_history()
    released = snapshot(
        12, object_z=0.05, gripper=0.08,
        contacts=(("obj", target, -0.001),),
    )
    stable = snapshot(
        15, object_z=0.05, gripper=0.08,
        contacts=(("obj", target, -0.001),),
    )
    return initial, grasped, lifted, released, stable


def cleaning_history(*, phase: str) -> tuple[PhysicsSnapshot, ...]:
    initial = snapshot(0, dirt=1.0)
    return (
        initial,
        snapshot(2, gripper=0.01, contacts=(("robot0_gripper", "obj", -0.001),), dirt=0.98),
        snapshot(4, gripper=0.01, contacts=(("obj", "cleaning_surface", -0.001),), dirt=0.75, extra_bodies={"obj": np.array((0.00, 0.00, 0.02, 1, 0, 0, 0))}),
        snapshot(6, gripper=0.01, contacts=(("obj", "cleaning_surface", -0.001),), dirt=0.50, extra_bodies={"obj": np.array((0.04, 0.00, 0.02, 1, 0, 0, 0))}),
        snapshot(8, gripper=0.01, contacts=(("obj", "cleaning_surface", -0.001),), dirt=0.20, extra_bodies={"obj": np.array((0.08, 0.00, 0.02, 1, 0, 0, 0))}),
    )


def test_grasp_window_ends_at_first_contact_coupled_lift_before_wipe():
    from tools.vla82_full_sim.predicates import _segment_phase_object_windows
    initial, contact, lifted = grasp_history()
    wiped = snapshot(12, contacts=(("obj", "cleaning_surface", -.001),), dirt=.7, extra_bodies={"obj": np.array((.04,0,.10,1,0,0,0))})
    windows = _segment_phase_object_windows(spec(("grasp", "wipe")), [initial, contact, lifted, wiped])
    grasp = windows[0][1][0]
    wipe = windows[1][1][0]
    assert grasp.end_step == lifted.step
    assert wipe.start_step == lifted.step


def test_grasp_phase_event_accepts_cumulative_closure_during_persistent_exact_contact():
    from tools.vla82_full_sim.predicates import _phase_event

    initial = snapshot(0, gripper=.08)
    first_contact = snapshot(2, gripper=.07, contacts=(("robot0_gripper", "obj", -.001),))
    continued_contact = snapshot(3, gripper=.06, contacts=(("robot0_gripper", "obj", -.001),))

    assert _phase_event("grasp", (initial, first_contact, continued_contact), 2, "obj") is True


def test_placement_events_reject_early_support_contact_until_release_is_stable():
    from tools.vla82_full_sim.predicates import _phase_event

    timeline = (
        snapshot(0),
        snapshot(2, gripper=.01, contacts=(("robot0_gripper", "obj", -.001),)),
        # Wiping may legitimately touch the source support while still held.
        snapshot(4, object_z=.08, gripper=.01, contacts=(("robot0_gripper", "obj", -.001), ("obj", "cleaning_surface", -.001))),
        snapshot(6, object_z=.12, gripper=.01, contacts=(("robot0_gripper", "obj", -.001),)),
        # The actual return is only the released, consecutively supported end.
        snapshot(8, object_z=.02, gripper=.08, contacts=(("obj", "cleaning_surface", -.001),)),
        snapshot(10, object_z=.02, gripper=.08, contacts=(("obj", "cleaning_surface", -.001),)),
    )

    assert _phase_event("return", timeline, 2, "obj") is False
    assert _phase_event("return", timeline, 4, "obj") is True
    assert _phase_event("place", timeline, 2, "obj") is False
    assert _phase_event("place", timeline, 4, "obj") is True


def test_water_bottle_place_rejects_a_sideways_final_pose():
    history = list(release_history(target="target_surface"))
    tipped = np.array((0.0, 0.0, 0.05, np.sqrt(.5), 0.0, np.sqrt(.5), 0.0))
    history[-1] = replace(history[-1], body_poses={"obj": tipped})

    result = evaluate_operation(
        spec(("place",), selection_id="VLA82-017"),
        history[0], history[1:], history[-1],
    )

    assert result.success is False
    assert "water_bottle_not_upright" in result.errors


def test_gripper_width_is_total_two_finger_aperture_not_single_finger_mean():
    from tools.vla82_full_sim.predicates import _gripper_width

    state = snapshot(1)
    state = replace(state, gripper_qpos=np.array((.03, -.03)))
    assert _gripper_width(state) == pytest.approx(.06)


def test_exact_object_contact_accepts_real_panda_finger_geom_as_gripper_contact() -> None:
    from tools.vla82_full_sim.predicates import _named_contact
    state = snapshot(1, contacts=(("gripper0_right_finger2_collision", "obj_collision", -0.001),))
    assert _named_contact(state, object_id="obj", gripper=True)


def spray_history() -> tuple[PhysicsSnapshot, ...]:
    return (
        snapshot(0),
        snapshot(2, gripper=0.01, contacts=(("robot0_gripper", "spray_trigger", -0.001),)),
        snapshot(4, gripper=0.01, contacts=(("obj", "cleaning_surface", -0.001),), spray=0.55),
    )


def joint_history(name: str, start: float, end: float) -> tuple[PhysicsSnapshot, ...]:
    return (
        snapshot(0, joint={name: start}),
        snapshot(1, gripper=0.01, contacts=(("robot0_gripper", name.replace("_joint", "_handle"), -0.001),), joint={name: start}),
        snapshot(3, gripper=0.01, contacts=(("robot0_gripper", name.replace("_joint", "_handle"), -0.001),), joint={name: (start + end) / 2}),
        snapshot(7, gripper=0.01, contacts=(("robot0_gripper", name.replace("_joint", "_handle"), -0.001),), joint={name: end}),
    )


@pytest.mark.parametrize(
    ("phase", "history", "expected_metric", "selection_id"),
    [
        ("grasp", grasp_history(), "lift_distance", "VLA82-001"),
        ("place", release_history(target="target_surface"), "stable_release", "VLA82-001"),
        ("return", release_history(target="home_surface"), "stable_release", "VLA82-001"),
        ("deposit", release_history(target="target_bin"), "inside_target_volume", "VLA82-001"),
        ("wipe", cleaning_history(phase="wipe"), "dirt_reduction", "VLA82-001"),
        ("scrub", cleaning_history(phase="scrub"), "dirt_reduction", "VLA82-001"),
        ("spray", spray_history(), "spray_coverage", "VLA82-001"),
        ("insert", release_history(target="target_receptacle"), "containment", "VLA82-001"),
        ("stack", release_history(target="support_object"), "stable_support", "VLA82-001"),
        ("arrange", release_history(target="target_reference"), "spatial_relation", "VLA82-001"),
        ("open", joint_history("oven_door_joint", 0.0, 1.0), "joint_target", "VLA82-008"),
        ("close", joint_history("oven_door_joint", 1.0, 0.0), "joint_target", "VLA82-008"),
        ("pull", joint_history("toasteroven_rack_joint", 0.0, 0.35), "joint_target", "VLA82-009"),
        ("push", joint_history("toasteroven_rack_joint", 0.35, 0.0), "joint_target", "VLA82-009"),
        ("turn_knob", joint_history("stove_knob_joint", 0.0, 0.8), "joint_target", "VLA82-007"),
        ("press", (
            snapshot(0),
            snapshot(3, gripper=0.01, contacts=(("robot0_gripper", "soap_dispenser_button", -0.001),)),
            snapshot(6, gripper=0.01, contacts=(("robot0_gripper", "soap_dispenser_button", -0.001),), dispensed=1.0),
        ), "dispensed_amount", "VLA82-001"),
        ("close_lid", joint_history("laptop_lid_joint", 1.0, 0.0), "joint_target", "VLA82-049"),
    ],
)
def test_each_operation_family_requires_physical_contact_and_metric(phase, history, expected_metric, selection_id):
    result = evaluate_phase(phase, history[0], history[1:], history[-1], spec=spec((phase,), selection_id=selection_id))
    assert result.success is True
    assert expected_metric in result.metrics
    assert result.contact_verified is True


def test_identical_terminal_state_without_contact_fails():
    initial = snapshot(0)
    final = snapshot(10, object_z=0.02)
    result = evaluate_phase("deposit", initial, (final,), final, spec=spec(("deposit",)))
    assert result.success is False
    assert "contact_missing" in result.errors


def test_direct_teleport_final_state_only_fails_even_if_target_contact_is_present():
    initial = snapshot(0)
    teleported = snapshot(10, object_z=0.05, gripper=0.08, contacts=(("obj", "target_bin", -0.001),))
    result = evaluate_phase("deposit", initial, (teleported,), teleported, spec=spec(("deposit",)))
    assert result.success is False
    assert "transit_or_grasp_evidence_missing" in result.errors


def test_source_textured_mujoco_geom_prefix_is_treated_as_the_manipulated_object():
    initial = snapshot(0)
    grasped = snapshot(3, object_z=0.03, gripper=0.01, contacts=(("robot0_gripper", "MCJFObj_0_collision", -0.001),))
    lifted = snapshot(7, object_z=0.10, gripper=0.01, contacts=(("robot0_gripper", "MCJFObj_0_collision", -0.001),))
    released = snapshot(11, object_z=0.05, gripper=0.08, contacts=(("MCJFObj_0_collision", "target_bin", -0.001),))
    stable = snapshot(14, object_z=0.05, gripper=0.08, contacts=(("MCJFObj_0_collision", "target_bin", -0.001),))
    result = evaluate_phase("deposit", initial, (grasped, lifted, released, stable), stable, spec=spec(("deposit",)))
    assert result.success is True


def test_turn_knob_accepts_cumulative_post_contact_joint_motion_not_only_one_large_control_tick():
    history = (
        snapshot(0, joint={"stove_knob_joint": 0.0}),
        snapshot(1, gripper=0.01, contacts=(("robot0_gripper", "stove_knob_handle", -0.001),), joint={"stove_knob_joint": 0.0}),
        snapshot(2, gripper=0.01, contacts=(("robot0_gripper", "stove_knob_handle", -0.001),), joint={"stove_knob_joint": 0.08}),
        snapshot(3, gripper=0.01, contacts=(("robot0_gripper", "stove_knob_handle", -0.001),), joint={"stove_knob_joint": 0.16}),
        snapshot(4, gripper=0.01, contacts=(("robot0_gripper", "stove_knob_handle", -0.001),), joint={"stove_knob_joint": 0.25}),
    )
    result = evaluate_operation(spec(("turn_knob",), selection_id="VLA82-007"), history[0], history[1:], history[-1])
    assert result.success is True
    assert result.phase_results[0]["transition_step"] == 4


def test_composite_fails_when_a_required_phase_is_missing():
    history = grasp_history()
    result = evaluate_operation(spec(("grasp", "wipe", "return")), history[0], history[1:], history[-1])
    assert result.success is False
    assert "phase_failed:wipe" in result.errors


def test_composite_rejects_out_of_order_observed_phase_transitions():
    history = (
        snapshot(0),
        snapshot(2, gripper=0.08, contacts=(("obj", "cleaning_surface", -0.001),), dirt=0.4),
        snapshot(5, object_z=0.1, gripper=0.01, contacts=(("robot0_gripper", "obj", -0.001),), dirt=0.4),
    )
    result = evaluate_operation(spec(("grasp", "wipe")), history[0], history[1:], history[-1])
    assert result.success is False
    assert "phase_order_violation" in result.errors


def test_wipe_requires_spatially_distinct_surface_coverage_not_repeated_contact_frames():
    history = (
        snapshot(0, dirt=1.0),
        snapshot(2, gripper=0.01, contacts=(("robot0_gripper", "obj", -0.001),), dirt=0.98),
        snapshot(4, gripper=0.01, contacts=(("obj", "cleaning_surface", -0.001),), dirt=0.70),
        snapshot(6, gripper=0.01, contacts=(("obj", "cleaning_surface", -0.001),), dirt=0.45),
        snapshot(8, gripper=0.01, contacts=(("obj", "cleaning_surface", -0.001),), dirt=0.20),
    )

    result = evaluate_phase("wipe", history[0], history[1:], history[-1], spec=spec(("wipe",)))

    assert result.success is False
    assert "surface_spatial_coverage_insufficient" in result.errors


def test_wipe_grid_uses_same_floor_quantization_as_live_coverage_capture() -> None:
    from tools.vla82_full_sim.predicates import surface_coverage_cell

    # Rounding merges the first two 2-cm cells into zero; capture uses a
    # floor-based grid, so predicate verification must use the same geometry.
    cells = {surface_coverage_cell((x, 0.0, 0.0)) for x in (-0.009, 0.009, 0.029)}
    assert cells == {(-1, 0), (0, 0), (1, 0)}


def test_stack_requires_the_declared_above_support_relation_in_addition_to_contact():
    history = release_history(target="support_object")
    invalid = tuple(
        replace(
            item,
            target_geometries={
                target_id: replace(target, spatial_relation="on")
                for target_id, target in item.target_geometries.items()
            },
        )
        for item in history
    )

    result = evaluate_phase("stack", invalid[0], invalid[1:], invalid[-1], spec=spec(("stack",)))

    assert result.success is False
    assert "correct_support_missing" in result.errors


def test_multi_object_operation_requires_independent_evidence_for_every_object():
    history = release_history(target="target_bin")
    result = evaluate_operation(spec(("deposit",), objects=("obj", "annotated_01")), history[0], history[1:], history[-1])
    assert result.success is False
    assert "object_evidence_missing:annotated_01" in result.errors


def test_each_phase_row_records_auditable_per_object_evidence_result():
    history = release_history(target="target_bin")
    source = spec(("deposit",), objects=("obj", "annotated_01"))
    result = evaluate_operation(source, history[0], history[1:], history[-1])
    object_results = result.phase_results[0]["object_results"]
    assert object_results[0]["object"] == "obj"
    assert object_results[0]["evidence"] is True
    assert object_results[1]["object"] == "annotated_01"
    assert object_results[1]["evidence"] is False


def test_wrong_fixture_joint_cannot_satisfy_the_selection_specific_predicate():
    history = joint_history("oven_door_joint", 0.0, 0.8)
    result = evaluate_phase("turn_knob", history[0], history[1:], history[-1], spec=spec(("turn_knob",), selection_id="VLA82-007"))
    assert result.success is False
    assert "fixture_joint_mismatch" in result.errors


def test_all_authoritative_phases_are_registered_with_nonempty_metric_contracts():
    specs = compile_all(REGISTRY)
    all_phases = {phase for source in specs for phase in source.phases}
    assert len(specs) == 60
    assert all_phases <= set(PREDICATES)
    assert all(PHASE_CONTRACTS[phase]["metrics"] for phase in all_phases)
    assert all(PHASE_CONTRACTS[phase]["thresholds"] for phase in all_phases)


def test_phase_result_is_immutable():
    history = grasp_history()
    result = evaluate_phase("grasp", history[0], history[1:], history[-1], spec=spec(("grasp",)))
    with pytest.raises((AttributeError, TypeError)):
        result.success = False
    with pytest.raises(TypeError):
        result.metrics["lift_distance"] = 0.0
    with pytest.raises(TypeError):
        result.phase_results[0]["phase"] = "forged"
    exported = result.to_dict()
    assert isinstance(exported["metrics"], dict)
    assert json.loads(json.dumps(exported))["success"] is True


def test_segmenter_assigns_unique_strictly_increasing_windows_to_pull_then_close():
    source = spec(("pull", "close"), selection_id="VLA82-027")
    history = (
        strict_snapshot(0, joint={"fixture/drawer_joint": 0.0}),
        strict_snapshot(1, joint={"fixture/drawer_joint": 0.0}, contact=("gripper", "drawer_handle", "obj", "drawer", "fixture", "fixture/drawer_joint")),
        strict_snapshot(2, joint={"fixture/drawer_joint": 0.35}, contact=("gripper", "drawer_handle", "obj", "drawer", "fixture", "fixture/drawer_joint")),
        strict_snapshot(5, joint={"fixture/drawer_joint": 0.0}, contact=("gripper", "drawer_handle", "obj", "drawer", "fixture", "fixture/drawer_joint")),
    )
    windows = segment_history(source, history)
    assert tuple(window.phase for window in windows) == ("pull", "close")
    assert windows[0].end_step < windows[1].end_step


def test_strict_object_target_evidence_rejects_same_class_wrong_fixture_and_auxiliary_presence():
    source = spec(("deposit",), objects=("obj", "annotated_01"))
    history = (
        strict_snapshot(0),
        strict_snapshot(2, object_z=0.06, contact=("gripper", "obj_geom", "obj", None, None, None)),
        strict_snapshot(4, object_z=0.10, contact=("obj_geom", "wrong_bin", "obj", "wrong_bin", "wrong_fixture", None), targets={"target_bin": target()}),
        strict_snapshot(6, object_z=0.10, contact=("obj_geom", "wrong_bin", "obj", "wrong_bin", "wrong_fixture", None), targets={"target_bin": target()}),
    )
    result = evaluate_operation(source, history[0], history[1:], history[-1])
    assert result.success is False
    assert "target_contact_missing:target_bin" in result.errors
    assert "object_evidence_missing:annotated_01" in result.errors


def test_strict_segmented_grasp_wipe_return_and_pull_close_pass_with_distinct_events():
    cleaning = target_named("clean_surface")
    home = target_named("home_surface")
    source = spec(("grasp", "wipe", "return"))
    trace = (
        strict_snapshot(0, gripper=0.08, dirt=1.0, targets={"clean_surface": cleaning, "home_surface": home}),
        strict_snapshot(2, object_z=0.06, dirt=1.0, contact=("gripper", "obj_geom", "obj", None, None, None), targets={"clean_surface": cleaning, "home_surface": home}),
        strict_snapshot(4, object_x=0.00, object_z=0.06, dirt=0.8, contact=("obj_geom", "clean_surface", "obj", "clean_surface", "fixture", None), targets={"clean_surface": cleaning, "home_surface": home}),
        strict_snapshot(5, object_x=0.04, object_z=0.06, dirt=0.6, contact=("obj_geom", "clean_surface", "obj", "clean_surface", "fixture", None), targets={"clean_surface": cleaning, "home_surface": home}),
        strict_snapshot(6, object_x=0.08, object_z=0.06, dirt=0.4, contact=("obj_geom", "clean_surface", "obj", "clean_surface", "fixture", None), targets={"clean_surface": cleaning, "home_surface": home}),
        strict_snapshot(7, gripper=0.01, object_z=0.08, dirt=0.4, contact=("gripper", "obj_geom", "obj", None, None, None), targets={"clean_surface": cleaning, "home_surface": home}),
        strict_snapshot(8, gripper=0.08, object_z=0.04, dirt=0.4, contact=("obj_geom", "home_surface", "obj", "home_surface", "fixture", None), targets={"clean_surface": cleaning, "home_surface": home}),
        strict_snapshot(10, gripper=0.08, object_z=0.04, dirt=0.4, contact=("obj_geom", "home_surface", "obj", "home_surface", "fixture", None), targets={"clean_surface": cleaning, "home_surface": home}),
    )
    result = evaluate_operation(source, trace[0], trace[1:], trace[-1])
    assert result.success is True
    assert [row["transition_step"] for row in result.phase_results] == [2, 6, 10]

    joint_source = spec(("pull", "close"), selection_id="VLA82-027")
    joint_trace = (
        strict_snapshot(0, joint={"fixture/drawer_joint": 0.0}, fixture_joints={"pull": "fixture/drawer_joint", "close": "fixture/drawer_joint"}),
        strict_snapshot(1, joint={"fixture/drawer_joint": 0.0}, contact=("gripper", "drawer_handle", "obj", None, "fixture", "fixture/drawer_joint"), fixture_joints={"pull": "fixture/drawer_joint", "close": "fixture/drawer_joint"}),
        strict_snapshot(2, joint={"fixture/drawer_joint": 0.35}, contact=("gripper", "drawer_handle", "obj", None, "fixture", "fixture/drawer_joint"), fixture_joints={"pull": "fixture/drawer_joint", "close": "fixture/drawer_joint"}),
        strict_snapshot(5, joint={"fixture/drawer_joint": 0.0}, contact=("gripper", "drawer_handle", "obj", None, "fixture", "fixture/drawer_joint"), fixture_joints={"pull": "fixture/drawer_joint", "close": "fixture/drawer_joint"}),
    )
    joint_result = evaluate_operation(joint_source, joint_trace[0], joint_trace[1:], joint_trace[-1])
    assert joint_result.success is True
    assert [row["transition_step"] for row in joint_result.phase_results] == [2, 5]


def test_wrong_same_class_fixture_joint_is_rejected_by_exact_identity():
    source = spec(("pull",), selection_id="VLA82-027")
    trace = (
        strict_snapshot(0, joint={"fixture/drawer_joint": 0.0}, fixture_joints={"pull": "fixture/drawer_joint"}),
        strict_snapshot(2, joint={"same_class/drawer_joint": 0.35}, contact=("gripper", "drawer_handle", "obj", None, "same_class", "same_class/drawer_joint"), fixture_joints={"pull": "fixture/drawer_joint"}),
    )
    result = evaluate_operation(source, trace[0], trace[1:], trace[-1])
    assert result.success is False
    assert "fixture_joint_mismatch:fixture/drawer_joint" in result.errors


def test_repeated_phase_consumes_distinct_release_events_and_multi_object_needs_contact_transition_stability():
    target_bin = target_named("target_bin")
    repeated = spec(("deposit", "deposit"))
    trace = (
        strict_snapshot(0, gripper=0.08, targets={"target_bin": target_bin}),
        strict_snapshot(2, object_z=0.06, contact=("gripper", "obj_geom", "obj", None, None, None), targets={"target_bin": target_bin}),
        strict_snapshot(4, gripper=0.08, object_z=0.04, contact=("obj_geom", "target_bin", "obj", "target_bin", "fixture", None), targets={"target_bin": target_bin}),
        strict_snapshot(5, gripper=0.08, object_z=0.04, contact=("obj_geom", "target_bin", "obj", "target_bin", "fixture", None), targets={"target_bin": target_bin}),
        strict_snapshot(7, gripper=0.01, object_z=0.08, contact=("gripper", "obj_geom", "obj", None, None, None), targets={"target_bin": target_bin}),
        strict_snapshot(9, gripper=0.08, object_z=0.04, contact=("obj_geom", "target_bin", "obj", "target_bin", "fixture", None), targets={"target_bin": target_bin}),
        strict_snapshot(10, gripper=0.08, object_z=0.04, contact=("obj_geom", "target_bin", "obj", "target_bin", "fixture", None), targets={"target_bin": target_bin}),
    )
    result = evaluate_operation(repeated, trace[0], trace[1:], trace[-1])
    assert result.success is True
    assert [row["transition_step"] for row in result.phase_results] == [5, 10]


def test_wipe_requires_held_tool_continuous_target_coverage_and_dirt_reduction():
    surface = target_named("clean_surface")
    source = spec(("grasp", "wipe"))
    trace = (
        strict_snapshot(0, gripper=0.08, dirt=1.0, targets={"clean_surface": surface}),
        strict_snapshot(2, gripper=0.01, object_z=0.06, dirt=1.0, contact=("gripper", "obj_geom", "obj", None, None, None), targets={"clean_surface": surface}),
        strict_snapshot(4, object_x=0.00, gripper=0.01, object_z=0.06, dirt=0.8, contact=("obj_geom", "clean_surface", "obj", "clean_surface", "fixture", None), targets={"clean_surface": surface}),
        strict_snapshot(5, object_x=0.04, gripper=0.01, object_z=0.06, dirt=0.6, contact=("obj_geom", "clean_surface", "obj", "clean_surface", "fixture", None), targets={"clean_surface": surface}),
        strict_snapshot(6, object_x=0.08, gripper=0.01, object_z=0.06, dirt=0.4, contact=("obj_geom", "clean_surface", "obj", "clean_surface", "fixture", None), targets={"clean_surface": surface}),
    )
    result = evaluate_operation(source, trace[0], trace[1:], trace[-1])
    assert result.success is True

    no_dirt = tuple(replace(item, dirt_fraction=1.0) for item in trace)
    assert "dirt_reduction_insufficient" in evaluate_operation(source, no_dirt[0], no_dirt[1:], no_dirt[-1]).errors


def test_strict_thresholds_export_real_numeric_values_and_reject_under_threshold_spray_press():
    spray = spray_history()
    low_spray = tuple(replace(item, spray_coverage=0.19 if item.step == spray[-1].step else item.spray_coverage) for item in spray)
    spray_result = evaluate_phase("spray", low_spray[0], low_spray[1:], low_spray[-1], spec=spec(("spray",)))
    assert "spray_coverage_insufficient" in spray_result.errors

    press_trace = (
        snapshot(0),
        snapshot(2, gripper=0.01, contacts=(("robot0_gripper", "soap_dispenser_button", -0.001),)),
        snapshot(4, gripper=0.01, contacts=(("robot0_gripper", "soap_dispenser_button", -0.001),), dispensed=0.009),
    )
    press_result = evaluate_phase("press", press_trace[0], press_trace[1:], press_trace[-1], spec=spec(("press",)))
    assert "dispensed_amount_insufficient" in press_result.errors

    good = evaluate_phase("spray", spray[0], spray[1:], spray[-1], spec=spec(("spray",)))
    assert isinstance(good.metrics["spray_coverage"], float)


def test_public_registry_and_combine_cannot_bypass_strict_physics_gate():
    initial = snapshot(0)
    teleported = snapshot(4, object_z=0.05, gripper=0.08, contacts=(("obj", "target_bin", -0.001),))
    source = spec(("deposit",))
    assert PREDICATES["deposit"](source, initial, (teleported,), teleported).success is False
    assert combine_ordered_phase_results(source, (), (initial, teleported)).success is False


def target() -> TargetGeometry:
    return TargetGeometry("target_bin", "fixture", np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.2]), "support", "near")


def target_named(name: str) -> TargetGeometry:
    return TargetGeometry(name, "fixture", np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.2]), "support", "near")


def strict_snapshot(
    step: int,
    *,
    object_x: float = 0.0,
    object_y: float = 0.0,
    object_z: float = 0.02,
    gripper: float = 0.01,
    dirt: float = 1.0,
    joint: dict[str, float] | None = None,
    contact: tuple[str, str, str | None, str | None, str | None, str | None] | None = None,
    targets: dict[str, TargetGeometry] | None = None,
    fixture_joints: dict[str, str] | None = None,
) -> PhysicsSnapshot:
    evidence = () if contact is None else (ContactEvidence(*contact, -0.001),)
    return PhysicsSnapshot(
        step=step,
        robot_qpos=np.zeros(7), gripper_qpos=np.array([gripper]),
        body_poses={"obj": np.array([object_x, object_y, object_z, 1.0, 0.0, 0.0, 0.0])},
        joint_positions=joint or {}, contacts=(), dirt_fraction=dirt, spray_coverage=0.0, dispensed_amount=0.0,
        contact_evidence=evidence, target_geometries=targets or {},
        fixture_joint_ids=fixture_joints or {},
    )
