from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import gymnasium as gym

from tools.vla82_full_sim.annotations import DEFAULT_REGISTRY_PATH, OperationSpec, compile_all
from tools.vla82_full_sim.assets import AssetSpec, load_asset_catalog, resolve_asset
from tools.vla82_full_sim.environment import (
    EnvironmentValidationError,
    PhysicsSnapshot,
    SceneRequest,
    _RoboCasaGymAdapter,
    _flatten_qpos_indexes,
    _mapping_for,
    build_physics_capture_contract,
    robocasa_environment_kwargs,
    build_scene_request,
    fixture_binding,
    make_environment,
    snapshot_from_environment,
    validate_scene_objects,
)


def test_counter_projection_clamps_robot_base_to_safe_object_center_bounds() -> None:
    """A reset sampler must keep the complete object inside the counter top."""
    from tools.vla82_full_sim.environment import projected_counter_object_center

    center, lower, upper = projected_counter_object_center(
        robot_base_world=(1.02, 2.90),
        counter_world=(1.0, 2.0),
        counter_yaw=np.pi / 2,
        reset_region_offset=(0.10, -0.20),
        reset_region_size=(0.80, 0.50),
        object_footprint=(0.10, 0.10),
        edge_clearance=0.02,
    )

    # The base projects to (0.90, -0.02) in local coordinates.  Only its x
    # component is outside the safe interval, so it is clamped to +0.43.
    assert np.allclose(center, (0.43, -0.02))
    assert np.allclose(lower, (-0.23, -0.38))
    assert np.allclose(upper, (0.43, -0.02))


def test_native_asset_footprint_uses_model_bbox_instead_of_stale_catalog_dimensions() -> None:
    from tools.vla82_full_sim.environment import native_asset_reset_footprint

    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-002")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    footprint = native_asset_reset_footprint(asset)

    # The source catalog declared a 5.5 x 3.5 cm nominal sponge.  The native
    # collision model that MuJoCo actually samples is 6.74 x 8.98 cm.
    assert np.allclose(footprint, (0.0674404596, 0.0898226838), atol=1e-8)


def test_environment_applies_and_records_selected_object_material_friction() -> None:
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-006")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    request = build_scene_request(spec, asset)

    environment = make_environment(request, seed=2000)
    try:
        evidence = environment.friction_evidence
        assert evidence["version"] == "vla82-material-friction-v1"
        assert evidence["material"] == "wood"
        assert evidence["geom_names"]
        raw = environment.env
        for name in evidence["geom_names"]:
            geom_id = raw.sim.model.geom_name2id(name)
            assert np.allclose(raw.sim.model.geom_friction[geom_id], (0.82, 0.008, 0.0003))
    finally:
        environment.close()


def test_vla045_audit_camera_uses_wide_operation_focus() -> None:
    from tools.vla82_full_sim.environment import audit_camera_focus, audit_camera_offsets

    focus = audit_camera_focus(
        "VLA82-045",
        target=(1.86, -0.44, 0.94),
        eef=(1.90, -0.77, 1.31),
        destination=(1.925, -0.76, 0.76),
    )

    assert np.allclose(focus, (1.8925, -0.60, 0.85))
    assert audit_camera_offsets("VLA82PickPlace", "VLA82-045")[0][2] >= .85


def test_seeded_rng_scope_repeats_python_and_numpy_draws_without_leaking_global_state() -> None:
    from tools.vla82_full_sim.environment import seeded_rng_scope
    import random

    with seeded_rng_scope(820007):
        first = (random.random(), float(np.random.random()))
    with seeded_rng_scope(820007):
        second = (random.random(), float(np.random.random()))
    assert first == second


def test_live_scene_fingerprint_is_identical_for_two_fresh_same_seed_007_scenes() -> None:
    from tools.vla82_full_sim.environment import scene_runtime_fingerprint
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-007")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    request = build_scene_request(spec, asset)
    first = make_environment(request, seed=820007)
    second = make_environment(request, seed=820007)
    try:
        assert scene_runtime_fingerprint(first, request)["sha256"] == scene_runtime_fingerprint(second, request)["sha256"]
    finally:
        first.close()
        second.close()


def test_trash_can_scene_fingerprint_is_identical_for_two_fresh_same_seed_001_scenes() -> None:
    """The custom ObjectPlayEnv sampler must use the supplied episode seed."""
    from tools.vla82_full_sim.environment import scene_runtime_fingerprint
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-001")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    request = build_scene_request(spec, asset)
    first = make_environment(request, seed=820001)
    second = make_environment(request, seed=820001)
    try:
        first_fingerprint = scene_runtime_fingerprint(first, request)
        assert first_fingerprint["sha256"] == scene_runtime_fingerprint(second, request)["sha256"]
        # The static target pose is part of provenance, not merely the
        # movable source-can body positions.
        assert "trash_can:inner_cavity" in first_fingerprint["target_geometries"]
    finally:
        first.close()
        second.close()


def test_trash_can_target_is_a_static_scene_receptacle_not_a_free_object() -> None:
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-001")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    environment = make_environment(build_scene_request(spec, asset), seed=820001)
    try:
        target = environment.unwrapped.env.objects["trash_can"]
        assert tuple(target.joints) == ()
    finally:
        environment.close()


def test_trash_can_reset_samples_all_cans_outside_and_clear_of_the_static_receptacle() -> None:
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-001")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    environment = make_environment(build_scene_request(spec, asset), seed=2004)
    try:
        raw = environment.unwrapped.env
        contract = build_physics_capture_contract(environment)
        target = next(iter(contract.target_geometries.values()))
        lower, upper = np.asarray(target.min_corner), np.asarray(target.max_corner)
        # The sampler construction keeps every source body over 5cm beyond
        # the target cavity's +X edge.  This is stronger than merely testing
        # that its centre is not inside and excludes reset-time overlap.
        for name in ("obj", "annotated_01", "annotated_02"):
            obj = raw.objects[name]
            position = raw.sim.data.body_xpos[raw.sim.model.body_name2id(obj.root_body)]
            assert not bool(np.all(position[:2] >= lower[:2]) and np.all(position[:2] <= upper[:2]))
            assert np.linalg.norm(position[:2] - (lower[:2] + upper[:2]) / 2.0) >= .025
        source_geometries = {
            geometry for object_id in ("obj", "annotated_01", "annotated_02")
            for geometry in contract.object_geom_names[object_id]
        }
        target_geometries = set(contract.target_geometries)
        for index in range(raw.sim.data.ncon):
            contact = raw.sim.data.contact[index]
            names = {
                raw.sim.model.geom_id2name(contact.geom1), raw.sim.model.geom_id2name(contact.geom2),
            }
            assert not (names & source_geometries and names & target_geometries)
    finally:
        environment.close()


def test_vla82_trash_can_source_lanes_are_reachable_separated_outside_target_and_exact() -> None:
    from tools.vla82_full_sim.environment import vla82_trash_can_source_centers, vla82_trash_can_target_center

    centers = np.asarray(vla82_trash_can_source_centers(), dtype=float)
    assert centers.shape == (3, 2)
    # The exact source can's live horizontal radius is just under 4.75cm;
    # retain a 5mm clearance between complete collision cylinders.
    radius = .0475
    for left in range(3):
        for right in range(left + 1, 3):
            assert np.linalg.norm(centers[left] - centers[right]) > 2 * radius + .005
    cavity_center = np.asarray(vla82_trash_can_target_center(), dtype=float)
    assert all(np.linalg.norm(center - cavity_center) > .025 for center in centers)
    assert all(np.linalg.norm(center - cavity_center) > .025 for center in centers)
    fresh_eef_xy = np.array((-.0086697, .0817438))
    assert np.linalg.norm(centers[0] - fresh_eef_xy) < .015
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-001")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    request = build_scene_request(spec, asset)
    source_assets = request.object_assets[1:]
    assert len(source_assets) == 3
    assert all(item.kind == "custom_same_class" and "VLA82-001" in item.asset_path_or_group and "aux-0" in item.asset_path_or_group for item in source_assets)


def test_vla82_trash_can_target_center_keeps_exact_receptacle_supported_and_sources_external() -> None:
    from tools.vla82_full_sim.environment import (
        vla82_trash_can_cavity_slot_centers, vla82_trash_can_source_centers,
        vla82_trash_can_target_center,
    )

    target = np.asarray(vla82_trash_can_target_center(), dtype=float)
    assert np.allclose(target, (.27, .42))
    offsets = np.asarray(vla82_trash_can_cavity_slot_centers(relative=True), dtype=float)
    slots = np.asarray(vla82_trash_can_cavity_slot_centers(), dtype=float)
    assert np.allclose(offsets, ((-.05, -.045), (.05, -.045), (0., .045)))
    assert np.allclose(slots, offsets + target)
    # Conservative full-size-bin outer half-width and the table's documented
    # reachable support rectangle for the widened ObjectPlay table.
    outer_half_width = .13
    table_lower, table_upper = np.array((-.40, -.60)), np.array((.40, .60))
    assert np.all(target - outer_half_width >= table_lower)
    assert np.all(target + outer_half_width <= table_upper)
    cavity_half_width = .115
    radius = .0475
    assert all(np.linalg.norm(left - right) > 2 * radius for index, left in enumerate(offsets) for right in offsets[index + 1:])
    assert all(np.all(np.abs(slot) + radius < cavity_half_width) for slot in offsets)
    for source in np.asarray(vla82_trash_can_source_centers(), dtype=float):
        assert bool(np.any(np.abs(source - target) > cavity_half_width + radius + .025))


def test_vla001_runtime_three_slots_fit_complete_live_can_cylinders_and_target_is_table_supported() -> None:
    """Capacity proof uses MuJoCo's live collision sizes, not nominal labels."""
    from tools.vla82_full_sim.environment import vla82_trash_can_cavity_slot_centers

    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-001")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    environment = make_environment(build_scene_request(spec, asset), seed=2006)
    try:
        raw = environment.unwrapped.env
        model = raw.sim.model
        contract = build_physics_capture_contract(environment)
        target = next(iter(contract.target_geometries.values()))
        lower, upper = np.asarray(target.min_corner, dtype=float), np.asarray(target.max_corner, dtype=float)
        # The fixed target is the original source-textured full-size receptacle,
        # not the earlier compact construction variant.
        assert environment.request.object_assets[0].asset_path_or_group.endswith("VLA82-001\\model.xml")
        assert "target-compact" not in environment.request.object_assets[0].asset_path_or_group
        assert np.allclose(environment.request.object_assets[0].dimensions, (.115, .115, .15))
        assert np.allclose((lower[:2] + upper[:2]) / 2.0, (.27, .42), atol=.003)
        assert np.allclose((upper - lower)[:2] / 2.0, (.107, .107), atol=.004)
        assert np.allclose((lower[2], upper[2]), (.81, 1.11), atol=.003)
        radii = []
        for name in ("obj", "annotated_01", "annotated_02"):
            geom_ids = [model.geom_name2id(geom) for geom in contract.object_geom_names[name]]
            radii.append(max(float(model.geom_size[geom_id][0]) for geom_id in geom_ids))
        radius = max(radii)
        slots = np.asarray(vla82_trash_can_cavity_slot_centers(), dtype=float)
        assert all(np.all(slot - radius >= lower[:2]) and np.all(slot + radius <= upper[:2]) for slot in slots)
        assert all(np.linalg.norm(left - right) >= 2 * radius for index, left in enumerate(slots) for right in slots[index + 1:])
        # Every collision-wall footprint remains inside the widened table's
        # physical support rectangle; Y enlargement must not move the base-X.
        table_half = np.asarray(raw.table_full_size[:2], dtype=float) / 2.0
        assert np.allclose(raw.table_full_size[:2], (.8, 1.2))
        for geom in contract.target_geom_names[target.target_id]:
            geom_id = model.geom_name2id(geom)
            half = np.asarray(model.geom_size[geom_id][:2], dtype=float)
            center = np.asarray(raw.sim.data.geom_xpos[geom_id][:2], dtype=float)
            assert np.all(center - half >= -table_half - 1e-6)
            assert np.all(center + half <= table_half + 1e-6)
        for name in ("obj", "annotated_01", "annotated_02"):
            body = model.body_name2id(raw.objects[name].root_body)
            source = np.asarray(raw.sim.data.body_xpos[body][:2], dtype=float)
            nearest = np.minimum(np.maximum(source, lower[:2]), upper[:2])
            assert np.linalg.norm(source - nearest) - radius > .025
    finally:
        environment.close()


def test_vla002_cleaning_contract_uses_live_counter_for_wipe_and_return_not_generic_cabinet() -> None:
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-002")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    request = build_scene_request(spec, asset)
    environment = make_environment(request, seed=820002)
    try:
        raw = environment.unwrapped.env
        assert Path(raw.objects["obj"].mjcf_path).resolve() == Path(request.primary_asset.asset_path_or_group).resolve()
        assert environment.runtime_asset_identity["matches"] is True
        assert environment.runtime_asset_identity["selection_id"] == "VLA82-002"
        contract = build_physics_capture_contract(environment, request)
        assert contract.target_geometries
        assert all("counter" in target.fixture_id for target in contract.target_geometries.values())
        assert all("counter" in target_id for target_id in contract.target_geometries)
        assert all(target.spatial_relation == "clean-region" for target in contract.target_geometries.values())
        snap = snapshot_from_environment(environment, 0, contract=contract)
        target = next(iter(contract.target_geometries.values()))
        point = snap.body_poses["obj"][:3]
        assert point[2] <= np.asarray(target.max_corner)[2]
    finally:
        environment.close()


def fixture_operation_spec() -> OperationSpec:
    return OperationSpec(
        selection_id="VLA82-016",
        task="衣物整理",
        object_name="书籍",
        operation_text="摆放",
        source_kind="operation_json",
        source_path="C:/source/operation.json",
        phases=("place",),
        manipulated_objects=("书籍", "凉鞋", "凉鞋"),
        predicate_names=("place_completed",),
        source_sha256="b" * 64,
    )


def fixture_asset() -> AssetSpec:
    return AssetSpec(
        selection_id="VLA82-016",
        semantic_class="书籍",
        kind="custom_same_class",
        asset_path_or_group="assets/custom/book.xml",
        exact_class=True,
        collision_validated=True,
        visible_validated=True,
        affordances=("grasp", "place"),
        dimensions=(0.110, 0.075, 0.016),
        geometry="book",
    )


def test_trash_deposit_runtime_identity_validates_receptacle_not_first_manipulated_can() -> None:
    from tools.vla82_full_sim.environment import runtime_primary_asset_identity

    asset = fixture_asset()
    request = SceneRequest(
        selection_id="VLA82-001", task_class="VLA82TrashCanDeposit", robot_name="PandaOmron",
        camera_names=(), primary_asset=asset, manipulated_objects=("can",), object_assets=(asset,),
        fixture_requirements=(), source_fixture="cans", target_fixture="trash_can", target_relation="inside",
    )
    raw = SimpleNamespace(objects={
        "trash_can": SimpleNamespace(mjcf_path=asset.asset_path_or_group),
        "obj": SimpleNamespace(mjcf_path="assets/custom/can.xml"),
    })

    identity = runtime_primary_asset_identity(raw, request)
    assert identity["matches"] is True
    assert Path(identity["loaded_mjcf_path"]).as_posix().endswith("assets/custom/book.xml")


def test_scene_request_contains_complete_robot_and_all_annotated_objects() -> None:
    request = build_scene_request(fixture_operation_spec(), fixture_asset())

    assert request.robot_name == "PandaOmron"
    assert request.camera_names == ("robot0_agentview_left", "robot0_eye_in_hand")
    assert request.manipulated_objects == ("书籍", "凉鞋", "凉鞋")
    assert request.forbidden_direct_state_writes is True


def test_native_furniture_operation_uses_exact_fixture_part_contract() -> None:
    """A stove knob is operated on the native fixture, never injected as obj."""
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-007")
    request = build_scene_request(spec, resolve_asset(spec, _mapping_for(spec.selection_id)))

    assert request.primary_asset.kind == "fixture_part"


def test_task2_request_preserves_cabinet_to_counter_target_semantics() -> None:
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-004")
    request = build_scene_request(spec, resolve_asset(spec, _mapping_for(spec.selection_id)))

    assert (request.source_fixture, request.target_fixture, request.target_relation) == ("cabinet", "counter", "on")


def test_counter_to_drawer_profile_is_selected_by_authoritative_task_semantics() -> None:
    from tools.vla82_full_sim.environment import is_counter_to_drawer_request

    specs = {item.selection_id: item for item in compile_all(DEFAULT_REGISTRY_PATH)}
    drawer = build_scene_request(specs["VLA82-006"], resolve_asset(specs["VLA82-006"], _mapping_for("VLA82-006")))
    cabinet = build_scene_request(specs["VLA82-004"], resolve_asset(specs["VLA82-004"], _mapping_for("VLA82-004")))
    assert is_counter_to_drawer_request(drawer)
    assert not is_counter_to_drawer_request(cabinet)


def test_vla004_spawn_anchor_aligns_with_source_object_without_crossing_cabinet_front() -> None:
    from tools.vla82_full_sim.environment import open_drawer_robot_anchor, source_cabinet_robot_anchor

    anchor = source_cabinet_robot_anchor(
        cabinet_center=(.20, -4.186, 1.84),
        native_anchor=(.80, -4.186, 0.0),
        object_center=(.226, -4.298, 1.495),
        normal_standoff=.56,
    )

    # Retain the native outward-facing side and ground height, while aligning
    # tangentially with the object.  This is a reset pose, not an episode write.
    assert np.allclose(anchor, (.76, -4.298, 0.0))
    assert np.allclose(
        open_drawer_robot_anchor(
            drawer_low=(1.7475, -.856945, .665), drawer_high=(2.1025, 0., .736),
            native_anchor=(2.2384, -.7931, 0.), standoff=.15,
        ),
        (1.95, -1.006945, 0.),
    )


def test_ordinary_contract_refuses_an_undeclared_or_missing_counter_instead_of_falling_back_to_wall() -> None:
    """No arbitrary fixture / first-visible-geom target is accepted for VLA82-004."""
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-004")
    request = build_scene_request(spec, resolve_asset(spec, _mapping_for(spec.selection_id)))

    class Model:
        ngeom = 2
        geom_rgba = np.ones((2, 4))
        geom_size = np.full((2, 3), 0.02)
        geom_bodyid = np.array([0, 1])

        @staticmethod
        def geom_id2name(index):
            return ("obj_geom", "wall_visible")[index]

        @staticmethod
        def geom_name2id(name):
            return {"obj_geom": 0, "wall_visible": 1}[name]

    raw = type("Raw", (), {})()
    raw.objects = {"obj": type("Obj", (), {"naming_prefix": "obj_"})()}
    raw.fixtures = {"wall": type("Wall", (), {"naming_prefix": "wall_"})()}
    raw.sim = type("Sim", (), {"model": Model(), "data": type("Data", (), {"geom_xpos": np.zeros((2, 3))})()})()
    environment = type("Environment", (), {"env": raw, "request": request})()

    with pytest.raises(EnvironmentValidationError, match="declared target fixture counter is absent"):
        build_physics_capture_contract(environment)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("selection_id", "joint_phase"),
    (("VLA82-004", None), ("VLA82-007", "turn_knob")),
)
def test_real_task2_scene_yields_only_live_capture_bindings(selection_id: str, joint_phase: str | None) -> None:
    """Run one ordinary and one appliance scene against their live MuJoCo model."""
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == selection_id)
    request = build_scene_request(spec, resolve_asset(spec, _mapping_for(spec.selection_id)))
    environment = make_environment(request, seed=17)
    try:
        contract = build_physics_capture_contract(environment)
        snapshot = snapshot_from_environment(environment, 0, contract=contract)
        assert contract.object_geom_names.get("obj")
        assert contract.target_geometries
        assert snapshot.contact_evidence
        if joint_phase is None:
            target_id, target = next(iter(contract.target_geometries.items()))
            assert request.target_fixture == "counter"
            assert target.fixture_id == "counter_3_right_group_1"
            assert target.spatial_relation == "on"
            assert all("counter_" in geom and "_top_" in geom for geom in contract.target_geom_names[target_id])
            assert all("wall" not in geom.lower() for geom in contract.target_geom_names[target_id])
            # The object starts in the declared cabinet, so target identity is
            # established without pretending an initial counter contact.
            assert not any(item.object_id == "obj" and item.target_id == target_id for item in snapshot.contact_evidence)
        else:
            raw = environment.unwrapped.env
            joint_id = contract.fixture_joint_ids[joint_phase]
            assert contract.fixture_contact_joints
            assert raw.knob in joint_id
            assert "obj" in snapshot.body_poses
            assert not any(item.joint_id == joint_id for item in snapshot.contact_evidence)

            # Contract-unit evidence: pass a synthetic contact record through
            # the production capture layer, but use the live task's exact
            # MuJoCo model names and selected joint.  This is explicitly not a
            # rollout-success claim; Task 4 must provide the env.step contact.
            gripper_geom = next(
                raw.sim.model.geom_id2name(index) or ""
                for index in range(int(raw.sim.model.ngeom))
                if "finger1_pad_collision" in (raw.sim.model.geom_id2name(index) or "")
            )
            target_geom = next(iter(contract.target_geom_names.values()))[0]
            wrong_geom = next(
                raw.sim.model.geom_id2name(index) or ""
                for index in range(int(raw.sim.model.ngeom))
                if "wall_" in (raw.sim.model.geom_id2name(index) or "")
            )

            def capture_with_pair(first: str, second: str):
                model = raw.sim.model
                data = SimpleNamespace(
                    qpos=raw.sim.data.qpos,
                    body_xpos=raw.sim.data.body_xpos,
                    body_xquat=raw.sim.data.body_xquat,
                    geom_xpos=raw.sim.data.geom_xpos,
                    ncon=1,
                    contact=(SimpleNamespace(geom1=model.geom_name2id(first), geom2=model.geom_name2id(second), dist=-0.001),),
                )
                capture_raw = SimpleNamespace(robots=raw.robots, objects={}, sim=SimpleNamespace(model=model, data=data))
                capture_env = SimpleNamespace(env=capture_raw, request=request)
                return snapshot_from_environment(capture_env, 1, contract=contract)

            mapped = capture_with_pair(gripper_geom, target_geom).contact_evidence[0]
            rejected = capture_with_pair(gripper_geom, wrong_geom).contact_evidence[0]
            assert (mapped.object_id, mapped.target_id, mapped.joint_id) == ("obj", next(iter(contract.target_geometries)), joint_id)
            assert (rejected.object_id, rejected.target_id, rejected.joint_id) == (None, None, None)
    finally:
        environment.close()


@pytest.mark.slow
def test_vla005_inside_target_is_the_live_drawer_inner_cavity_not_outer_bottom() -> None:
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-005")
    asset = resolve_asset(spec, {**_mapping_for(spec.selection_id), "asset": load_asset_catalog()[spec.selection_id]})
    environment = make_environment(build_scene_request(spec, asset), seed=2000)
    try:
        contract = build_physics_capture_contract(environment)
        target_id, target = next(iter(contract.target_geometries.items()))
        names = contract.target_geom_names[target_id]
        assert target.spatial_relation == "inside"
        assert names
        assert all("inner_" in name and "visual" not in name for name in names)
        assert target.max_corner[2] >= .817
        assert target.max_corner[1] <= -.314
    finally:
        environment.close()


def test_trash_can_deposit_uses_source_matched_receptacle_scene() -> None:
    spec = OperationSpec(
        selection_id="VLA82-001",
        task="室内卫生清洁",
        object_name="垃圾桶",
        operation_text="将垃圾投放至垃圾桶",
        source_kind="operation_json",
        source_path="C:/source/operation.json",
        phases=("deposit",),
        manipulated_objects=("垃圾桶",),
        predicate_names=("deposit_completed",),
        source_sha256="c" * 64,
    )
    asset = AssetSpec(
        selection_id="VLA82-001",
        semantic_class="垃圾桶",
        kind="custom_same_class",
        asset_path_or_group=str(Path(__file__).resolve().parents[1] / "assets" / "vla82_source_textured" / "VLA82-001" / "model.xml"),
        exact_class=True,
        collision_validated=True,
        visible_validated=True,
        affordances=("deposit",),
        dimensions=(0.115, 0.115, 0.150),
        geometry="open_receptacle",
    )

    request = build_scene_request(spec, asset)

    assert request.task_class == "VLA82TrashCanDeposit"
    assert request.object_assets[0].asset_path_or_group.endswith("VLA82-001\\model.xml")
    assert "trash_receptacle" in request.fixture_requirements
    assert (request.source_fixture, request.target_fixture, request.target_relation) == ("cans", "trash_can", "inside")


@pytest.mark.slow
def test_real_vla82_trash_can_contract_binds_three_cans_to_open_inner_cavity() -> None:
    """VLA82-001 starts unsolved: three cans, one exact open-cavity target."""
    spec = next(item for item in compile_all(DEFAULT_REGISTRY_PATH) if item.selection_id == "VLA82-001")
    request = build_scene_request(spec, resolve_asset(spec, _mapping_for(spec.selection_id)))
    environment = make_environment(request, seed=17)
    try:
        contract = build_physics_capture_contract(environment)
        snapshot = snapshot_from_environment(environment, 0, contract=contract)
    finally:
        environment.close()

    target_id = "trash_can:inner_cavity"
    assert tuple(contract.object_geom_names) == ("obj", "annotated_01", "annotated_02")
    assert set(contract.target_geometries) == {target_id}
    target = contract.target_geometries[target_id]
    assert (target.fixture_id, target.spatial_relation, target.support_id) == ("trash_can", "inside", target_id)
    assert len(contract.target_geom_names[target_id]) == 5
    assert all("MCJFObj_0_collision_" in name for name in contract.target_geom_names[target_id])
    assert np.all(np.asarray(target.min_corner) < np.asarray(target.max_corner))
    inside = [
        bool(np.all(np.asarray(snapshot.body_poses[object_id])[:3] >= np.asarray(target.min_corner))
             and np.all(np.asarray(snapshot.body_poses[object_id])[:3] <= np.asarray(target.max_corner)))
        for object_id in contract.object_geom_names
    ]
    assert not all(inside)
    assert not all(
        any(contact.object_id == object_id and contact.target_id == target_id for contact in snapshot.contact_evidence)
        for object_id in contract.object_geom_names
    )


def test_scene_request_expands_all_objects_from_authoritative_instruction() -> None:
    # VLA82-016 source instruction explicitly requires one book and two sandals.
    from tools.vla82_full_sim.annotations import DEFAULT_REGISTRY_PATH, compile_all

    spec = compile_all(DEFAULT_REGISTRY_PATH)[15]
    asset = AssetSpec(
        selection_id=spec.selection_id,
        semantic_class=spec.object_name,
        kind="custom_same_class",
        asset_path_or_group="assets/custom/book.xml",
        exact_class=False,
        collision_validated=False,
        visible_validated=False,
        affordances=("place",),
        source_path=spec.source_path,
        dimensions=(0.110, 0.075, 0.016),
        geometry="book",
    )

    request = build_scene_request(spec, asset)

    assert request.manipulated_objects == ("书籍", "凉鞋", "凉鞋")
    assert len(request.object_assets) == 3
    assert len({item.asset_key for item in request.object_assets}) == 3


class _Robot:
    robot_joints = tuple(f"joint{i}" for i in range(7))


class _RawEnvironment:
    robots = [_Robot()]
    camera_names = ("robot0_agentview_left", "robot0_eye_in_hand")


class _Unwrapped:
    env = _RawEnvironment()


class _FakeEnvironment:
    unwrapped = _Unwrapped()


def test_make_environment_rejects_incomplete_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    request = build_scene_request(fixture_operation_spec(), fixture_asset())
    _RawEnvironment.robots = [type("ShortRobot", (), {"robot_joints": ("j0",) * 6})()]
    monkeypatch.setattr(
        "tools.vla82_full_sim.environment._build_registered_or_custom_robocasa_task",
        lambda request, seed: _FakeEnvironment(),
    )

    with pytest.raises(EnvironmentValidationError, match="complete PandaOmron"):
        make_environment(request, 17)


def test_scene_validation_requires_every_authoritatively_annotated_object() -> None:
    request = build_scene_request(fixture_operation_spec(), fixture_asset())
    raw = type("Raw", (), {"objects": {"obj": object(), "annotated_01": object()}})()

    with pytest.raises(EnvironmentValidationError, match="annotated_02"):
        validate_scene_objects(raw, request)


def test_snapshot_exposes_physics_surface() -> None:
    snapshot = PhysicsSnapshot(
        step=2,
        robot_qpos=np.zeros(7),
        gripper_qpos=np.zeros(2),
        body_poses={"book": np.zeros(7)},
        joint_positions={"drawer": 0.2},
        contacts=(("finger", "book", 1.0),),
        dirt_fraction=0.5,
        spray_coverage=0.1,
        dispensed_amount=0.0,
    )

    assert snapshot.robot_qpos.shape == (7,)
    assert snapshot.contacts[0][1] == "book"


def test_snapshot_index_normalizer_accepts_pandaomron_gripper_group_mapping() -> None:
    assert _flatten_qpos_indexes({"right": [np.int32(11), np.int32(12)]}).tolist() == [11, 12]


def test_adapter_is_a_gymnasium_environment() -> None:
    raw = type(
        "Raw",
        (),
        {
            "action_spec": (np.full(12, -1.0), np.full(12, 1.0)),
        },
    )()
    request = build_scene_request(fixture_operation_spec(), fixture_asset())

    assert isinstance(_RoboCasaGymAdapter(raw, request), gym.Env)


def test_fixture_part_scene_does_not_pass_object_group_to_fixture_task() -> None:
    fixture_asset = AssetSpec(
        selection_id="VLA82-007",
        semantic_class="炉灶",
        kind="fixture_part",
        asset_path_or_group="TurnOnStove",
        exact_class=True,
        collision_validated=True,
        visible_validated=True,
        affordances=("turn_knob",),
    )
    request = SceneRequest(
        selection_id="VLA82-007",
        task_class="TurnOnStove",
        robot_name="PandaOmron",
        camera_names=("robot0_agentview_left", "robot0_eye_in_hand"),
        primary_asset=fixture_asset,
        manipulated_objects=("炉灶",),
        object_assets=(fixture_asset,),
        fixture_requirements=("knob", "work_surface"),
    )

    assert "obj_groups" not in robocasa_environment_kwargs(request, 820007)


def test_fixture_binding_rejects_an_unrelated_fixture() -> None:
    fixture_asset = AssetSpec(
        selection_id="VLA82-007", semantic_class="炉灶", kind="fixture_part",
        asset_path_or_group="TurnOnStove", exact_class=False, collision_validated=False,
        visible_validated=False, affordances=("turn_knob",), dimensions=(.12, .12, .05), geometry="fixture",
    )
    request = SceneRequest("VLA82-007", "TurnOnStove", "PandaOmron", ("robot0_agentview_left", "robot0_eye_in_hand"),
                           fixture_asset, ("炉灶",), (fixture_asset,), ("knob", "work_surface"))
    raw = type("Raw", (), {"fixtures": {"unrelated_oven": type("Oven", (), {"naming_prefix": "oven_"})()}})()

    with pytest.raises(EnvironmentValidationError, match="fixture target"):
        fixture_binding(raw, request)
