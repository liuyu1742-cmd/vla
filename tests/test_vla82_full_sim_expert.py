"""Contracts for source-backed, step-only expert demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.vla82_full_sim.annotations import OperationSpec


def _spec(selection_id: str = "VLA82-002") -> OperationSpec:
    return OperationSpec(
        selection_id=selection_id,
        task="test", object_name="object", operation_text="source",
        source_kind="test", source_path="test", phases=("grasp", "place"),
        manipulated_objects=("object",), predicate_names=("grasp_completed", "place_completed"),
        source_sha256="source-hash",
    )


class _StepOnlyEnv:
    """Tiny real-interface stand-in: state changes only when step is invoked."""

    def __init__(self):
        self.request = type("R", (), {"selection_id": "VLA82-002", "camera_names": ("agentview", "robot0_robotview")})()
        self.action_space = type("A", (), {"shape": (12,), "low": np.full(12, -1.0), "high": np.full(12, 1.0)})()
        controller = type("C", (), {"_action_split_indexes": {"right": (0, 6), "torso": (6, 7), "base": (7, 10), "right_gripper": (10, 11)}})()
        robot = type("Robot", (), {"composite_controller": controller})()
        self.unwrapped = type("U", (), {"env": type("Raw", (), {"robots": (robot,)})()})()
        self.calls = 0
        self.closed = False
        self.friction_evidence = {
            "version": "vla82-material-friction-v1",
            "material": "rubber_sponge",
            "material_source": "selection_id",
            "geom_names": ["obj_collision"],
            "before": {"obj_collision": [0.95, 0.3, 0.1]},
            "after": {"obj_collision": [1.0, 0.012, 0.0005]},
            "application_stage": "post_load_pre_step",
        }

    def reset(self, *, seed=None):
        return self._obs(), {"seed": seed}

    def _obs(self):
        return {"video.agentview": np.full((8, 8, 3), self.calls, dtype=np.uint8), "video.robot0_robotview": np.zeros((8, 8, 3), dtype=np.uint8)}

    def step(self, action):
        assert np.asarray(action).shape == (12,)
        self.calls += 1
        return self._obs(), 0.0, False, False, {}

    def close(self):
        self.closed = True


def test_episode_npz_is_non_pickle_aligned_and_never_acceptance(monkeypatch, tmp_path: Path):
    from tools.vla82_full_sim import expert
    env = _StepOnlyEnv()
    monkeypatch.setattr(expert, "make_environment", lambda scene, seed: env)
    monkeypatch.setattr(expert, "scene_runtime_fingerprint", lambda env, scene: {"sha256": expert.scene_fingerprint(scene)})
    monkeypatch.setattr(expert, "build_physics_capture_contract", lambda env, scene: object())
    monkeypatch.setattr(expert, "snapshot_from_environment", lambda env, step, contract=None: {"step": step})
    monkeypatch.setattr(expert, "evaluate_operation", lambda spec, initial, history, final: type("P", (), {"success": True, "to_dict": lambda self: {"success": True}})())
    report = expert.collect_expert_episode(_spec(), object(), 2001, tmp_path)
    assert report.status == "PASS"
    assert report.training_expert is True
    assert env.calls > 0
    with np.load(report.npz_path, allow_pickle=False) as data:
        assert set(("primary", "wrist", "proprio", "actions", "phases", "training_expert", "seed", "spec_sha256", "scene_sha256", "asset_sha256", "predicate_json")).issubset(data.files)
        assert data["primary"].shape[0] == data["actions"].shape[0] == data["phases"].shape[0]
        assert data["proprio"].shape[1] == 8
        assert bool(data["training_expert"].item()) is True
    diagnostic = json.loads(Path(report.diagnostic_path).read_text(encoding="utf-8"))
    assert diagnostic["friction_evidence"] == env.friction_evidence


def test_resume_rejects_wrong_provenance_even_when_npz_exists(monkeypatch, tmp_path: Path):
    from tools.vla82_full_sim import expert
    env = _StepOnlyEnv()
    monkeypatch.setattr(expert, "make_environment", lambda scene, seed: env)
    monkeypatch.setattr(expert, "scene_runtime_fingerprint", lambda env, scene: {"sha256": expert.scene_fingerprint(scene)})
    monkeypatch.setattr(expert, "build_physics_capture_contract", lambda env, scene: object())
    monkeypatch.setattr(expert, "snapshot_from_environment", lambda env, step, contract=None: {"step": step})
    monkeypatch.setattr(expert, "evaluate_operation", lambda *args: type("P", (), {"success": True, "to_dict": lambda self: {"success": True}})())
    first = expert.collect_expert_episode(_spec(), object(), 2002, tmp_path)
    assert expert.validated_resume_episode(Path(first.npz_path), _spec(), object(), 2002)
    changed = _spec(); object.__setattr__(changed, "source_sha256", "changed")
    assert not expert.validated_resume_episode(Path(first.npz_path), changed, object(), 2002)


def test_direct_write_sentinel_marks_episode_failed(monkeypatch, tmp_path: Path):
    from tools.vla82_full_sim import expert
    env = _StepOnlyEnv()
    monkeypatch.setattr(expert, "make_environment", lambda scene, seed: env)
    monkeypatch.setattr(expert, "scene_runtime_fingerprint", lambda env, scene: {"sha256": expert.scene_fingerprint(scene)})
    monkeypatch.setattr(expert, "build_physics_capture_contract", lambda env, scene: object())
    monkeypatch.setattr(expert, "snapshot_from_environment", lambda env, step, contract=None: {"step": step})
    monkeypatch.setattr(expert, "evaluate_operation", lambda *args: type("P", (), {"success": True, "to_dict": lambda self: {"success": True}})())
    monkeypatch.setattr(expert, "_direct_write_detected", lambda *args: True)
    report = expert.collect_expert_episode(_spec(), object(), 2003, tmp_path)
    assert report.status == "FAIL"
    assert "direct_state_write_detected" in report.errors


def test_knob_gate_requires_step_contact_exact_joint_and_nonnoise_motion():
    from tools.vla82_full_sim.expert import validate_knob_gate
    assert validate_knob_gate(
        "stove_knob_joint", "stove_knob_joint", 0.0, 0.4,
        ({"joint_id": "stove_knob_joint", "gripper_contact": True, "after_step": True},),
    ) == ()
    assert "knob_contact_missing" in validate_knob_gate("stove_knob_joint", "stove_knob_joint", 0.0, 0.4, ())
    assert "knob_joint_delta_noise" in validate_knob_gate("stove_knob_joint", "stove_knob_joint", 0.0, 0.001, ({"joint_id": "stove_knob_joint", "gripper_contact": True, "after_step": True},))


def test_collect_demos_cli_is_bounded_and_resumable(monkeypatch, tmp_path: Path):
    import tools.run_vla82_full_simulation as cli
    from tools.vla82_full_sim.expert import EpisodeReport
    calls: list[int] = []
    monkeypatch.setattr(cli, "DEMO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_load_compiled_specs", lambda: [_spec()])
    monkeypatch.setattr(cli, "_scene_for_spec", lambda spec: object())
    def collect(spec, scene, seed, output):
        calls.append(seed)
        return EpisodeReport(spec.selection_id, seed, "PASS", True, str(tmp_path / f"{seed}.npz"), str(tmp_path / f"{seed}.json"), True, (), "a", "b", "c", 3)
    monkeypatch.setattr(cli, "collect_expert_episode", collect)
    args = cli.build_parser().parse_args(["collect-demos", "--episodes-per-object", "2", "--selection-id", "VLA82-002"])
    assert args.handler(args) == 0
    assert calls == [2000, 2001]


def test_openvla_action_is_expanded_to_pandaomron_public_action_vector():
    from tools.vla82_full_sim.expert import ActionLayout, expand_openvla_action
    layout = ActionLayout(arm=(0, 6), torso=6, base=(7, 10), gripper=10, base_mode=11)
    action = expand_openvla_action(np.array([.1, .2, .3, .4, .5, .6, .7], dtype=np.float32), 12, layout=layout)
    assert action.shape == (12,)
    np.testing.assert_allclose(action[:6], [.1, .2, .3, .4, .5, .6])
    assert action[6] == 0.0  # torso, never the gripper
    np.testing.assert_allclose(action[7:10], 0.0)
    assert action[10] == .7  # PandaOmron right-gripper controller
    assert action[11] == -1.0


@pytest.mark.parametrize(
    ("splits", "expected"),
    [
        ({"right": (0, 6), "torso": (6, 7), "base": (7, 10), "right_gripper": (10, 11)}, (0, 6, 6, 7, 10, 10, 11)),
        ({"right": (0, 6), "right_gripper": (6, 7), "base": (7, 10), "torso": (10, 11)}, (0, 6, 10, 7, 10, 6, 11)),
    ],
)
def test_action_layout_is_derived_from_live_controller_part_splits(splits, expected):
    from tools.vla82_full_sim.expert import ActionLayout
    controller = type("C", (), {"_action_split_indexes": splits})()
    robot = type("R", (), {"composite_controller": controller})()
    environment = type("E", (), {"action_space": type("A", (), {"shape": (12,)})(), "unwrapped": type("U", (), {"env": type("Raw", (), {"robots": (robot,)})()})()})()
    layout = ActionLayout.from_env(environment)
    assert (layout.arm[0], layout.arm[1], layout.torso, layout.base[0], layout.base[1], layout.gripper, layout.base_mode) == expected


def test_trash_can_scene_uses_right_only_world_whole_body_ik_configuration():
    from tools.vla82_full_sim.environment import vla82_trash_can_whole_body_ik_config

    config = vla82_trash_can_whole_body_ik_config()
    specific = config["composite_controller_specific_configs"]
    assert config["type"] == "WHOLE_BODY_IK"
    assert specific["actuation_part_names"] == ["right"]
    assert specific["ref_name"] == ["gripper0_right_grip_site"]
    assert specific["ik_input_ref_frame"] == "world"
    assert specific["ik_input_type"] == "keyboard"
    # ``controller_configs`` is passed directly to RoboSuite, unlike a JSON
    # path passed through its normal loader; its body parts must therefore
    # already be flattened to the runtime part names.
    assert set(config["body_parts"]) == {"right", "torso", "base"}


def test_whole_body_layout_uses_only_real_whole_body_split_indexes():
    from tools.vla82_full_sim.expert import WholeBodyActionLayout

    controller = type("C", (), {
        "_whole_body_controller_action_split_indexes": {
            "right": (0, 6), "torso": (6, 7), "base": (7, 10), "right_gripper": (10, 11),
        },
    })()
    layout = WholeBodyActionLayout.from_controller(controller, action_dim=11)
    assert (layout.right, layout.torso, layout.base, layout.right_gripper) == ((0, 6), (6, 7), (7, 10), (10, 11))
    assert not hasattr(layout, "base_mode")


def test_whole_body_pose_action_uses_public_vector_builder_with_absolute_right_pose():
    from tools.vla82_full_sim.expert import WholeBodyActionLayout, whole_body_pose_action

    class _Controller:
        _whole_body_controller_action_split_indexes = {
            "right": (0, 6), "torso": (6, 7), "base": (7, 10), "right_gripper": (10, 11),
        }

        def __init__(self):
            self.received = None

        def create_action_vector(self, action_dict):
            self.received = action_dict
            return np.concatenate((action_dict["right"], action_dict["torso"], action_dict["base"], action_dict["right_gripper"]))

    controller = _Controller()
    layout = WholeBodyActionLayout.from_controller(controller, action_dim=11)
    action = whole_body_pose_action(
        controller, layout, position=np.array((.1, .2, .3)), axis_angle=np.array((.4, .5, .6)),
        torso_qpos=.21, closed=False,
    )
    np.testing.assert_allclose(controller.received["right"], [.1, .2, .3, .4, .5, .6])
    np.testing.assert_allclose(controller.received["torso"], [.21])
    np.testing.assert_allclose(controller.received["base"], [0., 0., 0.])
    np.testing.assert_allclose(controller.received["right_gripper"], [-1.])
    assert action.shape == (11,)


def test_pad_midpoint_world_uses_only_two_finger_pad_collision_geometries():
    from tools.vla82_full_sim.expert import pad_midpoint_world

    names = ("table", "robot0_rightfinger_pad_collision", "robot0_leftfinger_pad_collision", "finger_visual")
    raw = type("Raw", (), {
        "sim": type("Sim", (), {
            "model": type("Model", (), {"ngeom": len(names), "geom_id2name": lambda self, index: names[index]})(),
            "data": type("Data", (), {"geom_xpos": np.array(((9., 9., 9.), (.1, .2, .3), (.5, .6, .7), (8., 8., 8.)))})(),
        })(),
    })()
    np.testing.assert_allclose(pad_midpoint_world(raw), [.3, .4, .5])


def test_eef_target_tracks_desired_pad_midpoint_from_live_pad_midpoint():
    from tools.vla82_full_sim.expert import eef_target_for_pad_midpoint

    target = eef_target_for_pad_midpoint(
        eef_world=np.array((.1, .2, .3)), live_pad_midpoint=np.array((.11, .23, .31)),
        desired_pad_midpoint=np.array((.4, .5, .6)),
    )
    np.testing.assert_allclose(target, [.39, .47, .59])


def test_grasp_pad_target_stays_strictly_inside_side_band_and_avoids_old_can_center_drop():
    from tools.vla82_full_sim.expert import grasp_pad_target

    can = np.array((.04, .10, .855))
    half_height, margin, inset = .055, .012, .003
    target = grasp_pad_target(can, half_height, margin=margin, inset=inset)
    lower, upper = can[2] - half_height + margin, can[2] + half_height - margin
    np.testing.assert_allclose(target[:2], can[:2])
    assert lower < target[2] < upper
    assert np.isclose(can[2] - target[2], -(half_height - margin - inset))
    assert target[2] - can[2] > .039  # old center target demanded a ~4cm deeper drop


def test_whole_body_grasp_gate_requires_side_band_and_three_physical_pinch_frames_before_lift():
    from tools.vla82_full_sim.expert import WholeBodyGraspGate

    gate = WholeBodyGraspGate()
    assert gate.observe(horizontal_distance=.020, pad_in_side_band=False, raw_grasp=False, two_pad_contact=False) == "approach_safe"
    assert gate.observe(horizontal_distance=.003, pad_in_side_band=False, raw_grasp=False, two_pad_contact=False) == "descend_to_side_band"
    assert gate.observe(horizontal_distance=.003, pad_in_side_band=True, raw_grasp=False, two_pad_contact=False) == "close"
    assert gate.observe(horizontal_distance=.003, pad_in_side_band=True, raw_grasp=True, two_pad_contact=True) == "close"
    assert gate.observe(horizontal_distance=.003, pad_in_side_band=True, raw_grasp=True, two_pad_contact=True) == "close"
    assert gate.observe(horizontal_distance=.003, pad_in_side_band=True, raw_grasp=True, two_pad_contact=True) == "lift"


def test_whole_body_descend_keeps_descending_through_small_xy_drift_but_recovers_after_30mm():
    from tools.vla82_full_sim.expert import WholeBodyGraspGate

    gate = WholeBodyGraspGate()
    assert gate.observe(horizontal_distance=.003, pad_in_side_band=False, raw_grasp=False, two_pad_contact=False) == "descend_to_side_band"
    assert gate.observe(horizontal_distance=.006, pad_in_side_band=False, raw_grasp=False, two_pad_contact=False) == "descend_to_side_band"
    assert gate.observe(horizontal_distance=.031, pad_in_side_band=False, raw_grasp=False, two_pad_contact=False) == "approach_safe"


def test_vla82_001_routes_whole_body_ik_to_new_expert_and_keeps_basic_on_legacy_expert():
    from tools.vla82_full_sim.expert import (
        MultiObjectDepositExpert, WholeBodyDepositExpert, deposit_expert_class_for_controller,
    )

    whole_body = type("Whole", (), {"_whole_body_controller_action_split_indexes": {"right": (0, 6)}})()
    basic = type("Basic", (), {})()
    assert deposit_expert_class_for_controller("VLA82-001", whole_body) is WholeBodyDepositExpert
    assert deposit_expert_class_for_controller("VLA82-001", basic) is MultiObjectDepositExpert
    assert deposit_expert_class_for_controller("VLA82-002", whole_body) is MultiObjectDepositExpert


def test_whole_body_deposit_captures_initial_torso_hold_once_and_reuses_it(monkeypatch):
    from tools.vla82_full_sim import expert

    class _Torso:
        def __init__(self):
            self.reads = 0

        @property
        def joint_pos(self):
            self.reads += 1
            return np.array((.2 + .01 * self.reads,))

    torso = _Torso()
    controller = type("C", (), {
        "_whole_body_controller_action_split_indexes": {"right": (0, 6), "torso": (6, 7), "base": (7, 10), "right_gripper": (10, 11)},
        "part_controllers": {"torso": torso},
    })()
    names = ("robot0_rightfinger_pad_collision", "robot0_leftfinger_pad_collision")
    model = type("M", (), {"ngeom": 2, "nsite": 0, "geom_id2name": lambda self, index: names[index], "body_name2id": lambda self, name: 0})()
    data = type("D", (), {"site_xpos": np.array(((.0, .0, 1.0),)), "site_xmat": np.array((np.eye(3).reshape(-1),)), "body_xpos": np.array(((.0, .0, .85),)), "geom_xpos": np.array(((.0, .0, .85), (.0, .0, .85))), "ncon": 0})()
    obj = type("O", (), {"root_body": "obj", "bottom_offset": np.array((0., 0., .055))})()
    robot = type("R", (), {"composite_controller": controller, "eef_site_id": {"right": 0}, "gripper": {"right": object()}})()
    raw = type("Raw", (), {"robots": (robot,), "sim": type("S", (), {"model": model, "data": data})(), "objects": {"obj": obj}, "_check_grasp": lambda *args: False})()
    env = type("E", (), {"action_space": type("A", (), {"shape": (11,)})(), "unwrapped": type("U", (), {"env": raw})()})()
    received = []
    monkeypatch.setattr(expert, "whole_body_pose_action", lambda controller, layout, **kwargs: received.append(np.asarray(kwargs["torso_qpos"]).copy()) or np.zeros(11, dtype=np.float32))
    primitive = expert.WholeBodyDepositExpert(type("Contract", (), {"object_geom_names": {"obj": ()}})())
    rollout = primitive.run(env)
    next(rollout)
    next(rollout)
    assert len(received) == 2
    np.testing.assert_allclose(received[0], received[1])
    assert torso.reads == 1


def test_whole_body_deposit_gate_requires_pinch_release_geometry_and_eight_open_stable_frames():
    from tools.vla82_full_sim.expert import WholeBodyDepositGate

    low, high = np.array((.22, .20, .81)), np.array((.34, .32, 1.11))
    gate = WholeBodyDepositGate(cavity_min=low, cavity_max=high, signed_bottom_offset=-.055)
    raised = np.array((.0, .0, 1.205))  # bottom is 4cm above the rim
    assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=raised, pose_stable=False) == "transit_above_cavity"
    above_center = np.array((.28, .26, 1.205))
    assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=above_center, pose_stable=False) == "lower_inside"
    inside = np.array((.28, .26, .96))
    assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=inside, pose_stable=False) == "release"
    assert gate.observe(raw_grasp=False, two_pad_contact=False, can_position=inside, pose_stable=True) == "settle"
    for _ in range(7):
        assert gate.observe(raw_grasp=False, two_pad_contact=False, can_position=inside, pose_stable=True) == "settle"
    assert gate.observe(raw_grasp=False, two_pad_contact=False, can_position=inside, pose_stable=True) == "complete"


def test_whole_body_deposit_gate_refuses_transport_after_lost_two_pad_pinch():
    from tools.vla82_full_sim.expert import WholeBodyDepositGate

    gate = WholeBodyDepositGate(cavity_min=np.array((.22, .20, .81)), cavity_max=np.array((.34, .32, 1.11)), signed_bottom_offset=-.055)
    assert gate.observe(raw_grasp=False, two_pad_contact=False, can_position=np.array((.0, .0, 1.0)), pose_stable=False) == "reacquire"


def test_whole_body_deposit_gate_requires_slot_alignment_before_lowering() -> None:
    """Merely crossing the cavity edge must not start a wall-striking descent."""
    from tools.vla82_full_sim.expert import WholeBodyDepositGate

    low, high = np.array((.167, .317, .81)), np.array((.373, .523, 1.11))
    slot = np.array((.22, .375))
    gate = WholeBodyDepositGate(cavity_min=low, cavity_max=high, signed_bottom_offset=-.055, slot_xy=slot)
    raised = np.array((.0, .0, 1.21))
    assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=raised, pose_stable=False) == "transit_above_cavity"
    # This is physically inside the raw cavity bounds, but the complete can is
    # still about 38 mm from its slot and cannot descend without hitting a wall.
    edge_crossing = np.array((.1903, .3512, 1.21))
    assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=edge_crossing, pose_stable=False) == "transit_above_cavity"
    aligned = np.array((.222, .372, 1.21))
    assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=aligned, pose_stable=False) == "lower_inside"


def test_whole_body_deposit_gate_bounds_an_unreached_raise_above_rim():
    from tools.vla82_full_sim.expert import WholeBodyDepositGate

    gate = WholeBodyDepositGate(cavity_min=np.array((.22, .20, .81)), cavity_max=np.array((.34, .32, 1.11)), signed_bottom_offset=-.055)
    below_rim = np.array((.0, .0, 1.0))
    for _ in range(128):
        assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=below_rim, pose_stable=False) == "raise_above_rim"
    assert gate.observe(raw_grasp=True, two_pad_contact=True, can_position=below_rim, pose_stable=False) == "raise_above_rim"
    assert gate.failed is True


def test_whole_body_expert_continues_closed_into_raise_after_lift_proof(monkeypatch):
    from tools.vla82_full_sim import expert

    class _Grasp:
        LIMITS = {"close": 2, "lift": 2}
        def __init__(self): self.state, self.failed, self.pinch_steps = "close", False, 3
        def observe(self, **kwargs): self.state = "lift"; return self.state
    monkeypatch.setattr(expert, "WholeBodyGraspGate", _Grasp)
    controller = type("C", (), {"_whole_body_controller_action_split_indexes": {"right": (0, 6), "torso": (6, 7), "base": (7, 10), "right_gripper": (10, 11)}, "part_controllers": {"torso": type("T", (), {"joint_pos": np.array((.2,))})()}})()
    names = ("robot0_rightfinger_pad_collision", "robot0_leftfinger_pad_collision")
    model = type("M", (), {"ngeom": 2, "nsite": 0, "geom_id2name": lambda self, index: names[index], "body_name2id": lambda self, name: 0})()
    data = type("D", (), {"site_xpos": np.array(((.0, .0, 1.0),)), "site_xmat": np.array((np.eye(3).reshape(-1),)), "body_xpos": np.array(((.0, .0, .85),)), "geom_xpos": np.array(((.0, .0, .85), (.0, .0, .85))), "ncon": 0})()
    obj = type("O", (), {"root_body": "obj", "bottom_offset": np.array((0., 0., -.055))})()
    robot = type("R", (), {"composite_controller": controller, "eef_site_id": {"right": 0}, "gripper": {"right": object()}})()
    raw = type("Raw", (), {"robots": (robot,), "sim": type("S", (), {"model": model, "data": data})(), "objects": {"obj": obj}, "_check_grasp": lambda *args: True})()
    env = type("E", (), {"action_space": type("A", (), {"shape": (11,)})(), "unwrapped": type("U", (), {"env": raw})(), "step": lambda self, action: setattr(data.body_xpos, "__setitem__", data.body_xpos.__setitem__)})()
    # The fake environment only needs to make the physical can exceed the 45mm lift proof.
    def step(action): data.body_xpos[0, 2] = .90
    env.step = step
    monkeypatch.setattr(expert, "whole_body_pose_action", lambda controller, layout, **kwargs: np.array([0.] * 10 + [1.], dtype=np.float32))
    monkeypatch.setattr(expert, "selected_geom_has_two_finger_pad_contacts", lambda *args: True)
    target = type("Target", (), {"min_corner": (.22, .20, .81), "max_corner": (.34, .32, 1.11)})()
    primitive = expert.WholeBodyDepositExpert(type("Contract", (), {"object_geom_names": {"obj": ()}, "target_geometries": {"bin": target}})())
    rollout = primitive.run(env)
    next(rollout)
    env.step(None)
    _, second = next(rollout)
    assert second[10] == 1.0
    assert primitive.trace[-1]["stage"] == "raise_above_rim"
    desired = np.asarray(primitive.trace[-1]["desired_can_world"], dtype=float)
    assert desired[2] - .055 >= 1.11 + .045 - 1e-9


def test_vla001_whole_body_plan_binds_each_source_can_once_to_its_own_slot():
    from tools.vla82_full_sim.expert import vla82_whole_body_deposit_plan

    plan = vla82_whole_body_deposit_plan()
    assert tuple(item[0] for item in plan) == ("obj", "annotated_01", "annotated_02")
    assert len({tuple(slot) for _, slot in plan}) == 3
    assert np.allclose(np.asarray([slot for _, slot in plan]), ((.22, .375), (.32, .375), (.27, .465)))


def test_whole_body_expert_runs_three_independent_object_slot_cycles_and_exits_between(monkeypatch):
    from tools.vla82_full_sim import expert

    calls: list[tuple[str, tuple[float, float]]] = []
    exits: list[str] = []
    def one(self, environment, object_id, slot_xy):
        calls.append((object_id, tuple(slot_xy)))
        yield "deposit", np.zeros(11, dtype=np.float32)
        return True
    def exit_bin(self, environment, *, after_object_id):
        exits.append(after_object_id)
        yield "deposit", np.zeros(11, dtype=np.float32)
        return True
    monkeypatch.setattr(expert.WholeBodyDepositExpert, "_run_one", one)
    monkeypatch.setattr(expert.WholeBodyDepositExpert, "_exit_open_bin", exit_bin)
    primitive = expert.WholeBodyDepositExpert(object())
    rollout = primitive.run(object())
    while True:
        try:
            next(rollout)
        except StopIteration:
            break
    assert tuple(item[0] for item in calls) == ("obj", "annotated_01", "annotated_02")
    assert exits == ["obj", "annotated_01"]


def test_diagnostic_json_accepts_live_scene_numpy_scalars(tmp_path: Path):
    from tools.vla82_full_sim.expert import _atomic_json
    target = tmp_path / "diagnostic.json"
    _atomic_json(target, {"layout_id": np.int64(7), "pose": np.array([1.0, 2.0])})
    assert '"layout_id": 7' in target.read_text(encoding="utf-8")


def test_cleaning_coverage_tracker_derives_reduction_only_from_exact_contact_cells():
    """The capture record may derive coverage, but never write MuJoCo metrics."""
    from tools.vla82_full_sim.environment import ContactEvidence, PhysicsSnapshot, TargetGeometry
    from tools.vla82_full_sim.expert import CleaningCoverageTracker

    target_id = "counter:clean_region"
    target = TargetGeometry(target_id, "counter", (-.2, -.2, .0), (.2, .2, .1), target_id, "clean-region")
    def physical(step: int, x: float) -> PhysicsSnapshot:
        return PhysicsSnapshot(
            step=step, robot_qpos=np.zeros(7), gripper_qpos=np.array([.01]),
            body_poses={"obj": np.array([x, .0, .05, 1., 0., 0., 0.])}, joint_positions={},
            contacts=(("obj_collision", "counter_surface", -.001),), dirt_fraction=1., spray_coverage=0., dispensed_amount=0.,
            contact_evidence=(ContactEvidence("obj_collision", "counter_surface", "obj", target_id, "counter", None, -.001),),
            target_geometries={target_id: target}, fixture_joint_ids={},
        )
    tracker = CleaningCoverageTracker(target_id)
    first = tracker.derive(physical(1, .00))
    duplicate = tracker.derive(physical(2, .00))
    third = tracker.derive(physical(3, .04))
    assert first.dirt_fraction == pytest.approx(.90)
    assert duplicate.dirt_fraction == pytest.approx(.90)
    assert third.dirt_fraction == pytest.approx(.80)
    assert tracker.covered_cells == 2


def test_cleaning_path_uses_three_distinct_in_target_grid_cells():
    from tools.vla82_full_sim.environment import TargetGeometry
    from tools.vla82_full_sim.expert import cleaning_coverage_points

    target = TargetGeometry("counter:clean", "counter", (-.20, -.20, .0), (.20, .20, .1), "counter", "clean-region")
    points = cleaning_coverage_points(np.array((.0, .0, .02)), target)

    assert len(points) >= 3
    assert all(np.all(point[:2] >= np.asarray(target.min_corner)[:2]) for point in points)
    assert all(np.all(point[:2] <= np.asarray(target.max_corner)[:2]) for point in points)
    assert min(float(np.linalg.norm(first[:2] - second[:2])) for index, first in enumerate(points) for second in points[index + 1:]) >= .02
    assert all(point[1] == pytest.approx(points[0][1]) for point in points)


def test_wipe_regrasp_recovery_requires_loss_and_is_bounded_to_two_attempts():
    from tools.vla82_full_sim.expert import should_recover_wipe_grasp

    assert should_recover_wipe_grasp(grasped=False, exact_gripper_contact=True, attempts=0) is True
    assert should_recover_wipe_grasp(grasped=True, exact_gripper_contact=True, attempts=0) is False
    assert should_recover_wipe_grasp(grasped=False, exact_gripper_contact=True, attempts=2) is False
    assert should_recover_wipe_grasp(grasped=False, exact_gripper_contact=False, attempts=0, wipe_complete=True) is False


def test_cleaning_credit_requires_raw_grasp_and_two_pad_contact():
    from tools.vla82_full_sim.expert import held_for_cleaning_credit
    assert held_for_cleaning_credit(raw_grasp=True, two_pad_contact=True) is True
    assert held_for_cleaning_credit(raw_grasp=False, two_pad_contact=True) is False
    assert held_for_cleaning_credit(raw_grasp=True, two_pad_contact=False) is False


def test_compliant_descend_switches_to_micro_limit_within_three_mm():
    from tools.vla82_full_sim.expert import compliant_descend_limit
    assert compliant_descend_limit(.004) == pytest.approx(.05)
    assert compliant_descend_limit(.003) == pytest.approx(.01)


def test_compliant_descend_command_has_fast_mid_and_contact_bands():
    from tools.vla82_full_sim.expert import compliant_descend_command
    assert compliant_descend_command(.02) == pytest.approx(-.70)
    assert compliant_descend_command(.005) == pytest.approx(-.10)
    assert compliant_descend_command(.002) == pytest.approx(-.01)


def test_composite_expert_selects_live_cleaning_controller(monkeypatch):
    from tools.vla82_full_sim import expert
    contract = type("Contract", (), {"target_geometries": {"counter:clean": object()}})()
    selected: list[str] = []
    class FakeCleaning:
        def __init__(self, phases, received_contract):
            assert phases == ("grasp", "wipe", "return")
            assert received_contract is contract
            self.trace = []
        def actions(self, environment):
            selected.append("cleaning")
            yield "grasp", np.zeros(12, dtype=np.float32)
    monkeypatch.setattr(expert, "CleaningPrimitive", FakeCleaning)
    emitted = list(expert.CompositeExpert(("grasp", "wipe", "return"), contract).run(object()))
    assert selected == ["cleaning"]
    assert emitted[0][0] == "grasp"


@pytest.mark.parametrize("selection_id", ("VLA82-004", "VLA82-005"))
def test_composite_expert_selects_live_pick_place_controller_for_grasp_place_family(monkeypatch, selection_id):
    from tools.vla82_full_sim import expert

    selected: list[tuple[str, ...]] = []
    class FakePickPlace:
        def __init__(self, phases, contract):
            assert contract == "contract"
            self.phases = tuple(phases)
            self.trace = [{"selected": "pick_place"}]
        def actions(self, environment):
            selected.append(self.phases)
            yield "grasp", np.zeros(12, dtype=np.float32)
    monkeypatch.setattr(expert, "PickPlaceExpert", FakePickPlace, raising=False)
    monkeypatch.setattr(expert.PrimitiveExpert, "actions", lambda self, environment: iter((np.zeros(12, dtype=np.float32),)))
    environment = type("E", (), {"request": type("R", (), {"selection_id": selection_id})()})()

    controller = expert.CompositeExpert(("grasp", "place"), "contract")
    phase, _ = next(controller.run(environment))

    assert phase == "grasp"
    assert selected == [("grasp", "place")]


def test_pick_place_base_reach_never_promotes_a_far_source_after_24_steps():
    from tools.vla82_full_sim.expert import pick_place_base_reach_state, pick_place_base_reach_threshold

    assert pick_place_base_reach_state(distance=.45, steps=24, maximum_steps=96) == "reach_base"
    assert pick_place_base_reach_state(distance=.30, steps=24, maximum_steps=96) == "approach"
    assert pick_place_base_reach_state(distance=.45, steps=96, maximum_steps=96) == "failed"
    assert pick_place_base_reach_threshold("VLA82-053") == pytest.approx(.50)
    assert pick_place_base_reach_threshold("VLA82-058") == pytest.approx(.65)


def test_toothbrush_release_accepts_measured_clearance_from_the_cup():
    from tools.vla82_full_sim.expert import pick_place_retreat_complete

    assert pick_place_retreat_complete(
        "VLA82-058",
        eef=(.055, -.174, .950),
        object_center=(.141, -.018, .878),
        retreat_target=(.20, -.20, 1.00),
    )
    assert not pick_place_retreat_complete(
        "VLA82-058",
        eef=(.141, -.018, .900),
        object_center=(.141, -.018, .878),
        retreat_target=(.20, -.20, 1.00),
    )


def test_toothbrush_is_not_released_against_the_outside_of_the_cup():
    from tools.vla82_full_sim.expert import pick_place_target_contact_release_ready

    assert not pick_place_target_contact_release_ready(
        "VLA82-058", target_contact=True, inside_xy=False, eef_distance=.01,
    )
    assert pick_place_target_contact_release_ready(
        "VLA82-058", target_contact=True, inside_xy=True, eef_distance=.01,
    )


def test_toothbrush_clears_and_centres_over_the_cup_before_lowering():
    from tools.vla82_full_sim.expert import (
        pick_place_lift_proof_height,
        pick_place_transit_ready_for_selection,
    )

    assert pick_place_lift_proof_height("VLA82-058", .045) == pytest.approx(.12)
    assert not pick_place_transit_ready_for_selection(
        "VLA82-058", eef=(0., 0., 0.), desired=(0., 0., 0.),
        object_center=(.14, -.09, .99), release=(.14, -.12, .906),
    )
    assert pick_place_transit_ready_for_selection(
        "VLA82-058", eef=(0., 0., 0.), desired=(0., 0., 0.),
        object_center=(.14, -.115, .99), release=(.14, -.12, .906),
    )


def test_vla044_plate_uses_observed_reachable_arm_workspace_threshold():
    from tools.vla82_full_sim.expert import pick_place_base_reach_threshold

    assert pick_place_base_reach_threshold("VLA82-044") == pytest.approx(.66)


def test_retry_lift_never_drives_a_displaced_object_back_down_to_spawn_height():
    from tools.vla82_full_sim.expert import pick_place_lift_target

    target = pick_place_lift_target(
        source=(1.85, -.44, 1.04),
        grasp_source=(1.85, -.44, 1.54),
        grasp_offset=(0., 0., .02),
        lift_height=.12,
    )

    assert target[2] == pytest.approx(1.68)


def test_drawer_retreat_keeps_holding_with_opposed_finger_shell_contact():
    from tools.vla82_full_sim.expert import drawer_hold_valid

    assert drawer_hold_valid(raw_grasp=False, two_finger_contact=True)
    assert drawer_hold_valid(raw_grasp=True, two_finger_contact=False)
    assert not drawer_hold_valid(raw_grasp=False, two_finger_contact=False)


def test_only_drawer_transport_accepts_opposed_shell_contact_as_active_hold():
    from tools.vla82_full_sim.expert import pick_place_transport_hold_valid

    assert pick_place_transport_hold_valid(
        drawer_pick_place=True, raw_grasp=False, two_finger_contact=True,
    )
    assert not pick_place_transport_hold_valid(
        drawer_pick_place=False, raw_grasp=False, two_finger_contact=True,
    )


def test_vla004_pick_place_uses_front_entry_after_collision_free_reset_alignment():
    from tools.vla82_full_sim.expert import (
        cabinet_bypass_y_target, cabinet_extraction_target, cabinet_outward_from_base_rotation,
        pick_place_extraction_lift_height, pick_place_initial_state,
        pick_place_lift_torso_command, pick_place_post_lift_state,
        pick_place_close_world_target, pick_place_descent_reached,
        pick_place_transit_reached,
        open_drawer_release_point,
        inside_release_height,
        rotated_geom_vertical_half_extent,
        drawer_drop_ready,
        update_stable_release_counter,
        pick_place_lower_limit,
        drawer_orientation_correction,
        drawer_depth_alignment_error,
        drawer_release_depth_coordinate,
        drawer_pitch_rotation_direction,
        drawer_transit_world_target,
        drawer_base_retreat_required,
        drawer_base_retreat_local_direction,
        drawer_wrist_relief_required,
        pick_place_maximum_gravity_drop,
        drawer_gravity_release_ceiling,
        drawer_transit_ready,
            drawer_horizontal_rotation_command,
            drawer_orientation_pose_ready,
            update_drawer_horizontal_latch,
            drawer_transport_level_rotation_command,
            drawer_vertical_transport_threshold,
            drawer_transit_control_stage,
            drawer_wrist_relief_feedback,
            drawer_lower_pose_correction_required,
            drawer_torso_reserve_required,
            drawer_lower_torso_command,
            drawer_base_retreat_distance,
            drawer_lower_rotation_command,
            drawer_gravity_release_pose_ready,
            drawer_lower_base_recenter_required,
            drawer_lower_base_recenter_local_direction,
            drawer_release_front_bias,
            drawer_post_release_retreat_target,
            drawer_front_insertion_targets,
            pick_place_recorded_stability_ready,
            drawer_lower_yaw_tolerance,
            drawer_lower_base_recenter_enabled,
            drawer_lower_pose_release_threshold,
            drawer_transit_xy_tolerance,
        shortest_axis_alignment_error,
    )

    # The two measured open-door spans leave no safe direct x corridor for the
    # 0.25 m half-depth mobile base.  Choose the nearest exterior endpoint
    # with 2 cm clearance before attempting the cross-corridor x segment.
    assert pick_place_initial_state("VLA82-004") == "cabinet_front_stage"
    assert pick_place_post_lift_state("VLA82-004") == "cabinet_extract"
    assert pick_place_post_lift_state("VLA82-005") == "orient_for_drawer"
    assert pick_place_post_lift_state("VLA82-002") == "transit"
    assert pick_place_lift_torso_command("VLA82-004", grasped=True) == pytest.approx(.2)
    assert pick_place_lift_torso_command("VLA82-004", grasped=False) == 0.0
    assert pick_place_lift_torso_command("VLA82-002", grasped=True) == 0.0
    assert pick_place_extraction_lift_height("VLA82-004", nominal=.14) == pytest.approx(.05)
    assert pick_place_extraction_lift_height("VLA82-002", nominal=.14) == pytest.approx(.14)
    assert np.allclose(cabinet_outward_from_base_rotation(-np.eye(3)), (1., 0.))
    assert np.allclose(
        cabinet_extraction_target(
            source=(.226, -4.298, 1.495), grasp_offset=(.027, 0., .001),
            outward=(1., 0.), lift_height=.08, extraction_distance=.25,
        ),
        (.503, -4.298, 1.576),
    )
    assert cabinet_bypass_y_target(
        door_y_mins=(-5.0275, -4.2275, -2.7775), door_y_maxs=(-4.2305, -3.8305, -2.2305),
        base_half_y=.25, current_y=-4.10875,
    ) == pytest.approx(-3.5605)
    assert pick_place_initial_state("VLA82-002") == "reach_base"
    assert pick_place_initial_state("VLA82-005") == "align_yaw"
    assert abs(shortest_axis_alignment_error((1., 0.), (0., 1.))) == pytest.approx(np.pi / 2)
    assert pick_place_descent_reached(distance=.074, any_gripper_contact=True)
    assert not pick_place_descent_reached(distance=.074, any_gripper_contact=False)
    assert np.allclose(
        pick_place_close_world_target("VLA82-005", eef=(1., 2., 3.), obj=(4., 5., 6.), any_gripper_contact=True),
        (1., 2., 3.),
    )
    assert np.allclose(
        pick_place_close_world_target("VLA82-004", eef=(1., 2., 3.), obj=(4., 5., 6.), any_gripper_contact=True),
        (4., 5., 6.),
    )
    assert pick_place_transit_reached(eef=(1.002, 2.001, 1.15), desired=(1., 2., 1.06))
    assert not pick_place_transit_reached(eef=(1.08, 2., 1.06), desired=(1., 2., 1.06))
    assert np.allclose(
        open_drawer_release_point(
            target_low=(1.7475, -.856945, .665), target_high=(2.1025, 0., .736),
            robot_base=(2.48, -.80, 0.), margin=.14,
        )[:2],
        (1.925, -.716945),
    )
    rotation = np.array(((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.)))
    assert inside_release_height(
        target_low_z=.665, geom_size=(.012, .012, .105), geom_rotation=rotation,
    ) == pytest.approx(.682)
    assert rotated_geom_vertical_half_extent(
        geom_size=(.012, .012, .105), geom_rotation=rotation,
    ) == pytest.approx(.012)
    assert drawer_drop_ready(
        object_center=(1.95, -.714, .923), target_low=(1.7475, -.8569, .665),
        target_high=(2.1025, 0., .736), horizontal_margin=.03, maximum_drop=.20,
    )
    assert not drawer_drop_ready(
        object_center=(1.95, -.714, 1.10), target_low=(1.7475, -.8569, .665),
        target_high=(2.1025, 0., .736), horizontal_margin=.03, maximum_drop=.20,
    )
    assert drawer_drop_ready(
        object_center=(1.95, -.714, .98), target_low=(1.7475, -.8569, .665),
        target_high=(2.1025, 0., .736), horizontal_margin=.03, maximum_drop=.25,
    )
    assert not drawer_drop_ready(
        object_center=(1.95, -.714, .78), target_low=(1.7475, -.8569, .665),
        target_high=(2.1025, 0., .736), horizontal_margin=.03,
    )
    assert drawer_drop_ready(
        object_center=(1.95, -.714, .85), target_low=(1.7475, -.8569, .665),
        target_high=(2.1025, 0., .736), horizontal_margin=.03, maximum_drop=.12,
    )
    assert drawer_drop_ready(
        object_center=(1.95, -.714, .735), target_low=(1.7475, -.8569, .665),
        target_high=(2.1025, 0., .736), horizontal_margin=.03,
    )
    assert update_stable_release_counter(previous=(1., 2., 3.), current=(1., 2., 3.0005), target_contact=True, count=1) == 2
    assert update_stable_release_counter(previous=(1., 2., 3.), current=(1., 2., 3.01), target_contact=True, count=1) == 0
    assert pick_place_lower_limit("VLA82-005") == pytest.approx(.8)
    assert pick_place_lower_limit("VLA82-004") == pytest.approx(.22)
    assert drawer_orientation_correction(.055) == pytest.approx(.6)
    assert drawer_orientation_correction(.025) == 0.0
    # Once horizontal, the whisk's 210 mm collision long axis must run along
    # the draw's 510 mm depth, not across its 233 mm usable width.
    assert abs(drawer_depth_alignment_error(
        long_axis=(1., 0., 0.),
        target_low=(1.7475, -.856945, .665),
        target_high=(2.1025, 0., .736),
    )) == pytest.approx(np.pi / 2)
    assert drawer_depth_alignment_error(
        long_axis=(0., -1., 0.),
        target_low=(1.7475, -.856945, .665),
        target_high=(2.1025, 0., .736),
    ) == pytest.approx(0.)
    assert drawer_release_depth_coordinate(
        target_low=(1.7475, -.856945, .665),
        target_high=(2.1025, 0., .736),
        robot_base=(1.9, -1.006945, 0.),
        world_half_extents=(.012, .105, .012),
        clearance=.015,
    ) == (1, pytest.approx(-.736945))
    assert drawer_pitch_rotation_direction(
        target_low=(1.7475, -.856945, .665),
        target_high=(2.1025, 0., .736),
        robot_base=(1.9, -1.006945, 0.),
    ) == pytest.approx(-1.)
    assert np.allclose(
        drawer_transit_world_target(
            eef=(2.0168, -.8143, 1.0989), desired=(1.8985, -.8100, .8342),
        ),
        (1.8985, -.8100, 1.0989),
    )
    assert drawer_base_retreat_required(
        base=(1.9, -1.0069, 0.), release=(1.925, -.7369, .70), minimum_distance=.65,
    )
    assert not drawer_base_retreat_required(
        base=(1.9, -1.4069, 0.), release=(1.925, -.7369, .70), minimum_distance=.65,
    )
    assert np.allclose(
        drawer_base_retreat_local_direction(
            base=(1.9, -1.0069, 0.), release=(1.925, -.7369, .70), base_rotation=np.eye(3),
        ),
        (-.0922, -.9957), atol=1e-3,
    )
    assert drawer_wrist_relief_required(joint5=-2.89755)
    assert drawer_wrist_relief_required(joint5=-2.75)
    assert not drawer_wrist_relief_required(joint5=-2.70)
    assert pick_place_maximum_gravity_drop("VLA82-005") == pytest.approx(.12)
    assert pick_place_maximum_gravity_drop("VLA82-006") == pytest.approx(.03)
    assert pick_place_maximum_gravity_drop("VLA82-004") == 0.0
    assert drawer_transit_xy_tolerance("VLA82-006") == pytest.approx(.012)
    assert drawer_transit_xy_tolerance("VLA82-005") == pytest.approx(.035)
    assert drawer_gravity_release_ceiling(
        floor_top_z=.731, vertical_half_extent=.030, clearance=.005, maximum_drop=.12,
    ) == pytest.approx(.886)
    assert not drawer_transit_ready(
        xy_reached=True, vertical_half_extent=.086, depth_alignment_error=np.deg2rad(3.), joint5=-2.60,
    )
    assert not drawer_transit_ready(
        xy_reached=True, vertical_half_extent=.029, depth_alignment_error=np.deg2rad(3.), joint5=-2.85,
    )
    assert drawer_transit_ready(
        xy_reached=True, vertical_half_extent=.029, depth_alignment_error=np.deg2rad(3.), joint5=-2.60,
    )
    assert np.allclose(
        drawer_horizontal_rotation_command(
            long_axis=(0., 0., 1.), target_low=(1.7, -.85, .70), target_high=(2.1, -.31, .82),
        ),
        (-.25, 0., 0.), atol=1e-6,
    )
    assert np.allclose(
        drawer_horizontal_rotation_command(
            long_axis=(0., -.01, .999), target_low=(1.7, -.85, .70), target_high=(2.1, -.31, .82),
        ),
        (-.25, 0., 0.), atol=1e-6,
    )
    assert np.allclose(
        drawer_horizontal_rotation_command(
            long_axis=(0., -.8, .6), target_low=(1.7, -.85, .70), target_high=(2.1, -.31, .82),
        ),
        (.25, 0., 0.), atol=1e-6,
    )
    assert np.allclose(
        drawer_horizontal_rotation_command(
            long_axis=(0., 1., 0.), target_low=(1.7, -.85, .70), target_high=(2.1, -.31, .82),
        ),
        (0., 0., 0.), atol=1e-6,
    )
    assert drawer_orientation_pose_ready(vertical_half_extent=.03047)
    assert not drawer_orientation_pose_ready(vertical_half_extent=.036)
    assert update_drawer_horizontal_latch(previous=False, vertical_half_extent=.03047)
    assert update_drawer_horizontal_latch(previous=True, vertical_half_extent=.08)
    assert np.allclose(
        drawer_transport_level_rotation_command(
            long_axis=(0., 0., 1.),
            target_low=(1.7, -.85, .70),
            target_high=(2.1, -.31, .82),
        ),
        (-.2, 0., 0.),
    )
    assert drawer_vertical_transport_threshold("VLA82-045") == pytest.approx(.035)
    assert drawer_vertical_transport_threshold("VLA82-006") == pytest.approx(.03)
    assert drawer_transit_control_stage(
        vertical_half_extent=.029, depth_alignment_error=np.deg2rad(3.),
        xy_reached=False, joint5=-2.89,
    ) == "translate"
    assert drawer_transit_control_stage(
        vertical_half_extent=.029, depth_alignment_error=np.deg2rad(20.),
        xy_reached=False, joint5=-2.89,
    ) == "translate"
    assert drawer_transit_control_stage(
        vertical_half_extent=.029, depth_alignment_error=np.deg2rad(30.),
        xy_reached=False, joint5=-2.89,
    ) == "align_depth"
    # Once depth alignment has been physically achieved, do not chatter at
    # the transport threshold while the long object is still moving in XY.
    # The stricter live-angle check is restored above the drawer.
    assert drawer_transit_control_stage(
        vertical_half_extent=.029, depth_alignment_error=np.deg2rad(30.),
        xy_reached=False, joint5=-2.89, depth_alignment_latched=True,
    ) == "translate"
    assert drawer_transit_control_stage(
        vertical_half_extent=.029, depth_alignment_error=np.deg2rad(30.),
        xy_reached=True, joint5=-2.89, depth_alignment_latched=True,
    ) == "align_depth"
    assert drawer_transit_control_stage(
        vertical_half_extent=.035, depth_alignment_error=np.deg2rad(3.),
        xy_reached=False, joint5=-2.89, horizontal_latched=True,
    ) == "translate"
    assert drawer_transit_control_stage(
        vertical_half_extent=.035, depth_alignment_error=np.deg2rad(3.),
        xy_reached=True, joint5=-2.89, horizontal_latched=True,
    ) == "level"
    assert drawer_transit_control_stage(
        vertical_half_extent=.029, depth_alignment_error=np.deg2rad(3.),
        xy_reached=True, joint5=-2.89,
    ) == "relieve_wrist"
    assert drawer_wrist_relief_feedback(
        joint5=-2.895, baseline_joint5=-2.897, attempts=6,
        direction=.4, already_flipped=False,
    ) == (-.4, True)
    assert drawer_wrist_relief_feedback(
        joint5=-2.86, baseline_joint5=-2.897, attempts=6,
        direction=.4, already_flipped=False,
    ) == (.4, False)
    assert not drawer_lower_pose_correction_required(
        vertical_half_extent=.033, horizontal_latched=True,
    )
    assert not drawer_lower_pose_correction_required(
        vertical_half_extent=.050, horizontal_latched=True,
    )
    assert drawer_lower_pose_correction_required(
        vertical_half_extent=.060, horizontal_latched=True,
    )
    assert drawer_lower_pose_correction_required(
        vertical_half_extent=.040, horizontal_latched=True, release_threshold=.030,
    )
    assert drawer_lower_pose_release_threshold("VLA82-006") == pytest.approx(.04)
    assert drawer_lower_pose_release_threshold("VLA82-005") == pytest.approx(.055)
    assert drawer_lower_pose_correction_required(
        vertical_half_extent=.033, horizontal_latched=False,
    )
    assert drawer_torso_reserve_required(
        selection_id="VLA82-006", torso_qpos=.02,
    )
    assert not drawer_torso_reserve_required(
        selection_id="VLA82-006", torso_qpos=.16,
    )
    assert not drawer_torso_reserve_required(
        selection_id="VLA82-005", torso_qpos=.02,
    )
    assert drawer_lower_torso_command(
        selection_id="VLA82-006", joint5=-2.78, torso_qpos=.12,
    ) == pytest.approx(-.5)
    assert drawer_lower_torso_command(
        selection_id="VLA82-006", joint5=-2.60, torso_qpos=.12,
    ) == 0.0
    assert drawer_lower_torso_command(
        selection_id="VLA82-006", joint5=-2.78, torso_qpos=.01,
    ) == 0.0
    assert drawer_base_retreat_distance("VLA82-005") == pytest.approx(.65)
    assert drawer_base_retreat_distance("VLA82-006") == pytest.approx(.75)
    assert not drawer_lower_pose_correction_required(
        vertical_half_extent=.070, horizontal_latched=True, release_threshold=.075,
    )
    assert drawer_lower_yaw_tolerance("VLA82-006") == pytest.approx(np.deg2rad(5.))
    assert drawer_lower_yaw_tolerance("VLA82-005") == pytest.approx(np.deg2rad(25.))
    assert np.allclose(
        drawer_lower_rotation_command(
            long_axis=(0., -.98, .20),
            target_low=(1.7, -.85, .70), target_high=(2.1, -.31, .82),
        ),
        (.25, 0., 0.),
    )
    assert np.allclose(
        drawer_lower_rotation_command(
            long_axis=(0., -.98, -.20),
            target_low=(1.7, -.85, .70), target_high=(2.1, -.31, .82),
        ),
        (-.25, 0., 0.),
    )
    assert drawer_gravity_release_pose_ready(
        pose_correction=False, yaw_correction=False, joint_correction=True,
    )
    assert not drawer_gravity_release_pose_ready(
        pose_correction=True, yaw_correction=False, joint_correction=False,
    )
    assert drawer_lower_base_recenter_required(
        object_center=(1.75, -.70, .92), release=(1.925, -.71, .77),
    )
    assert not drawer_lower_base_recenter_required(
        object_center=(1.88, -.70, .92), release=(1.925, -.71, .77),
    )
    assert not drawer_lower_base_recenter_enabled("VLA82-006")
    assert drawer_lower_base_recenter_enabled("VLA82-005")
    assert np.allclose(
        drawer_lower_base_recenter_local_direction(
            object_center=(1.75, -.70, .92), release=(1.925, -.71, .77),
            base_rotation=np.eye(3), magnitude=.3,
        ),
        (.3, -.017142857), atol=1e-6,
    )
    assert drawer_release_front_bias(
        selection_id="VLA82-006", coordinate=-.712, robot_coordinate=-1.25,
    ) == pytest.approx(-.732)
    assert drawer_release_front_bias(
        selection_id="VLA82-005", coordinate=-.712, robot_coordinate=-1.25,
    ) == pytest.approx(-.712)
    assert drawer_release_front_bias(
        selection_id="VLA82-006", coordinate=-.712, robot_coordinate=-1.25,
        active=False,
    ) == pytest.approx(-.712)
    assert np.allclose(
        drawer_post_release_retreat_target(
            eef=(1.93, -.79, .92), object_center=(1.89, -.69, .92),
        ),
        (1.959711, -.864278, .98), atol=1e-6,
    )
    staging, inserted = drawer_front_insertion_targets(
        target_low=(1.7785, -.8569, .701),
        target_high=(2.0715, -.3169, .819),
        robot_base=(1.9, -1.35, 0.),
        world_half_extents=(.02, .13, .03), floor_top_z=.731,
    )
    assert np.allclose(staging, (1.925, -1.0069, .801), atol=1e-6)
    # The long tool's center must reach the center of the physical inner
    # bottom.  A near-lip center leaves its leading end on the drawer handle
    # and lets it fall outside after the gripper opens.
    assert np.allclose(inserted, (1.925, -.5869, .801), atol=1e-6)
    assert not pick_place_recorded_stability_ready(3)
    assert pick_place_recorded_stability_ready(4)


def test_vla004_cabinet_alignment_commands_one_live_local_axis_at_a_time():
    from tools.vla82_full_sim.expert import cabinet_alignment_local_command

    rotation = -np.eye(2)
    assert np.allclose(cabinet_alignment_local_command(axis="x", eef_xy=(.538, -4.109), object_xy=(.226, -4.298), base_rotation=rotation), (.5, 0.))
    assert np.allclose(cabinet_alignment_local_command(axis="y", eef_xy=(.282, -4.109), object_xy=(.226, -4.298), base_rotation=rotation), (0., .5))
    assert np.allclose(cabinet_alignment_local_command(axis="x", eef_xy=(.39, -4.109), object_xy=(.226, -4.298), base_rotation=rotation), (0., 0.))


def test_mobile_base_fixed_fixture_contact_catches_oven_but_not_floor_or_object():
    from tools.vla82_full_sim.expert import is_mobile_base_fixed_fixture_contact

    assert is_mobile_base_fixed_fixture_contact(("mobilebase0_pedestal_feet_col", "oven_left_group_g25"))
    assert not is_mobile_base_fixed_fixture_contact(("mobilebase0_pedestal_feet_col", "floor_2_room_g0"))
    assert not is_mobile_base_fixed_fixture_contact(("mobilebase0_pedestal_feet_col", "obj_collision"))


def test_pick_place_torso_assist_is_live_bounded_and_stops_after_height_is_reached():
    from tools.vla82_full_sim.expert import pick_place_torso_command

    assert pick_place_torso_command(eef_z=1.52, target_z=1.645, torso_qpos=.07) == pytest.approx(.5)
    assert pick_place_torso_command(eef_z=1.63, target_z=1.645, torso_qpos=.07) == pytest.approx(0.)
    assert pick_place_torso_command(eef_z=1.52, target_z=1.645, torso_qpos=.18) == pytest.approx(0.)


def test_flat_tool_orientation_gate_uses_live_collision_half_extents_only():
    from tools.vla82_full_sim.expert import needs_flat_tool_reorientation
    model = type("Model", (), {
        "geom_name2id": lambda self, name: {"sponge_collision": 0, "can_collision": 1}[name],
        "geom_size": np.array(((0.055, 0.035, 0.018), (0.025, 0.025, 0.085))),
    })()
    raw = type("Raw", (), {"sim": type("Sim", (), {"model": model})()})()
    assert needs_flat_tool_reorientation(raw, ("sponge_collision",)) is True
    assert needs_flat_tool_reorientation(raw, ("can_collision",)) is False


def test_deposit_state_machine_advances_only_after_observed_guards_and_is_bounded():
    """A can is not released until its live XY projection is inside the cavity."""
    from tools.vla82_full_sim.expert import DepositStateMachine

    machine = DepositStateMachine()
    cavity_min = np.array((-.10, -.10, .80))
    cavity_max = np.array((.10, .10, 1.10))
    assert machine.advance(distance=.02, horizontal_distance=.044, grasped=False, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "reach"
    assert machine.advance(distance=.02, horizontal_distance=.016, grasped=False, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "reach"
    assert machine.advance(distance=.02, horizontal_distance=.007, grasped=False, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "grasp"
    machine = DepositStateMachine()
    assert machine.advance(distance=.20, grasped=False, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "reach"
    assert machine.advance(distance=.02, grasped=False, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "grasp"
    assert machine.advance(distance=.02, grasped=True, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "lift"
    # A transport phase may never continue with a lost pinch.
    assert machine.advance(distance=.02, grasped=False, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "reach"
    assert machine.advance(distance=.02, grasped=False, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "grasp"
    assert machine.advance(distance=.02, grasped=True, can_position=np.array((.20, .0, .85)), cavity_min=cavity_min, cavity_max=cavity_max) == "lift"
    assert machine.advance(distance=.02, grasped=True, can_position=np.array((.20, .0, .92)), cavity_min=cavity_min, cavity_max=cavity_max) == "above_cavity"
    # Deposit releases above the opening, not by driving the can through the
    # cavity.  It must be centered and its bottom 2--8cm over the rim.
    assert machine.advance(distance=.02, grasped=True, can_position=np.array((.08, .0, 1.04)), cavity_min=cavity_min, cavity_max=cavity_max, cavity_center_distance=.08, object_bottom_offset=-.055) == "above_cavity"
    assert machine.advance(distance=.02, grasped=True, can_position=np.array((.0, .0, 1.20)), cavity_min=cavity_min, cavity_max=cavity_max, cavity_center_distance=.0, object_bottom_offset=-.055) == "release"
    assert machine.advance(distance=.02, grasped=False, can_position=np.array((.0, .0, .96)), cavity_min=cavity_min, cavity_max=cavity_max) == "settle"
    assert machine.limit_for("reach") == 192
    # A cross-table trash-can transfer measured 0.30 m at ~3 mm/step.  Keep
    # the transport cap finite but large enough to reach the far cavity.
    assert machine.limit_for("above_cavity") == 360
    # The verified world-frame descent advances roughly 1 mm / public step;
    # 144 permits a 14.4 cm drop from the rim while remaining finite.
    assert machine.limit_for("descend") == 144
    assert machine.limit_for("settle") > 0


def test_deposit_vertical_descend_action_is_a_bounded_local_z_command():
    from tools.vla82_full_sim.expert import deposit_vertical_descend_action

    action = deposit_vertical_descend_action(np.full(12, -1.0), np.full(12, 1.0))
    assert action.shape == (12,)
    assert action[2] == pytest.approx(-.70)
    assert action[6] == pytest.approx(0.0)
    assert action[10] == pytest.approx(-1.0)
    assert action[11] == pytest.approx(-1.0)
    assert np.count_nonzero(action[:2]) == 0


def test_deposit_downward_orientation_action_rotates_wrist_without_moving_base_or_torso():
    from tools.vla82_full_sim.expert import ActionLayout, deposit_downward_orientation_action
    layout = ActionLayout(arm=(0, 6), torso=6, base=(7, 10), gripper=10, base_mode=11)
    action = deposit_downward_orientation_action(np.full(12, -1.0), np.full(12, 1.0), layout=layout)
    assert action[5] == pytest.approx(.6)
    assert action[10] == pytest.approx(-1.0)
    assert action[6] == action[7] == action[8] == action[9] == 0.0


def test_torso_probe_action_uses_live_torso_slot_and_keeps_base_stationary():
    from tools.vla82_full_sim.expert import ActionLayout, torso_probe_action

    layout = ActionLayout(arm=(0, 6), torso=6, base=(7, 10), gripper=10, base_mode=11)
    action = torso_probe_action(np.full(12, -1.0), np.full(12, 1.0), layout=layout, command=-1.0)

    assert action[6] == pytest.approx(-1.0)
    assert action[10] == pytest.approx(-1.0)
    assert action[11] == pytest.approx(-1.0)
    assert np.count_nonzero(action[:6]) == 0
    assert np.count_nonzero(action[7:10]) == 0


def test_direct_grasp_control_prevents_arm_z_fallback_from_overwriting_it():
    from tools.vla82_full_sim.expert import should_issue_grasp_vertical_descent

    assert not should_issue_grasp_vertical_descent(direct_action_active=True, contact_hold_active=False, vertical_gap=.095)
    assert should_issue_grasp_vertical_descent(direct_action_active=False, contact_hold_active=False, vertical_gap=.095)


def test_grasp_closure_target_is_at_the_object_pinching_height():
    from tools.vla82_full_sim.expert import grasp_closure_target

    target = grasp_closure_target(np.array((.04, .08, .901)), np.array((.04, .10, .856)))

    np.testing.assert_allclose(target, (.04, .10, .861))


def test_can_grasp_controller_uses_mutually_exclusive_physical_substates():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    assert controller.state == "init_orientation"
    assert controller.observe(orientation_complete=True, horizontal_distance=.03, vertical_gap=.14, raw_grasp=False, two_pad_contact=False, eef_z=1.0) == "xy_center"
    assert controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.14, raw_grasp=False, two_pad_contact=False, eef_z=1.0) == "torso_descend"
    assert controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.874) == "final_xy_refine"
    assert controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.874) == "close"
    # A recenter never restarts orientation.
    assert controller.recenter() == "xy_center"
    assert controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.874) == "torso_descend"
    assert controller.observe(orientation_complete=True, horizontal_distance=.004, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.874) == "final_xy_refine"
    assert controller.observe(orientation_complete=True, horizontal_distance=.004, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.874) == "close"
    for _ in range(2):
        assert controller.observe(orientation_complete=True, horizontal_distance=.004, vertical_gap=.018, raw_grasp=True, two_pad_contact=True, eef_z=.874) == "close"
    assert controller.observe(orientation_complete=True, horizontal_distance=.004, vertical_gap=.018, raw_grasp=True, two_pad_contact=True, eef_z=.874) == "torso_lift"
    assert controller.observe(orientation_complete=True, horizontal_distance=.004, vertical_gap=.018, raw_grasp=True, two_pad_contact=True, eef_z=.925) == "ready_for_deposit"


def test_can_grasp_controller_stops_descent_or_close_on_five_mm_xy_drift():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.observe(orientation_complete=True, horizontal_distance=.02, vertical_gap=.12, raw_grasp=False, two_pad_contact=False, eef_z=1.0)
    controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.12, raw_grasp=False, two_pad_contact=False, eef_z=1.0)
    assert controller.state == "torso_descend"
    assert controller.observe(orientation_complete=True, horizontal_distance=.006, vertical_gap=.10, raw_grasp=False, two_pad_contact=False, eef_z=.98) == "xy_center"


def test_can_grasp_controller_uses_one_measured_torso_pulse_before_recentering():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.observe(orientation_complete=True, horizontal_distance=.02, vertical_gap=.12, raw_grasp=False, two_pad_contact=False, eef_z=1.0)
    controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.12, raw_grasp=False, two_pad_contact=False, eef_z=1.0)
    assert controller.state == "torso_descend"
    controller.begin_torso_pulse(1.0)
    assert controller.observe(orientation_complete=True, horizontal_distance=.004, vertical_gap=.09, raw_grasp=False, two_pad_contact=False, eef_z=.982) == "xy_center"
    assert controller.last_torso_world_dz == pytest.approx(-.018)


def test_can_grasp_controller_allows_six_mm_only_for_pulse_and_requires_four_mm_to_close():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.observe(orientation_complete=True, horizontal_distance=.02, vertical_gap=.12, raw_grasp=False, two_pad_contact=False, eef_z=1.0)
    controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.12, raw_grasp=False, two_pad_contact=False, eef_z=1.0)
    controller.begin_torso_pulse(1.0)
    # At grasp height, 5mm is insufficient for close and must enter final refine.
    assert controller.observe(orientation_complete=True, horizontal_distance=.005, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.98) == "final_xy_refine"
    assert controller.observe(orientation_complete=True, horizontal_distance=.005, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.98) == "final_xy_refine"
    assert controller.observe(orientation_complete=True, horizontal_distance=.003, vertical_gap=.018, raw_grasp=False, two_pad_contact=False, eef_z=.98) == "close"


def test_deposit_grasp_budget_covers_the_bounded_multi_pulse_controller():
    from tools.vla82_full_sim.expert import DepositStateMachine

    assert DepositStateMachine.LIMITS["grasp"] >= 256


def test_deposit_grasp_budget_covers_axis2_saturation_recovery_and_final_refines():
    from tools.vla82_full_sim.expert import DepositStateMachine

    # A public -0.2 axis-2 command moves the pads millimetres, so the fixed
    # scene needs enough bounded steps for side-band descent plus XY repairs.
    assert DepositStateMachine.LIMITS["grasp"] >= 400


def test_axis2_pad_descent_has_a_finite_budget_for_both_asymmetric_pads():
    from tools.vla82_full_sim.expert import CanGraspController

    # The two physical pads can start at visibly different heights; the
    # fallback must remain bounded while leaving room for the higher pad.
    assert CanGraspController.MAX_AXIS2_PULSES >= 128


def test_can_grasp_uses_axis2_pad_descent_after_torso_saturation_until_side_band():
    """A saturated torso must hand off to bounded, observed axis-2 descent."""
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.state = "torso_descend"
    controller.torso_pulses = controller.MAX_TORSO_PULSES
    controller.begin_torso_pulse(1.0)
    assert controller.state == "axis2_pad_descend"
    assert controller.failed is False

    controller.begin_axis2_pulse((.942, .941))
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.003, vertical_gap=.055,
        raw_grasp=False, two_pad_contact=False, eef_z=.999,
        pad_in_side_band=False, pad_z=(.9408, .9398),
    ) == "axis2_pad_descend"
    assert controller.last_axis2_pad_dz < -.0002

    controller.begin_axis2_pulse((.9408, .9398))
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.003, vertical_gap=.019,
        raw_grasp=False, two_pad_contact=False, eef_z=.998,
        pad_in_side_band=True, pad_z=(.9397, .9387),
    ) == "final_xy_refine"


def test_can_grasp_hands_off_to_axis2_when_a_torso_pulse_is_physically_saturated():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.state = "torso_descend"
    controller.begin_torso_pulse(1.0)
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.003, vertical_gap=.055,
        raw_grasp=False, two_pad_contact=False, eef_z=1.0001,
        pad_in_side_band=False, pad_z=(.941, .942),
    ) == "axis2_pad_descend"
    assert controller.failed is False


def test_axis2_pad_descent_tolerates_small_xy_motion_before_final_refine():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.state = "axis2_pad_descend"
    controller.requires_axis2_pad_band = True
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.009, vertical_gap=.045,
        raw_grasp=False, two_pad_contact=False, eef_z=.95,
        pad_in_side_band=False, pad_z=(.94, .96),
    ) == "axis2_pad_descend"
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.013, vertical_gap=.045,
        raw_grasp=False, two_pad_contact=False, eef_z=.95,
        pad_in_side_band=False, pad_z=(.94, .96),
    ) == "final_xy_refine"


def test_low_pad_in_band_high_pad_above_band_enters_level_pads_before_close():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.state = "axis2_pad_descend"
    controller.requires_axis2_pad_band = True
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.004, vertical_gap=.03,
        raw_grasp=False, two_pad_contact=False, eef_z=.90,
        pad_in_side_band=False, pad_z=(.879, .917), pad_level_required=True,
    ) == "level_pads"
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.004, vertical_gap=.03,
        raw_grasp=False, two_pad_contact=False, eef_z=.90,
        pad_in_side_band=False, pad_z=(.890, .895), pad_level_required=False,
    ) == "level_pads"
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.004, vertical_gap=.03,
        raw_grasp=False, two_pad_contact=False, eef_z=.90,
        pad_in_side_band=False, pad_z=(.891, .893), pad_level_required=False,
    ) == "final_xy_refine"


def test_level_pads_action_uses_best_observed_axis4_negative_rotation_only():
    from tools.vla82_full_sim.expert import ActionLayout, level_pads_action

    layout = ActionLayout(arm=(0, 6), torso=6, base=(7, 10), gripper=10, base_mode=11)
    action = level_pads_action(np.full(12, -1.0), np.full(12, 1.0), layout=layout)
    assert action[4] == pytest.approx(-.20)
    assert action[6] == pytest.approx(0.0)
    assert action[10] == pytest.approx(-1.0)
    assert action[11] == pytest.approx(-1.0)
    assert np.count_nonzero(action[[0, 1, 2, 3, 5, 7, 8, 9]]) == 0


def test_level_pads_allows_safe_temporary_xy_drift_while_reducing_pad_difference():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.state = "level_pads"
    controller.requires_axis2_pad_band = True
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.020, vertical_gap=.03,
        raw_grasp=False, two_pad_contact=False, eef_z=.90,
        pad_in_side_band=False, pad_z=(.902, .931), pad_level_required=False,
    ) == "level_pads"


def test_level_pads_stops_after_four_nonimproving_live_difference_observations():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.state = "level_pads"
    controller.requires_axis2_pad_band = True
    for _ in range(5):
        controller.observe(
            orientation_complete=True, horizontal_distance=.004, vertical_gap=.03,
            raw_grasp=False, two_pad_contact=False, eef_z=.90,
            pad_in_side_band=False, pad_z=(.880, .920), pad_level_required=False,
        )
    assert controller.failed is True


def test_axis2_pad_descent_action_is_negative_bounded_and_leaves_torso_stationary():
    from tools.vla82_full_sim.expert import ActionLayout, axis2_pad_descent_action

    layout = ActionLayout(arm=(0, 6), torso=6, base=(7, 10), gripper=10, base_mode=11)
    action = axis2_pad_descent_action(np.full(12, -1.0), np.full(12, 1.0), layout=layout)

    assert action[2] == pytest.approx(-.20)
    assert action[6] == pytest.approx(0.0)
    assert action[10] == pytest.approx(-1.0)
    assert action[11] == pytest.approx(-1.0)
    assert np.count_nonzero(action[[0, 1, 3, 4, 5, 7, 8, 9]]) == 0


def test_vla82_001_fresh_pregrasp_keeps_all_wrist_rotation_axes_neutral():
    from tools.vla82_full_sim.expert import ActionLayout, vla82_001_pregrasp_action

    layout = ActionLayout(arm=(0, 6), torso=6, base=(7, 10), gripper=10, base_mode=11)
    action = vla82_001_pregrasp_action(np.full(12, -1.0), np.full(12, 1.0), layout=layout)
    np.testing.assert_allclose(action[3:6], (0.0, 0.0, 0.0))
    assert action[10] == pytest.approx(-1.0)
    assert action[11] == pytest.approx(-1.0)


def test_level_fresh_side_band_reaches_close_only_after_two_pad_band_observation():
    from tools.vla82_full_sim.expert import CanGraspController

    controller = CanGraspController()
    controller.state = "axis2_pad_descend"
    controller.requires_axis2_pad_band = True
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.003, vertical_gap=.02,
        raw_grasp=False, two_pad_contact=False, eef_z=.90,
        pad_in_side_band=True, pad_z=(.85, .851),
    ) == "final_xy_refine"
    assert controller.observe(
        orientation_complete=True, horizontal_distance=.003, vertical_gap=.02,
        raw_grasp=False, two_pad_contact=False, eef_z=.90,
        pad_in_side_band=True, pad_z=(.85, .851),
    ) == "close"


def test_axis2_pad_recovery_final_xy_refine_never_commands_arm_axis2():
    from tools.vla82_full_sim.expert import can_grasp_xy_feedback_jacobian

    jacobian = np.array(((.10, .01, .40), (.02, .12, -.30)))
    refined = can_grasp_xy_feedback_jacobian(jacobian, requires_axis2_pad_band=True)
    assert refined.shape == (2, 2)
    np.testing.assert_allclose(refined, jacobian[:, :2])


def test_eef_xy_jacobian_uses_symmetric_open_gripper_arm_probes():
    from tools.vla82_full_sim.expert import estimate_eef_xy_jacobian, eef_xy_damped_command

    samples = {
        0: {1.0: np.array((.010, .002)), -1.0: np.array((-.010, -.002))},
        1: {1.0: np.array((-.001, .016)), -1.0: np.array((.001, -.016))},
    }
    jacobian = estimate_eef_xy_jacobian(samples, amplitude=.08)
    np.testing.assert_allclose(jacobian, ((.125, -.0125), (.025, .20)))
    np.testing.assert_allclose(eef_xy_damped_command(jacobian, np.array((.01, -.02)), maximum=.30), (.069087, -.108611), atol=2e-5)


def test_eef_xy_jacobian_can_use_the_third_arm_translation_axis():
    from tools.vla82_full_sim.expert import estimate_eef_xy_jacobian, eef_xy_damped_command

    samples = {
        0: {1.0: np.array((.008, .0)), -1.0: np.array((-.008, .0))},
        1: {1.0: np.array((.0, .008)), -1.0: np.array((.0, -.008))},
        2: {1.0: np.array((.004, -.004)), -1.0: np.array((-.004, .004))},
    }
    jacobian = estimate_eef_xy_jacobian(samples, amplitude=.08)
    assert jacobian.shape == (2, 3)
    command = eef_xy_damped_command(jacobian, np.array((.01, -.01)), maximum=.30)
    assert command.shape == (3,)
    assert np.max(np.abs(command)) <= .30


def test_deposit_base_micro_align_uses_live_base_slots_and_arm_mode():
    from tools.vla82_full_sim.expert import ActionLayout, deposit_base_micro_align_action

    layout = ActionLayout(arm=(0, 6), torso=6, base=(7, 10), gripper=10, base_mode=11)
    action = deposit_base_micro_align_action(
        np.full(12, -1.0), np.full(12, 1.0), layout=layout,
        eef_world=np.array((.0, .0, 1.0)), can_world=np.array((.04, -.03, .86)),
        base_rotation=np.eye(3),
    )
    np.testing.assert_allclose(action[7:9], (-.35, .35))
    assert action[10] == pytest.approx(-1.0)
    assert action[11] == pytest.approx(1.0)


def test_deposit_grasp_verification_requires_raw_grasp_and_two_pad_contacts():
    from tools.vla82_full_sim.expert import deposit_grasp_verified
    assert deposit_grasp_verified(raw_grasp=True, two_pad_contact=True) is True
    assert deposit_grasp_verified(raw_grasp=True, two_pad_contact=False) is False
    assert deposit_grasp_verified(raw_grasp=False, two_pad_contact=True) is False


def test_deposit_descend_target_corrects_live_can_xy_while_descending():
    from tools.vla82_full_sim.expert import deposit_descend_world_target

    target = deposit_descend_world_target(
        eef=np.array((.10, -.08, 1.20)),
        can=np.array((.13, -.12, 1.16)),
        cavity_center=np.array((.08, -.10, .96)),
    )
    np.testing.assert_allclose(target, (.05, -.06, 1.19))


def test_deposit_transport_target_uses_live_can_center_with_a_nonzero_grasp_offset():
    """Transport registration must follow the can, not a stale wrist offset."""
    from tools.vla82_full_sim.expert import deposit_transport_world_target

    # The wrist is deliberately 5cm above the can.  The resulting target
    # preserves that measured grasp transform while moving the *can* centre
    # to the cavity centre and its bottom to a safe 5cm rim clearance.
    target = deposit_transport_world_target(
        eef=np.array((.13, -.12, 1.15)),
        can=np.array((.12, -.10, 1.10)),
        cavity_center=np.array((.02, .03, .96)),
        rim_z=1.10,
        rim_clearance=.05,
        can_bottom_offset=-.055,
    )
    np.testing.assert_allclose(target, (.03, .01, 1.255))


def test_deposit_transport_axis_target_corrects_largest_live_xy_error_only():
    from tools.vla82_full_sim.expert import deposit_transport_axis_target

    # Correct one live horizontal axis only.  Here |dy|>|dx|, and Z remains
    # locked until XY is inside the cavity safety margin.
    lateral = deposit_transport_axis_target(
        eef=np.array((.10, .20, 1.285)), can=np.array((.08, .18, 1.205)),
        cavity_center=np.array((-.04, -.05, .96)),
    )
    np.testing.assert_allclose(lateral, (.10, -.03, 1.285))


def test_deposit_transport_axis_calibration_flips_only_after_two_worsening_steps():
    from tools.vla82_full_sim.expert import update_deposit_axis_calibration

    assert update_deposit_axis_calibration(previous_abs_error=.10, current_abs_error=.102, sign=1.0, worsening_steps=0) == (1.0, 1)
    assert update_deposit_axis_calibration(previous_abs_error=.102, current_abs_error=.105, sign=1.0, worsening_steps=1) == (-1.0, 0)
    assert update_deposit_axis_calibration(previous_abs_error=.105, current_abs_error=.100, sign=-1.0, worsening_steps=1) == (-1.0, 0)


def test_deposit_base_transit_uses_live_base_distance_not_an_unreachable_42cm_arm_gate():
    from tools.vla82_full_sim.expert import deposit_requires_base_transit

    assert deposit_requires_base_transit(horizontal_distance=.08, base_distance=.467) is True
    assert deposit_requires_base_transit(horizontal_distance=.05, base_distance=.467) is False
    assert deposit_requires_base_transit(horizontal_distance=.08, base_distance=.29) is False


def test_base_feedback_command_uses_measured_2d_jacobian_pseudoinverse():
    from tools.vla82_full_sim.expert import base_feedback_command

    # Measured EEF displacement per public base command; the desired EEF
    # delta is (+4cm, -2cm), so the inverse command is (+.2, -.2).
    command = base_feedback_command(np.diag((.20, .10)), np.array((.04, -.02)), maximum=.30)
    np.testing.assert_allclose(command, (.20, -.20))


def test_held_can_damped_jacobian_command_uses_live_response_and_stays_bounded():
    from tools.vla82_full_sim.expert import held_can_damped_command

    jacobian = np.diag((.10, .20, .10))
    command = held_can_damped_command(jacobian, np.array((.01, -.02, .01)), damping=1e-6, maximum=.18)
    np.testing.assert_allclose(command, (.09999, -.099998, .09999), atol=2e-5)
    # A large requested displacement is clipped rather than saturated at the
    # public controller's +/-1 bound.
    assert np.max(np.abs(held_can_damped_command(jacobian, np.ones(3), maximum=.18))) == pytest.approx(.18)


def test_held_can_jacobian_estimate_uses_symmetric_closed_gripper_probes():
    from tools.vla82_full_sim.expert import estimate_held_can_jacobian

    samples = {
        0: [np.array((.008, .001, .0)), np.array((.012, -.001, .0))],
        1: [np.array((.0, .016, .0)), np.array((.0, .024, .0))],
        2: [np.array((.0, .0, .008)), np.array((.0, .0, .012))],
    }
    jacobian = estimate_held_can_jacobian(samples, amplitude=.08)
    np.testing.assert_allclose(jacobian, np.diag((.125, .25, .125)), atol=1e-8)


def test_deposit_opening_release_guard_requires_centered_can_and_safe_rim_clearance():
    from tools.vla82_full_sim.expert import deposit_opening_release_ready

    lower, upper = np.array((-.10, -.10, .81)), np.array((.10, .10, 1.10))
    assert deposit_opening_release_ready(np.array((.0, .0, 1.20)), lower, upper, -.055) is True
    assert deposit_opening_release_ready(np.array((.0, .0, 1.16)), lower, upper, -.055) is False
    assert deposit_opening_release_ready(np.array((.0, .0, 1.26)), lower, upper, -.055) is False
    assert deposit_opening_release_ready(np.array((.095, .0, 1.20)), lower, upper, -.055) is False


def test_deposit_pad_contact_detector_excludes_nonpad_finger_contacts():
    from tools.vla82_full_sim.expert import selected_geom_has_two_finger_pad_contacts
    names = ["can_collision", "gripper0_right_finger1_collision", "gripper0_right_finger2_collision"]
    contact = type("Contact", (), {"geom1": 0, "geom2": 1})()
    raw = type("Raw", (), {"sim": type("Sim", (), {"data": type("Data", (), {"ncon": 1, "contact": [contact]})(), "model": type("Model", (), {"geom_id2name": lambda self, index: names[index]})()})()})()
    assert selected_geom_has_two_finger_pad_contacts(raw, ("can_collision",)) is False
