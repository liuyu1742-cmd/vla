"""Regression for reset-time source-fixture reachability.

The midterm protocol evaluates object manipulation, not long-horizon indoor
navigation.  Every source-labelled pick/place scene must therefore reset the
mobile base in a collision-free pose from which the source object is within
the expert arm's verified base-reach threshold.
"""

from __future__ import annotations

import inspect
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.run_vla82_full_simulation import _load_compiled_specs, _scene_for_spec
from tools.vla82_full_sim.assets import (
    AssetResolutionError,
    _xml_for_asset,
    validate_vla82_037_drawer_asset_xml,
)
from tools.vla82_full_sim.environment import (
    drawer_source_placement_override,
    EnvironmentValidationError,
    gripper_type_for_request,
    make_environment,
    source_drawer_native_anchor_is_reachable,
    source_drawer_robot_anchor,
)
from tools.vla82_full_sim.expert import (
    PickPlaceExpert,
    _raw_environment,
    pick_place_close_transition,
    pick_place_grasp_strategy,
    pick_place_initial_state,
    pick_place_pad_center_target,
    pick_place_retry_grasp_state,
    pick_place_side_entry_point,
    is_gripper_pad_geometry_name,
    make_selected_gripper_contact_detail,
)


class SourceSpawnReachabilityTest(unittest.TestCase):
    def test_vla82_037_source_is_sampled_near_reachable_front_left_drawer_area(self) -> None:
        original = {"pos": (0.0, -0.25), "size": (0.30, 0.25), "rotation": 0.25}

        adjusted = drawer_source_placement_override("VLA82-037", original)

        self.assertEqual(adjusted["pos"], (0.0, 0.40))
        self.assertEqual(adjusted["size"], (0.22, 0.08))
        self.assertEqual(adjusted["offset"], (-0.12, 0.0))
        self.assertEqual(adjusted["rotation"], 0.25)
        self.assertEqual(original["pos"], (0.0, -0.25))
        self.assertIs(drawer_source_placement_override("VLA82-039", original), original)

    def test_vla82_037_can_retain_native_collision_free_anchor_when_reachable(self) -> None:
        self.assertTrue(source_drawer_native_anchor_is_reachable(
            "VLA82-037", native_base=(1.63, -0.76, 0.70), object_center=(1.80, -0.47, 0.75),
        ))
        self.assertTrue(source_drawer_native_anchor_is_reachable(
            "VLA82-037", native_base=(1.626, -0.763, 0.70), object_center=(1.924, -0.467, 0.75),
        ))
        # During RoboCasa reset this pre-settle pose follows the drawer joint
        # another 0.30 m toward the safe base before the first control step.
        self.assertTrue(source_drawer_native_anchor_is_reachable(
            "VLA82-037", native_base=(1.626, -0.763, 0.70), object_center=(1.924, -0.320, 0.75),
        ))
        self.assertFalse(source_drawer_native_anchor_is_reachable(
            "VLA82-037", native_base=(1.63, -0.76, 0.70), object_center=(2.40, 0.20, 0.75),
        ))
        self.assertFalse(source_drawer_native_anchor_is_reachable(
            "VLA82-039", native_base=(1.63, -0.76, 0.70), object_center=(1.80, -0.47, 0.75),
        ))

    def test_cabinet_inside_target_uses_real_bottom_and_first_shelf(self) -> None:
        from tools.vla82_full_sim.environment import cabinet_inside_cavity_geometry

        names = ("cab_bottom", "cab_shelf", "cab_shelf")
        positions = np.asarray(((0.0, 0.0, 1.40), (0.0, 0.0, 1.68), (0.0, 0.0, 1.94)))
        sizes = np.asarray(((0.30, 0.25, 0.02), (0.30, 0.25, 0.02), (0.30, 0.25, 0.02)))
        model = SimpleNamespace(
            ngeom=len(names),
            geom_contype=np.ones(len(names), dtype=int),
            geom_conaffinity=np.ones(len(names), dtype=int),
            geom_size=sizes,
            geom_id2name=lambda index: names[index],
        )
        data = SimpleNamespace(geom_xpos=positions, geom_xmat=np.tile(np.eye(3).reshape(1, 9), (len(names), 1)))
        raw = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))

        supports, lower, upper = cabinet_inside_cavity_geometry(raw, "cab_")

        self.assertEqual(supports, ("cab_bottom",))
        np.testing.assert_allclose(lower, (-.30, -.25, 1.42))
        np.testing.assert_allclose(upper, (.30, .25, 1.66))

    def test_vla82_052_fixed_table_source_skips_mobile_base_reach(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_base_reach_threshold

        self.assertEqual(pick_place_base_reach_threshold("VLA82-052"), .65)

    def test_vla82_052_side_grasp_does_not_close_on_one_finger_edge_contact(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_side_center_ready

        self.assertFalse(pick_place_side_center_ready(
            "VLA82-052", distance=.12, exact_contact=True, two_pad_contact=False,
        ))
        self.assertTrue(pick_place_side_center_ready(
            "VLA82-052", distance=.12, exact_contact=True, two_pad_contact=True,
        ))
        self.assertTrue(pick_place_side_center_ready(
            "VLA82-052", distance=.02, exact_contact=False, two_pad_contact=False,
        ))

    def test_vla82_037_live_top_approach_first_action_is_safe(self) -> None:
        from tools.vla82_full_sim import expert
        from tools.vla82_full_sim.environment import build_physics_capture_contract

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-037")
        scene = _scene_for_spec(spec)
        environment = make_environment(scene, seed=2000)
        try:
            raw = _raw_environment(environment)
            contract = build_physics_capture_contract(environment, scene)
            controller = PickPlaceExpert(spec.phases, contract)
            _, action = next(controller.actions(environment))
            layout = expert.ActionLayout.from_env(environment)
            np.testing.assert_allclose(action[layout.base[0]:layout.base[1]], 0.0)
            self.assertEqual(float(action[layout.base_mode]), -1.0)
            self.assertTrue(controller.trace)
            first = controller.trace[0]
            self.assertEqual(first["state"], "approach")
            self.assertFalse(first["drawer_entry_collision"])
            points = np.asarray(raw.drawer.get_bbox_points(), dtype=float)
            low, high = np.min(points, axis=0), np.max(points, axis=0)
            obj = np.asarray(first["object_world"], dtype=float)
            self.assertTrue(bool(np.all(obj[:2] >= low[:2] - .03)))
            self.assertTrue(bool(np.all(obj[:2] <= high[:2] + .03)))
            self.assertLess(float(action[layout.arm[0] + 2]), 0.0)
        finally:
            environment.close()

    def test_vla82_037_reset_is_arm_reachable_without_base_drawer_contact(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-037")
        environment = make_environment(_scene_for_spec(spec), seed=2000)
        try:
            raw = _raw_environment(environment)
            base, _ = raw.robots[0].composite_controller.part_controllers["base"].get_base_pose()
            object_body = raw.sim.model.body_name2id(raw.objects["obj"].root_body)
            source = np.asarray(raw.sim.data.body_xpos[object_body], dtype=float)
            # A collision scan of the live Omron / open-drawer geometry gives
            # a 0.409 m minimum collision-free centre distance.  Keep a small
            # arm-workspace margin while separately requiring true width-axis
            # alignment, so random lateral avoidance cannot satisfy the gate.
            self.assertLessEqual(float(np.linalg.norm(source[:2] - np.asarray(base)[:2])), 0.43)
            self.assertLessEqual(float(np.linalg.norm(source[:2] - np.asarray(base)[:2])), 0.43)
            self.assertEqual(raw._vla82_robot_spawn_diagnostics["mode"], "source_drawer_native_reachable")
            self.assertFalse(raw._vla82_robot_spawn_diagnostics["task_initially_solved"])
            drawer_prefix = str(raw.drawer.naming_prefix)
            for index in range(int(raw.sim.data.ncon)):
                contact = raw.sim.data.contact[index]
                names = {
                    str(raw.sim.model.geom_id2name(int(contact.geom1)) or ""),
                    str(raw.sim.model.geom_id2name(int(contact.geom2)) or ""),
                }
                self.assertFalse(
                    any(name.startswith("mobilebase0_") for name in names)
                    and any(name.startswith(drawer_prefix) for name in names)
                )
        finally:
            environment.close()

    def test_drawer_source_starts_selected_grasp_without_base_chase(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_initial_state_for_request

        specs = {item.selection_id: item for item in _load_compiled_specs()}
        self.assertEqual(
            pick_place_initial_state_for_request(_scene_for_spec(specs["VLA82-037"])),
            "approach",
        )
        self.assertEqual(
            pick_place_initial_state_for_request(_scene_for_spec(specs["VLA82-005"])),
            "align_yaw",
        )
        self.assertEqual(
            pick_place_initial_state_for_request(_scene_for_spec(specs["VLA82-036"])),
            "cabinet_front_stage",
        )

    def test_vla82_037_selects_reachable_top_grasp_strategy(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-037"))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-037"), "top")
        self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-037"), "approach")
        np.testing.assert_allclose(
            expert.pick_place_close_target(
                "VLA82-037", eef=(1., 2., 3.), tracking_target=(4., 5., 6.),
            ),
            np.array((1., 2., 3.)),
        )
        self.assertEqual(
            expert.pick_place_close_transition_for_selection(
                "VLA82-037", raw_grasp=True, two_pad_contact=False,
                broad_two_finger_contact=True,
            ),
            "secure",
        )
        self.assertEqual(
            expert.pick_place_close_transition_for_selection(
                "VLA82-037", raw_grasp=True, two_pad_contact=True,
                broad_two_finger_contact=True,
            ),
            "secure",
        )
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-038"), "top")
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-020"), "top")
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-021"), "side")

    def test_vla82_020_side_preapproach_accepts_measured_vertical_deadzone(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_side_preapproach_tolerance("VLA82-020"), .065)
        self.assertEqual(expert.pick_place_side_preapproach_tolerance("VLA82-021"), .04)

    def test_vla82_020_counter_source_uses_verified_close_standoff(self) -> None:
        from tools.vla82_full_sim.environment import counter_source_normal_standoff

        self.assertEqual(counter_source_normal_standoff("VLA82-020"), .45)
        self.assertEqual(counter_source_normal_standoff("VLA82-056"), .56)

    def test_vla82_020_grasps_upper_mug_body_instead_of_body_origin(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_high_grasp_target("VLA82-020"))
        np.testing.assert_allclose(
            expert.pick_place_high_grasp_target(
                "VLA82-020", object_center=(.30, -.20, .98),
                object_rotation=np.eye(3), object_half_length=.060,
            ),
            np.array((.30, -.20, 1.025)),
        )

    def test_vla82_056_budget_includes_post_grasp_transport_and_place(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_step_budget

        self.assertEqual(pick_place_step_budget("VLA82-056"), 900)

    def test_vla82_056_transit_raises_soap_to_cabinet_shelf_height(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_height_controlled_transit("VLA82-056"))

    def test_vla82_056_routes_outside_cabinet_before_raising_and_inserting(self) -> None:
        from tools.vla82_full_sim import expert

        kwargs = dict(
            selection_id="VLA82-056",
            release=(.195, -4.186, 1.432),
            target_low=(.030, -4.533, 1.410),
            target_high=(.360, -3.840, 1.677),
            robot_base=(.911, -4.379, .89),
            lift_height=.10,
        )
        outside, cleared, raised, final = expert.cabinet_target_transport_waypoint(
            object_center=(.311, -4.411, .934), cleared=False, raised=False, **kwargs,
        )
        self.assertFalse(cleared)
        self.assertFalse(raised)
        self.assertFalse(final)
        self.assertGreater(outside[0], .40)
        self.assertAlmostEqual(outside[1], -4.411)
        self.assertAlmostEqual(outside[2], .934)

        high, cleared, raised, final = expert.cabinet_target_transport_waypoint(
            object_center=outside, cleared=True, raised=False, **kwargs,
        )
        self.assertTrue(cleared)
        self.assertFalse(raised)
        self.assertFalse(final)
        self.assertAlmostEqual(high[2], 1.532)

        inside, cleared, raised, final = expert.cabinet_target_transport_waypoint(
            object_center=high, cleared=True, raised=True, **kwargs,
        )
        self.assertTrue(cleared)
        self.assertTrue(raised)
        self.assertTrue(final)
        np.testing.assert_allclose(inside, np.array((.195, -4.186, 1.532)))

    def test_vla82_056_soap_transport_is_acceleration_limited(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_transit_limit("VLA82-056"), .35)

    def test_vla82_056_uses_torso_to_reach_high_cabinet_shelf(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(
            expert.pick_place_transit_torso_command(
                "VLA82-056", eef_z=1.10, target_z=1.53, torso_qpos=.19,
            ),
            .5,
        )
        self.assertEqual(
            expert.pick_place_transit_torso_command(
                "VLA82-056", eef_z=1.51, target_z=1.53, torso_qpos=.30,
            ),
            0.0,
        )

    def test_vla82_056_cabinet_raise_locks_end_effector_xy(self) -> None:
        from tools.vla82_full_sim import expert

        desired = expert.cabinet_target_eef_waypoint(
            object_waypoint=(.420, -4.411, 1.532),
            eef=(.455, -4.412, 1.114),
            held_offset=(-.008, -.001, .019),
            raising=True,
        )
        np.testing.assert_allclose(desired, np.array((.455, -4.412, 1.551)))

    def test_drawer_front_orientation_requires_tool_and_opening_alignment(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.drawer_front_orientation_transition(.20, .02), "drawer_side_orient")
        self.assertEqual(expert.drawer_front_orientation_transition(.02, .20), "drawer_side_orient")
        self.assertEqual(expert.drawer_front_orientation_transition(.02, .02), "drawer_front_stage")

    def test_drawer_front_motion_advances_only_after_reach_and_aborts_collision(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(
            expert.drawer_front_motion_transition("drawer_front_stage", .03, False),
            "drawer_front_insert",
        )
        self.assertEqual(
            expert.drawer_front_motion_transition("drawer_front_insert", .03, False),
            "side_center",
        )
        self.assertEqual(
            expert.drawer_front_motion_transition("drawer_front_insert", .10, True),
            "failed",
        )
        self.assertEqual(
            expert.drawer_front_motion_transition("drawer_front_stage", .10, False),
            "drawer_front_stage",
        )

    def test_vla82_037_deep_drawer_stage_allows_full_safe_descent(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertGreaterEqual(expert.drawer_front_waypoint_step_cap("VLA82-037"), 220)
        self.assertGreaterEqual(expert.drawer_front_waypoint_step_cap("VLA82-039"), 300)
        self.assertEqual(expert.drawer_front_waypoint_step_cap("VLA82-005"), 120)

    def test_drawer_front_tool_axis_ignores_vertical_staging_clearance(self) -> None:
        from tools.vla82_full_sim import expert

        axis = expert.drawer_front_target_tool_axis(
            object_center=(1.928, -.584, .743),
            inserted_side_point=(2.0105, -.584, .784),
        )
        np.testing.assert_allclose(axis, np.array((-1., 0., 0.)), atol=1e-9)

    def test_vla82_037_uses_source_drawer_alignment(self) -> None:
        from tools.vla82_full_sim.environment import uses_drawer_source_alignment

        specs = {item.selection_id: item for item in _load_compiled_specs()}
        drawer_source = _scene_for_spec(specs["VLA82-037"])
        counter_to_drawer = _scene_for_spec(specs["VLA82-005"])
        cabinet_source = _scene_for_spec(specs["VLA82-036"])

        self.assertEqual(drawer_source.source_fixture, "drawer")
        self.assertTrue(uses_drawer_source_alignment(drawer_source))
        self.assertFalse(uses_drawer_source_alignment(counter_to_drawer))
        self.assertFalse(uses_drawer_source_alignment(cabinet_source))

    def test_drawer_source_anchor_preserves_slide_standoff_and_aligns_width(self) -> None:
        drawer = np.array((1.925, -0.300, 0.775))
        native = np.array((1.651, -0.943, 0.700))
        obj = np.array((1.928, -0.584, 0.743))
        slide = np.array((0.0, 1.0, 0.0))

        aligned = source_drawer_robot_anchor(
            drawer_center=drawer,
            native_anchor=native,
            object_center=obj,
            slide_axis_world=slide,
        )

        tangent = np.array((1.0, 0.0))
        self.assertAlmostEqual(float(np.dot(aligned[:2] - drawer[:2], slide[:2])),
                               float(np.dot(native[:2] - drawer[:2], slide[:2])))
        self.assertAlmostEqual(float(np.dot(aligned[:2] - obj[:2], tangent)), 0.0)
        self.assertAlmostEqual(float(aligned[2]), float(native[2]))
        self.assertLessEqual(float(np.linalg.norm(aligned[:2] - obj[:2])), 0.39)

    def test_drawer_source_anchor_rejects_zero_horizontal_slide_axis(self) -> None:
        with self.assertRaisesRegex(EnvironmentValidationError, "slide axis"):
            source_drawer_robot_anchor(
                drawer_center=(0.0, 0.0, 0.0),
                native_anchor=(0.0, -0.5, 0.0),
                object_center=(0.1, -0.2, 0.1),
                slide_axis_world=(0.0, 0.0, 1.0),
            )

    def test_drawer_front_side_waypoints_use_live_opening(self) -> None:
        from tools.vla82_full_sim import expert

        stage, inserted, grasp = expert.drawer_front_side_waypoints(
            object_center=(1.928, -.584, .743),
            robot_base=(1.928, -.993, .700),
            drawer_low=(1.7475, -.600, .670),
            drawer_high=(2.1025, 0., .880),
            slide_axis_world=(0., 1., 0.),
            geom_rotation=np.eye(3),
            geom_size=(.0475, .006, .006),
        )

        self.assertLess(stage[1], -.600)
        self.assertAlmostEqual(inserted[1], -.584, places=6)
        self.assertAlmostEqual(stage[0], inserted[0], places=6)
        self.assertLessEqual(stage[2], .860)
        self.assertGreaterEqual(stage[2], .764)
        self.assertAlmostEqual(grasp[0], 1.928, places=6)

    def test_drawer_front_side_stage_keeps_clear_of_counter_edge(self) -> None:
        """The VLA82-037 approach must remain outside the counter collision lip."""
        from tools.vla82_full_sim import expert

        stage, _, _ = expert.drawer_front_side_waypoints(
            object_center=(1.928, -.584, .743),
            robot_base=(1.928, -.993, .700),
            drawer_low=(1.7475, -.600, .670),
            drawer_high=(2.1025, 0., .880),
            slide_axis_world=(0., 1., 0.),
            geom_rotation=np.eye(3),
            geom_size=(.0475, .006, .006),
        )

        # The live counter starts at y=-.650 m and the wrist shell extends
        # beyond its pad centre.  Keep the full hand safely in front before
        # descending below the counter edge into the open drawer.
        self.assertLessEqual(float(stage[1]), -.760)

    def test_drawer_front_side_waypoints_rotate_with_slide_axis(self) -> None:
        from tools.vla82_full_sim import expert

        rotation = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
        stage, inserted, grasp = expert.drawer_front_side_waypoints(
            object_center=(-.584, 1.928, .743),
            robot_base=(-.993, 1.928, .700),
            drawer_low=(-.600, 1.7475, .670),
            drawer_high=(0., 2.1025, .880),
            slide_axis_world=(1., 0., 0.),
            geom_rotation=rotation,
            geom_size=(.0475, .006, .006),
        )

        self.assertLess(stage[0], -.600)
        self.assertAlmostEqual(inserted[0], -.584, places=6)
        self.assertAlmostEqual(stage[1], inserted[1], places=6)
        np.testing.assert_allclose(grasp, np.array((-.584, 1.928, .743)))

    def test_drawer_front_side_waypoints_reject_invalid_geometry(self) -> None:
        from tools.vla82_full_sim import expert

        kwargs = {
            "object_center": (0., 0., .10),
            "robot_base": (0., -.50, 0.),
            "drawer_low": (-.10, -.20, 0.),
            "drawer_high": (.10, .20, .30),
            "geom_rotation": np.eye(3),
            "geom_size": (.0475, .006, .006),
        }
        with self.assertRaisesRegex(EnvironmentValidationError, "slide axis"):
            expert.drawer_front_side_waypoints(
                **kwargs, slide_axis_world=(0., 0., 1.),
            )
        with self.assertRaisesRegex(EnvironmentValidationError, "drawer width"):
            expert.drawer_front_side_waypoints(
                **{**kwargs, "drawer_low": (-.02, -.20, 0.), "drawer_high": (.02, .20, .30)},
                slide_axis_world=(0., 1., 0.),
            )

    def test_vla82_037_tongs_materialize_with_long_axis_horizontal(self) -> None:
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        asset = _scene_for_spec(specs["VLA82-037"]).primary_asset
        root = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))
        bbox = np.fromstring(root.find(".//geom[@name='reg_bbox']").attrib["size"], sep=" ")
        visual = np.fromstring(root.find(".//geom[@name='visual']").attrib["size"], sep=" ")
        collision = np.fromstring(root.find(".//geom[@name='collision']").attrib["size"], sep=" ")

        self.assertIn(int(np.argmax(bbox)), (0, 1))
        self.assertEqual(sorted(bbox.tolist()), sorted(asset.dimensions))
        np.testing.assert_allclose(visual, bbox)
        np.testing.assert_allclose(collision, bbox)

    def test_vla82_056_soap_is_recognizable_and_gripper_compatible(self) -> None:
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        asset = _scene_for_spec(specs["VLA82-056"]).primary_asset
        root = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))

        body = root.find(".//geom[@name='soap_body']")
        collision = root.find(".//geom[@name='collision']")
        grooves = root.findall(".//geom[@name='soap_groove_left']")
        self.assertIsNotNone(body)
        self.assertEqual(body.attrib["type"], "ellipsoid")
        self.assertEqual(len(grooves), 1)
        self.assertIsNotNone(collision)
        self.assertEqual(collision.attrib["type"], "box")
        half_extents = np.fromstring(collision.attrib["size"], sep=" ")
        self.assertLessEqual(float(half_extents[1] * 2.0), .044)
        self.assertLess(float(half_extents[2]), float(half_extents[1]))

    def test_vla82_039_wooden_spoon_materializes_flat_for_its_source_drawer(self) -> None:
        """A 120-mm wooden spoon cannot start vertically inside a shallow drawer."""
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        asset = _scene_for_spec(specs["VLA82-039"]).primary_asset
        root = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))
        bbox = np.fromstring(root.find(".//geom[@name='reg_bbox']").attrib["size"], sep=" ")

        self.assertIn(int(np.argmax(bbox)), (0, 1))
        self.assertLess(float(bbox[2] * 2.0), 0.05)
        self.assertEqual(sorted(bbox.tolist()), sorted(asset.dimensions))

    def test_vla82_041_soup_spoon_materializes_flat_on_its_source_counter(self) -> None:
        """A spoon lying on a counter must not require an in-hand 90-degree rotation."""
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        asset = _scene_for_spec(specs["VLA82-041"]).primary_asset
        root = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))
        bbox = np.fromstring(root.find(".//geom[@name='reg_bbox']").attrib["size"], sep=" ")

        self.assertIn(int(np.argmax(bbox)), (0, 1))
        self.assertLess(float(bbox[2] * 2.0), 0.05)
        self.assertEqual(sorted(bbox.tolist()), sorted(asset.dimensions))

    def test_vla82_041_and_051_assets_have_recognizable_visual_parts(self) -> None:
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        spoon_request = _scene_for_spec(specs["VLA82-041"])
        spoon = _xml_for_asset(spoon_request.primary_asset, "source_texture.png")
        pencil_request = _scene_for_spec(specs["VLA82-051"])
        pencil = _xml_for_asset(pencil_request.object_assets[0], "source_texture.png")
        pencil_case = _xml_for_asset(pencil_request.object_assets[1], "source_texture.png")

        for name in ("spoon_bowl", "spoon_bowl_inner", "spoon_neck", "spoon_handle"):
            self.assertIn(f'name="{name}"', spoon)
        for name in (
            "pencil_body", "pencil_wood", "pencil_graphite",
            "pencil_ferrule", "pencil_eraser",
        ):
            self.assertIn(f'name="{name}"', pencil)
        for name in ("case_lining", "case_rim_front", "case_rim_back", "case_zip_pull"):
            self.assertIn(f'name="{name}"', pencil_case)

        self.assertEqual(spoon.count('class="collision"'), 2)
        self.assertEqual(pencil.count('class="collision"'), 2)
        self.assertEqual(pencil_case.count('class="collision"'), 6)
        self.assertIn('name="legacy_visual_mass_carrier"', spoon)
        self.assertIn('name="legacy_visual_mass_carrier"', pencil)
        self.assertNotIn('name="collision_top"', pencil_case)
        self.assertEqual(
            tuple(float(value) for value in ET.fromstring(pencil).find(".//geom[@name='reg_bbox']").get("size").split()),
            tuple(pencil_request.object_assets[0].dimensions),
        )
        self.assertEqual(
            tuple(float(value) for value in ET.fromstring(pencil_case).find(".//geom[@name='reg_bbox']").get("size").split()),
            tuple(pencil_request.object_assets[1].dimensions),
        )

    def test_vla82_039_uses_a_front_side_drawer_entry(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-039"), "drawer_front_side")
        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-039"))
        self.assertAlmostEqual(
            expert.drawer_over_door_height("VLA82-039", drawer_high_z=.88, nominal_height=.79),
            .96,
        )

    def test_vla82_037_front_stage_keeps_forearm_above_drawer_door(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertAlmostEqual(
            expert.drawer_over_door_height("VLA82-037", drawer_high_z=.88, nominal_height=.79),
            .96,
        )

    def test_vla82_039_spoon_is_reset_along_the_drawer_depth(self) -> None:
        from tools.vla82_full_sim.environment import build_physics_capture_contract

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-039")
        environment = make_environment(_scene_for_spec(spec), seed=2000)
        try:
            raw = _raw_environment(environment)
            geom_name = next(name for name in build_physics_capture_contract(
                environment, _scene_for_spec(spec),
            ).object_geom_names["obj"] if name.endswith("_collision"))
            geom_id = raw.sim.model.geom_name2id(geom_name)
            rotation = np.asarray(raw.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            size = np.asarray(raw.sim.model.geom_size[geom_id], dtype=float)
            long_axis = rotation[:, int(np.argmax(size))]
            self.assertGreater(abs(float(long_axis[1])), .95)
        finally:
            environment.close()

    def test_vla82_037_drawer_preflight_rejects_vertical_tongs(self) -> None:
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        asset = _scene_for_spec(specs["VLA82-037"]).primary_asset
        vertical = _xml_for_asset(asset, "source_texture.png").replace(
            'size="0.095000 0.012000 0.012000"',
            'size="0.012000 0.012000 0.095000"',
        )
        with self.assertRaisesRegex(AssetResolutionError, "horizontal drawer orientation"):
            validate_vla82_037_drawer_asset_xml(asset, vertical)

    def test_vla82_037_drawer_preflight_accepts_generated_asset(self) -> None:
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        asset = _scene_for_spec(specs["VLA82-037"]).primary_asset
        half_extents = validate_vla82_037_drawer_asset_xml(
            asset, _xml_for_asset(asset, "source_texture.png")
        )
        self.assertEqual(half_extents, (0.095, 0.012, 0.012))

    def test_real_video_direction_drives_expressible_scene_routes(self) -> None:
        expected = {
            "VLA82-019": "PickPlaceCounterToCabinet",
            "VLA82-020": "PickPlaceCounterToCabinet",
            "VLA82-043": "PickPlaceCabinetToCounter",
            "VLA82-044": "PickPlaceCounterToCabinet",
        }
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        for selection_id, task_class in expected.items():
            self.assertEqual(_scene_for_spec(specs[selection_id]).task_class, task_class)

    def test_dish_brush_cleaning_uses_a_reachable_counter_work_surface(self) -> None:
        specs = {item.selection_id: item for item in _load_compiled_specs()}

        scene = _scene_for_spec(specs["VLA82-003"])
        self.assertEqual(scene.task_class, "PickPlaceCounterToCabinet")
        self.assertEqual(scene.source_fixture, "counter")

    def test_dish_brush_grasp_target_is_inside_the_upper_handle(self) -> None:
        from tools.vla82_full_sim import expert

        np.testing.assert_allclose(
            expert.cleaning_grasp_target("VLA82-003", (1.0, 2.0, 1.015)),
            (1.0, 2.0, 1.080),
        )
        np.testing.assert_allclose(
            expert.cleaning_grasp_target("VLA82-002", (1.0, 2.0, 1.015)),
            (1.0, 2.0, 1.015),
        )

    def test_dish_brush_uses_the_verified_counter_front_source_sampler(self) -> None:
        from tools.vla82_full_sim import environment

        self.assertTrue(environment.uses_projected_counter_source_placement("VLA82-003"))
        self.assertTrue(environment.uses_projected_counter_source_placement("VLA82-002"))
        self.assertTrue(environment.uses_projected_counter_source_placement("VLA82-040"))
        self.assertFalse(environment.uses_projected_counter_source_placement("VLA82-004"))

    def test_pitcher_uses_a_reachable_upper_body_grasp(self) -> None:
        from tools.vla82_full_sim import environment, expert

        np.testing.assert_allclose(expert.pick_place_grasp_target(
            "VLA82-040",
            object_center=(1.0, 2.0, 1.1),
            object_rotation=np.eye(3),
        ), (1.0, 2.0, 1.155))

    def test_pitcher_asset_has_a_graspable_body_and_visible_handle(self) -> None:
        root = ET.parse(Path("assets/vla82_source_textured/VLA82-040/model.xml")).getroot()
        geoms = {geom.get("name"): geom for geom in root.iter("geom")}

        self.assertEqual(geoms["body_collision"].get("type"), "cylinder")
        self.assertLessEqual(float(geoms["body_collision"].get("size").split()[0]), .035)
        self.assertIn("handle_outer_visual", geoms)

    def test_work_study_desktop_requests_use_an_open_support_target(self) -> None:
        specs = {item.selection_id: item for item in _load_compiled_specs()}

        for selection_id in ("VLA82-046", "VLA82-050", "VLA82-052", "VLA82-053"):
            request = _scene_for_spec(specs[selection_id])
            self.assertEqual(request.task_class, "VLA82WorkStudyOpenSupport")
            self.assertEqual((request.source_fixture, request.target_fixture, request.target_relation), (
                "table", "table", "on",
            ))

    def test_vla82_055_uses_a_fixed_physical_bathroom_shelf_target(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-055")
        request = _scene_for_spec(spec)

        self.assertEqual(request.task_class, "VLA82BathroomShelfPlace")
        self.assertEqual(
            (request.source_fixture, request.target_fixture, request.target_relation),
            ("table", "bathroom_shelf", "on"),
        )
        target_xml = ET.fromstring(
            _xml_for_asset(request.object_assets[1], "source_texture.png")
        )
        target_geoms = {geom.get("name"): geom for geom in target_xml.iter("geom")}
        self.assertTrue({
            "bathroom_shelf_bottom", "bathroom_shelf_back",
            "bathroom_shelf_left_side", "bathroom_shelf_right_side",
            "bathroom_shelf_front_lip", "collision_bottom", "collision_back",
            "collision_left", "collision_right", "collision_front_lip",
        }.issubset(target_geoms))
        self.assertEqual(target_geoms["collision_bottom"].get("type"), "box")
        self.assertEqual(target_geoms["collision_bottom"].get("size"), "0.120000 0.070000 0.015000")
        self.assertEqual(target_geoms["collision_back"].get("size"), "0.120000 0.006 0.030")
        for name in ("collision_back", "collision_left", "collision_right", "collision_front_lip"):
            self.assertEqual(target_geoms[name].get("class"), "collision")

    def test_vla82_055_box_is_narrow_enough_for_a_top_grasp(self) -> None:
        from tools.vla82_full_sim.assets import curated_appearance_texture_path
        from tools.vla82_full_sim.environment import source_object_rotation_override

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-055")
        request = _scene_for_spec(spec)
        root = ET.parse(Path("assets/vla82_source_textured/VLA82-055/model.xml")).getroot()
        collision = tuple(
            float(value)
            for value in root.find(".//geom[@name='collision']").get("size").split()
        )
        carrier_mass = float(root.find(".//geom[@name='legacy_visual_mass_carrier']").get("mass"))

        self.assertEqual(request.primary_asset.dimensions, (.09, .022, .025))
        self.assertEqual(collision, (.09, .022, .025))
        self.assertEqual(carrier_mass, .03)
        self.assertEqual(source_object_rotation_override("VLA82-055", (-.785, .785)), 0.0)
        self.assertEqual(
            curated_appearance_texture_path(request.primary_asset).name,
            "appearance_texture.png",
        )
        generated = ET.fromstring(_xml_for_asset(request.primary_asset, "appearance_texture.png"))
        visual_names = {geom.get("name") for geom in generated.iter("geom")}
        self.assertTrue({
            "hygiene_box_body", "hygiene_front_label", "pad_center",
            "pad_left_wing", "pad_right_wing",
        }.issubset(visual_names))
        target_generated = ET.fromstring(
            _xml_for_asset(request.object_assets[1], "source_texture.png")
        )
        self.assertNotIn(
            "hygiene_box_body",
            {geom.get("name") for geom in target_generated.iter("geom")},
        )
        self.assertEqual(
            tuple(asset.semantic_class for asset in request.object_assets),
            ("卫生巾盒", "卫浴搁板"),
        )
        self.assertTrue(Path(request.object_assets[1].asset_path_or_group).is_file())

    def test_vla82_057_uses_the_same_fixed_bathroom_shelf_primitive(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-057")
        request = _scene_for_spec(spec)

        self.assertEqual(request.task_class, "VLA82BathroomShelfPlace")
        self.assertEqual(
            (request.source_fixture, request.target_fixture, request.target_relation),
            ("table", "bathroom_shelf", "on"),
        )
        self.assertEqual(
            tuple(asset.semantic_class for asset in request.object_assets),
            ("洗涤剂", "卫浴搁板"),
        )

    def test_vla82_055_live_scene_binds_box_and_exact_shelf_support(self) -> None:
        from tools.vla82_full_sim.environment import build_physics_capture_contract

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-055")
        request = _scene_for_spec(spec)
        environment = make_environment(request, seed=2000)
        try:
            raw = _raw_environment(environment)
            self.assertEqual(set(raw.objects), {"obj", "bathroom_shelf"})
            contract = build_physics_capture_contract(environment, request)
            target = next(iter(contract.target_geometries.values()))
            self.assertEqual((target.fixture_id, target.spatial_relation), ("bathroom_shelf", "on"))
            self.assertEqual(len(contract.target_geom_names[target.target_id]), 1)
            self.assertIn("collision_bottom", contract.target_geom_names[target.target_id][0])
            self.assertTrue(all(
                name.startswith(raw.objects["bathroom_shelf"].naming_prefix)
                for name in contract.target_geom_names[target.target_id]
            ))
        finally:
            environment.close()


    def test_vla82_051_requires_a_physical_open_pencil_case_target(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-051")
        request = _scene_for_spec(spec)

        self.assertEqual(request.task_class, "VLA82PencilCaseInsert")
        self.assertEqual(
            (request.source_fixture, request.target_fixture, request.target_relation),
            ("table", "pencil_case", "inside"),
        )
        self.assertEqual(tuple(asset.semantic_class for asset in request.object_assets), ("铅笔", "笔盒"))
        self.assertEqual(request.object_assets[1].geometry, "open_receptacle")
        self.assertGreater(request.object_assets[1].dimensions[1], request.primary_asset.dimensions[2])

    def test_vla82_058_requires_a_physical_open_mouthwash_cup_target(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-058")
        request = _scene_for_spec(spec)

        self.assertEqual(request.task_class, "VLA82ToothbrushCupInsert")
        self.assertEqual(
            (request.source_fixture, request.target_fixture, request.target_relation),
            ("table", "mouthwash_cup", "inside"),
        )
        self.assertEqual(tuple(asset.semantic_class for asset in request.object_assets), ("牙刷", "漱口杯"))
        self.assertEqual(request.object_assets[1].geometry, "open_receptacle")
        self.assertGreater(request.object_assets[1].dimensions[0] - .008, request.primary_asset.dimensions[0])
        from tools.vla82_full_sim.environment import open_receptacle_vertical_protrusion_allowance
        self.assertEqual(open_receptacle_vertical_protrusion_allowance("mouthwash_cup"), .08)

    def test_vla82_058_assets_have_recognizable_visual_parts(self) -> None:
        from tools.vla82_full_sim.assets import _xml_for_asset

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-058")
        request = _scene_for_spec(spec)
        toothbrush_xml = _xml_for_asset(request.object_assets[0], "source_texture.png")
        cup_xml = _xml_for_asset(request.object_assets[1], "source_texture.png")

        for name in (
            "toothbrush_handle_visual",
            "toothbrush_neck_visual",
            "toothbrush_head_visual",
            "toothbrush_bristle_white_0",
            "toothbrush_bristle_blue_0",
        ):
            self.assertIn(f'name="{name}"', toothbrush_xml)
        for name in (
            "cup_wall_visual_0",
            "cup_rim_visual_0",
            "cup_bottom_visual",
        ):
            self.assertIn(f'name="{name}"', cup_xml)
        self.assertEqual(toothbrush_xml.count('class="collision"'), 2)
        self.assertEqual(cup_xml.count('class="collision"'), 6)
        self.assertNotIn("fromto=", toothbrush_xml)
        self.assertNotIn("fromto=", cup_xml)
        for name in (
            "collision_bottom", "collision_front", "collision_back",
            "collision_left", "collision_right",
        ):
            self.assertIn(f'name="{name}"', cup_xml)

    def test_ten_strict_pass_assets_have_recognizable_visual_parts(self) -> None:
        expected = {
            "VLA82-004": ("spray_trigger", "spray_nozzle", "product_label"),
            "VLA82-005": ("whisk_handle", "whisk_wire_0", "whisk_wire_5"),
            "VLA82-014": ("measuring_handle", "measure_mark_25", "measure_mark_75"),
            "VLA82-017": ("bottle_cap", "bottle_shoulder", "grip_ring_2"),
            "VLA82-019": ("can_top_rim", "can_label", "pull_tab"),
            "VLA82-036": ("mug_rim", "mug_inner", "mug_handle_2"),
            "VLA82-038": ("pizza_blade", "blade_axle", "pizza_handle"),
            "VLA82-042": ("brush_grip", "brush_head", "bristle_group_3"),
            "VLA82-045": ("foil_roll", "foil_sheet", "serrated_cutter"),
            "VLA82-060": ("pump_head", "pump_nozzle", "liquid_layer"),
        }
        specs = {item.selection_id: item for item in _load_compiled_specs()}
        for selection_id, names in expected.items():
            request = _scene_for_spec(specs[selection_id])
            xml = _xml_for_asset(request.primary_asset, "source_texture.png")
            self.assertNotIn("fromto=", xml, selection_id)
            visual_default = next(
                default for default in ET.fromstring(xml).iter("default")
                if default.get("class") == "visual"
            )
            self.assertEqual(visual_default.find("geom").get("mass"), "0", selection_id)
            carrier = next(
                geom for geom in ET.fromstring(xml).iter("geom")
                if geom.get("name") == "legacy_visual_mass_carrier"
            )
            self.assertGreater(float(carrier.get("mass")), 0.0, selection_id)
            self.assertEqual(carrier.get("rgba"), "0 0 0 0", selection_id)
            registration = next(
                geom for geom in ET.fromstring(xml).iter("geom")
                if geom.get("name") == "reg_bbox"
            )
            self.assertEqual(
                tuple(float(value) for value in registration.get("size").split()),
                tuple(request.primary_asset.dimensions),
                selection_id,
            )
            self.assertEqual(xml.count('class="collision"'), 2, selection_id)
            for name in names:
                self.assertIn(f'name="{name}"', xml, selection_id)

    def test_vla82_051_live_scene_binds_pencil_and_open_case_cavity(self) -> None:
        from tools.vla82_full_sim.environment import build_physics_capture_contract

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-051")
        request = _scene_for_spec(spec)
        environment = make_environment(request, seed=2000)
        try:
            raw = _raw_environment(environment)
            self.assertEqual(set(raw.objects), {"obj", "pencil_case"})
            contract = build_physics_capture_contract(environment, request)
            self.assertIn("obj", contract.object_geom_names)
            target = next(iter(contract.target_geometries.values()))
            self.assertEqual(target.fixture_id, "pencil_case")
            self.assertEqual(target.spatial_relation, "inside")
            self.assertGreaterEqual(len(next(iter(contract.target_geom_names.values()))), 5)
            pencil = np.asarray(
                raw.sim.data.body_xpos[raw.sim.model.body_name2id(raw.objects["obj"].root_body)],
                dtype=float,
            )
            initially_inside = bool(
                np.all(pencil >= np.asarray(target.min_corner))
                and np.all(pencil <= np.asarray(target.max_corner))
            )
            self.assertFalse(initially_inside)
        finally:
            environment.close()

    def test_vla82_046_open_support_source_starts_in_direct_arm_reach(self) -> None:
        """The folder source must not require the constrained mobile base before grasp."""
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-046")
        environment = make_environment(_scene_for_spec(spec), seed=2000)
        try:
            raw = _raw_environment(environment)
            obj = raw.objects["obj"]
            object_pos = np.asarray(
                raw.sim.data.body_xpos[raw.sim.model.body_name2id(obj.root_body)], dtype=float,
            )
            eef = np.asarray(
                raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=float,
            )
            self.assertLess(float(np.linalg.norm(object_pos[:2] - eef[:2])), .16)
            # The horizontal-side gripper must approach the folder's near
            # long edge; source on the wrist's right makes that edge directly
            # accessible instead of crossing the loose folder.
            self.assertGreater(float(object_pos[0] - eef[0]), .10)
        finally:
            environment.close()

    def test_vla82_046_is_a_recognizable_folder_placed_on_the_desk(self) -> None:
        from tools.vla82_full_sim.environment import work_study_source_ranges
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-046")
        request = _scene_for_spec(spec)
        self.assertEqual(spec.object_name, "文件夹")
        self.assertEqual(spec.operation_text, "摆放至书桌")
        self.assertEqual(
            (request.source_fixture, request.target_fixture, request.target_relation),
            ("table", "table", "on"),
        )
        self.assertLessEqual(request.primary_asset.dimensions[1], .025)
        self.assertLessEqual(request.primary_asset.dimensions[2], .015)
        generated = ET.fromstring(_xml_for_asset(request.primary_asset, "source_texture.png"))
        geoms = {geom.get("name"): geom for geom in generated.iter("geom")}
        self.assertTrue({
            "folder_cover_top", "folder_cover_bottom", "folder_pages",
            "folder_spine", "folder_label", "collision",
        }.issubset(geoms))
        self.assertNotIn("folder_binding", geoms)
        self.assertNotIn("collision_binding", geoms)
        self.assertEqual(geoms["collision"].get("type"), "box")
        from tools.vla82_full_sim import expert

        target = expert.pick_place_grasp_target(
            "VLA82-046", object_center=(0., 0., 0.), object_rotation=np.eye(3),
        )
        self.assertLess(abs(target[0]), .01)
        self.assertLessEqual(abs(target[2]), .01)
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-046"), "top")
        self.assertEqual(expert.pick_place_initial_state("VLA82-046"), "align_yaw")
        _x_range, y_range = work_study_source_ranges("VLA82-046")
        self.assertLessEqual(y_range[1], .12)
        self.assertAlmostEqual(expert.pick_place_release_clearance("VLA82-046"), 0.0)

    def test_vla82_048_stacks_a_recognizable_notebook_on_a_physical_folder(self) -> None:
        from tools.vla82_full_sim import expert
        from tools.vla82_full_sim.environment import work_study_source_ranges

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-048")
        request = _scene_for_spec(spec)
        self.assertEqual(spec.object_name, "笔记本")
        self.assertEqual(spec.operation_text, "叠放于文件夹上")
        self.assertEqual(spec.phases, ("stack",))
        self.assertLessEqual(request.primary_asset.dimensions[2], .015)
        self.assertEqual(request.task_class, "VLA82NotebookStack")
        self.assertEqual(
            (request.source_fixture, request.target_fixture, request.target_relation),
            ("table", "folder", "on"),
        )
        folder = next(item for item in request.object_assets if item.semantic_class == "文件夹")
        notebook_xml = ET.fromstring(_xml_for_asset(request.primary_asset, "source_texture.png"))
        folder_xml = ET.fromstring(_xml_for_asset(folder, "source_texture.png"))
        notebook_geoms = {geom.get("name") for geom in notebook_xml.iter("geom")}
        folder_geoms = {geom.get("name") for geom in folder_xml.iter("geom")}
        self.assertTrue({
            "notebook_cover_top", "notebook_pages", "notebook_spine",
            "notebook_label", "collision",
        }.issubset(notebook_geoms))
        self.assertNotIn("notebook_binding", notebook_geoms)
        self.assertNotIn("collision_binding", notebook_geoms)
        self.assertTrue({
            "target_folder_cover", "target_folder_pages", "target_folder_label", "collision",
        }.issubset(folder_geoms))
        target = expert.pick_place_grasp_target(
            "VLA82-048", object_center=(0., 0., 0.), object_rotation=np.eye(3),
        )
        self.assertLess(abs(target[0]), .01)
        self.assertLessEqual(abs(target[2]), .01)
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-048"), "top")
        self.assertEqual(expert.pick_place_initial_state("VLA82-048"), "align_yaw")
        _x_range, y_range = work_study_source_ranges("VLA82-048")
        self.assertLessEqual(y_range[1], .12)
        self.assertEqual(
            expert.pick_place_execution_phases(("stack",)),
            (("grasp", "place"), "stack"),
        )
        self.assertEqual(expert.pick_place_base_reach_threshold("VLA82-048"), .65)

    def test_vla82_016_places_a_recognizable_book_on_a_physical_nightstand(self) -> None:
        from tools.vla82_full_sim import expert

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-016")
        request = _scene_for_spec(spec)
        self.assertEqual(spec.object_name, "书籍")
        self.assertEqual(spec.operation_text, "从床边取用并放至床头柜")
        self.assertEqual(spec.phases, ("grasp", "place"))
        self.assertEqual(request.task_class, "VLA82BedsideBookPlace")
        self.assertEqual(
            (request.source_fixture, request.target_fixture, request.target_relation),
            ("bed", "nightstand", "on"),
        )
        self.assertLessEqual(request.primary_asset.dimensions[0], .065)
        self.assertLessEqual(request.primary_asset.dimensions[1], .035)
        self.assertLessEqual(request.primary_asset.dimensions[2], .015)
        nightstand = next(item for item in request.object_assets if item.semantic_class == "床头柜")
        bed = next(item for item in request.object_assets if item.semantic_class == "床")
        self.assertEqual(bed.dimensions[2], .360)
        self.assertEqual(nightstand.dimensions[2], .340)
        book_xml = ET.fromstring(_xml_for_asset(request.primary_asset, "source_texture.png"))
        nightstand_xml = ET.fromstring(_xml_for_asset(nightstand, "source_texture.png"))
        bed_xml = ET.fromstring(_xml_for_asset(bed, "source_texture.png"))
        book_geoms = {geom.get("name") for geom in book_xml.iter("geom")}
        nightstand_geoms = {geom.get("name") for geom in nightstand_xml.iter("geom")}
        bed_geoms = {geom.get("name") for geom in bed_xml.iter("geom")}
        self.assertTrue({
            "book_pages", "book_cover_top", "book_cover_bottom", "book_spine",
            "book_title_label", "collision",
        }.issubset(book_geoms))
        self.assertNotIn("book_binding", book_geoms)
        self.assertNotIn("collision_binding", book_geoms)
        self.assertTrue({
            "nightstand_body", "nightstand_top", "nightstand_drawer_front",
            "nightstand_handle", "collision",
        }.issubset(nightstand_geoms))
        self.assertTrue({
            "bed_frame", "bed_mattress", "bed_headboard", "bed_pillow",
            "collision_mattress",
        }.issubset(bed_geoms))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-016"), "top")
        self.assertEqual(expert.pick_place_initial_state("VLA82-016"), "align_yaw")
        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-016"))
        self.assertEqual(expert.pick_place_base_reach_threshold("VLA82-016"), .65)
        self.assertAlmostEqual(expert.pick_place_release_clearance("VLA82-016"), 0.0)
        self.assertAlmostEqual(
            expert.pick_place_required_lift_height(
                source_fixture="bed", source_z=.535,
                support_top=.480, object_half_height=.015,
            ),
            .045,
        )

    def test_work_study_open_support_scene_binds_source_object_and_table(self) -> None:
        from tools.vla82_full_sim.environment import build_physics_capture_contract

        specs = {item.selection_id: item for item in _load_compiled_specs()}
        request = _scene_for_spec(specs["VLA82-052"])
        environment = make_environment(request, seed=2000)
        try:
            self.assertEqual(environment.env.horizon, 1000)
            raw = environment.env
            source = raw.sim.data.body_xpos[
                raw.sim.model.body_name2id(raw.objects["obj"].root_body)
            ]
            base, _ = raw.robots[0].composite_controller.part_controllers["base"].get_base_pose()
            self.assertLessEqual(float(np.linalg.norm(source[:2] - np.asarray(base)[:2])), .32)
            contract = build_physics_capture_contract(environment, request)
            self.assertIn("obj", contract.object_geom_names)
            self.assertIn("open_support_table:tabletop", contract.target_geometries)
        finally:
            environment.close()

    def test_duplicate_visible_place_labels_route_to_one_complete_pick_place_primitive(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_execution_phases

        self.assertEqual(
            pick_place_execution_phases(("place", "place")),
            (("grasp", "place"), "place"),
        )

    def test_thin_open_support_objects_use_geometry_specific_grasp_primitives(self) -> None:
        from tools.vla82_full_sim.expert import (
            pick_place_grasp_strategy,
            pick_place_base_reach_threshold,
            pick_place_initial_state,
            pick_place_post_side_orient_state,
            pick_place_retry_grasp_state,
            pick_place_uses_pad_center_alignment,
        )

        for selection_id in ("VLA82-020",):
            self.assertEqual(pick_place_grasp_strategy(selection_id), "side")
            self.assertTrue(pick_place_uses_pad_center_alignment(selection_id))
        self.assertEqual(pick_place_grasp_strategy("VLA82-052"), "horizontal_side")
        self.assertTrue(pick_place_uses_pad_center_alignment("VLA82-052"))
        self.assertEqual(pick_place_post_side_orient_state("VLA82-052"), "side_preapproach")
        self.assertEqual(pick_place_grasp_strategy("VLA82-055"), "top")
        self.assertEqual(pick_place_initial_state("VLA82-055"), "align_yaw")
        self.assertEqual(pick_place_retry_grasp_state("VLA82-055"), "approach")
        self.assertEqual(pick_place_base_reach_threshold("VLA82-055"), .65)
        self.assertEqual(pick_place_grasp_strategy("VLA82-057"), "top")
        self.assertTrue(pick_place_uses_pad_center_alignment("VLA82-057"))
        self.assertEqual(pick_place_base_reach_threshold("VLA82-057"), .65)
        for selection_id in ("VLA82-050",):
            self.assertEqual(pick_place_grasp_strategy(selection_id), "top")
            self.assertEqual(pick_place_initial_state(selection_id), "align_yaw")
            self.assertEqual(pick_place_retry_grasp_state(selection_id), "approach")
            self.assertTrue(pick_place_uses_pad_center_alignment(selection_id))

    def test_microwave_close_resets_to_a_normal_partially_open_door(self) -> None:
        from tools.vla82_full_sim.environment import build_physics_capture_contract

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-029")
        request = _scene_for_spec(spec)
        environment = make_environment(request, seed=2000)
        try:
            raw = environment.env
            joint_id = build_physics_capture_contract(environment, request).fixture_joint_ids["close"]
            joint_index = raw.sim.model.joint_name2id(joint_id)
            position = float(raw.sim.data.get_joint_qpos(joint_id))
            lower, upper = (float(value) for value in raw.sim.model.jnt_range[joint_index])
            self.assertGreater(abs(position), .08)
            self.assertLess(abs(position), max(abs(lower), abs(upper)) * .50)
        finally:
            environment.close()

    def test_microwave_close_reset_does_not_start_with_the_arm_in_the_door(self) -> None:
        """The handle approach must begin collision-free, not perched on the panel."""
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-029")
        environment = make_environment(_scene_for_spec(spec), seed=2001)
        try:
            raw = environment.env
            for index in range(int(raw.sim.data.ncon)):
                contact = raw.sim.data.contact[index]
                names = (
                    str(raw.sim.model.geom_id2name(contact.geom1) or "").lower(),
                    str(raw.sim.model.geom_id2name(contact.geom2) or "").lower(),
                )
                is_microwave_door = any("microwave_right_group_1" in name for name in names)
                is_arm_link = any("robot0_link" in name or "right_hand" in name for name in names)
                self.assertFalse(
                    is_microwave_door and is_arm_link and float(contact.dist) < -0.001,
                    f"microwave close reset starts in arm/door penetration: {names}, {contact.dist}",
                )
        finally:
            environment.close()

    def test_handle_close_reset_uses_a_narrow_safe_opening_for_cabinet_and_microwave(self) -> None:
        from tools.vla82_full_sim.environment import fixture_close_initial_open_fraction

        self.assertEqual(fixture_close_initial_open_fraction("VLA82-029"), .10)
        self.assertEqual(fixture_close_initial_open_fraction("VLA82-031"), .25)
        self.assertEqual(fixture_close_initial_open_fraction("VLA82-034"), .32)

    def test_oven_door_close_reset_uses_a_physical_hinge_damper(self) -> None:
        from tools.vla82_full_sim.environment import (
            fixture_close_reset_hinge_damping,
            fixture_close_reset_hinge_friction,
        )

        self.assertEqual(fixture_close_reset_hinge_damping("VLA82-008"), 100.0)
        self.assertEqual(fixture_close_reset_hinge_friction("VLA82-008"), 20.0)
        self.assertEqual(fixture_close_reset_hinge_damping("VLA82-031"), 20.0)
        self.assertEqual(fixture_close_reset_hinge_friction("VLA82-031"), 5.0)

    def test_pan_generator_includes_graspable_handle(self) -> None:
        from tools.vla82_full_sim.assets import AssetSpec, _xml_for_asset

        asset = AssetSpec(
            selection_id="VLA82-015", semantic_class="pan", kind="custom_same_class",
            asset_path_or_group="model.xml", exact_class=False,
            collision_validated=False, visible_validated=False,
            affordances=("grasp", "insert"), dimensions=(.105, .105, .035), geometry="pan",
        )
        root = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))
        names = {geom.attrib.get("name") for geom in root.findall(".//geom")}
        self.assertIn("pan_handle_visual", names)
        self.assertIn("pan_handle_collision", names)

    def test_pan_registration_bbox_matches_the_centered_physical_pan(self) -> None:
        from tools.vla82_full_sim.assets import AssetSpec, _xml_for_asset

        asset = AssetSpec(
            selection_id="VLA82-015", semantic_class="pan", kind="custom_same_class",
            asset_path_or_group="model.xml", exact_class=False,
            collision_validated=False, visible_validated=False,
            affordances=("grasp", "insert"), dimensions=(.105, .105, .035), geometry="pan",
        )
        root = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))
        bbox = np.fromstring(root.find(".//geom[@name='reg_bbox']").attrib["size"], sep=" ")
        self.assertTrue(np.allclose(bbox, (.175, .105, .035)))

    def test_pan_counter_reset_uses_its_physical_footprint(self) -> None:
        from tools.vla82_full_sim.environment import native_asset_reset_footprint

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-015")
        self.assertEqual(native_asset_reset_footprint(_scene_for_spec(spec).primary_asset), (.35, .21))

    def test_pan_counter_reset_samples_only_the_verified_safe_center(self) -> None:
        from tools.vla82_full_sim.environment import counter_source_placement_size

        self.assertEqual(counter_source_placement_size("VLA82-015", (.35, .21)), (0.0, 0.0))

    def test_pan_uses_the_fixed_wide_counter_layout_set(self) -> None:
        from tools.vla82_full_sim.environment import robocasa_environment_kwargs

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-015")
        kwargs = robocasa_environment_kwargs(_scene_for_spec(spec), seed=2000, obj_groups="VLA82-015")
        self.assertIsNone(kwargs["split"])
        self.assertEqual(kwargs.get("layout_and_style_ids"), ((11, 34), (15, 34), (18, 34), (40, 34), (50, 34)))

    def test_pan_grasp_target_is_centered_on_handle(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.pick_place_grasp_target(
            "VLA82-015", object_center=(.40, -2.90, .95),
            object_rotation=np.eye(3),
        )
        np.testing.assert_allclose(target, np.array((.505, -2.90, .95)))

    def test_pan_real_video_annotation_maps_counter_to_sink(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-015")
        request = _scene_for_spec(spec)
        self.assertEqual(request.task_class, "PickPlaceCounterToSink")
        self.assertEqual((request.source_fixture, request.target_fixture), ("counter", "sink"))
        from tools.vla82_full_sim.environment import uses_source_fixture_reset_anchor
        self.assertTrue(uses_source_fixture_reset_anchor(request))

    def test_boxed_drink_cabinet_retrieval_uses_source_fixture_anchor(self) -> None:
        from tools.vla82_full_sim.environment import uses_source_fixture_reset_anchor
        from tools.vla82_full_sim.environment import source_object_rotation_override
        from tools.vla82_full_sim.expert import pick_place_grasp_strategy

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-018")
        self.assertEqual(_scene_for_spec(spec).source_fixture, "cabinet")
        self.assertTrue(uses_source_fixture_reset_anchor(_scene_for_spec(spec)))
        self.assertEqual(pick_place_grasp_strategy("VLA82-018"), "top")
        self.assertEqual(source_object_rotation_override("VLA82-018", (-.785, .785)), 0.0)
        self.assertEqual(source_object_rotation_override("VLA82-017", (-.785, .785)), (-.785, .785))

    def test_native_counter_pick_place_recovery_uses_source_fixture_anchor(self) -> None:
        from tools.vla82_full_sim.environment import uses_source_fixture_reset_anchor

        specs = {item.selection_id: item for item in _load_compiled_specs()}
        for selection_id in (
            "VLA82-012", "VLA82-013", "VLA82-020", "VLA82-056", "VLA82-059",
        ):
            request = _scene_for_spec(specs[selection_id])
            self.assertEqual(request.source_fixture, "counter")
            self.assertTrue(uses_source_fixture_reset_anchor(request))

    def test_native_counter_pick_place_recovery_uses_object_aligned_anchor(self) -> None:
        from tools.vla82_full_sim.environment import uses_counter_source_alignment

        specs = {item.selection_id: item for item in _load_compiled_specs()}
        self.assertTrue(uses_counter_source_alignment(_scene_for_spec(specs["VLA82-012"])))
        self.assertTrue(uses_counter_source_alignment(_scene_for_spec(specs["VLA82-020"])))
        self.assertFalse(uses_counter_source_alignment(_scene_for_spec(specs["VLA82-058"])))
        self.assertTrue(uses_counter_source_alignment(_scene_for_spec(specs["VLA82-059"])))
        self.assertFalse(uses_counter_source_alignment(_scene_for_spec(specs["VLA82-037"])))

    def test_vla82_012_native_scene_installs_object_aligned_counter_anchor(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-012")
        environment = make_environment(_scene_for_spec(spec), seed=2000)
        try:
            raw = _raw_environment(environment)
            self.assertEqual(
                raw._vla82_robot_spawn_diagnostics["mode"],
                "source_counter_object_alignment",
            )
            base, _ = raw.robots[0].composite_controller.part_controllers["base"].get_base_pose()
            body_id = raw.sim.model.body_name2id(raw.objects["obj"].root_body)
            source = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
            self.assertLessEqual(float(np.linalg.norm(np.asarray(base)[:2] - source[:2])), .75)
        finally:
            environment.close()

    def test_vla82_012_binds_one_physical_cabinet_shelf_and_its_cavity(self) -> None:
        """A cabinet insertion target is the first usable shelf cavity, not its full shell."""
        from tools.vla82_full_sim.environment import build_physics_capture_contract, make_environment

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-012")
        environment = make_environment(_scene_for_spec(spec), seed=2000)
        try:
            contract = build_physics_capture_contract(environment, _scene_for_spec(spec))
            target = next(iter(contract.target_geometries.values()))
            target_names = next(iter(contract.target_geom_names.values()))
            self.assertEqual(target.spatial_relation, "inside")
            self.assertEqual(len(target_names), 1)
            self.assertTrue(target_names[0].endswith("level1_shelf"))
            # The actual floor is the top of level 1; the next shelf is the
            # ceiling. A bottle centre must be able to live between them.
            self.assertAlmostEqual(float(target.min_corner[2]), 1.7066667, places=4)
            self.assertAlmostEqual(float(target.max_corner[2]), 1.9733333, places=4)
        finally:
            environment.close()


    def test_dish_brush_generator_preserves_recognizable_parts(self) -> None:
        from tools.vla82_full_sim.assets import AssetSpec, _xml_for_asset

        asset = AssetSpec(
            selection_id="VLA82-042",
            semantic_class="dish brush",
            kind="custom_same_class",
            asset_path_or_group="model.xml",
            exact_class=False,
            collision_validated=False,
            visible_validated=False,
            affordances=("grasp",),
            dimensions=(.016, .016, .115),
            geometry="tool",
        )
        generated = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))
        visual_names = {
            geom.attrib.get("name")
            for geom in generated.findall(".//geom")
            if geom.attrib.get("class") == "visual"
        }
        self.assertTrue({"handle_visual", "brush_head_visual", "bristles_visual"} <= visual_names)

    def test_dish_brush_asset_has_recognizable_handle_head_and_bristles(self) -> None:
        model = Path("assets/vla82_source_textured/VLA82-042/model.xml")
        root = ET.parse(model).getroot()
        visual_names = {
            geom.attrib.get("name")
            for geom in root.findall(".//geom")
            if geom.attrib.get("class") == "visual"
        }
        self.assertTrue({"handle_visual", "brush_head_visual", "bristles_visual"} <= visual_names)

    def test_dish_brush_uses_deeper_solid_counter_margin(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_counter_margin("VLA82-042"), .065)
        self.assertEqual(expert.pick_place_counter_margin("VLA82-019"), .015)

    def test_drawer_close_audit_camera_prefers_unoccluded_mirror_side(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("CloseDrawer", "VLA82-030")
        self.assertLess(offsets[0][0], 0.0)

    def test_cleaning_contact_target_reseats_when_surface_contact_opens(self) -> None:
        from tools.vla82_full_sim import expert

        target = getattr(expert, "cleaning_contact_target", None)
        self.assertIsNotNone(target)
        point = np.array((0.50, -2.70, 0.94))
        offset = np.array((0.01, 0.02, 0.08))
        held = target(point, offset, contact_eef_z=1.02, target_contact=True)
        reseat = target(point, offset, contact_eef_z=1.02, target_contact=False)
        self.assertEqual(held[2], 1.02)
        self.assertEqual(reseat[2], 1.0195)

    def test_cabinet_spray_stops_translation_at_first_gripper_contact(self) -> None:
        from tools.vla82_full_sim import expert

        eef = np.array((0.24, -4.30, 1.50))
        obj = np.array((0.22, -4.30, 1.49))
        np.testing.assert_allclose(
            expert.pick_place_close_world_target(
                "VLA82-004", eef=eef, obj=obj, any_gripper_contact=True,
            ),
            eef,
        )

    def test_cabinet_spray_aligns_native_pad_center_before_closing(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-004"))

    def test_cabinet_spray_locks_wrist_only_after_close_contact(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.should_lock_vla004_wrist_for_close(
            "VLA82-004", state="close", any_gripper_contact=True,
        ))
        self.assertFalse(expert.should_lock_vla004_wrist_for_close(
            "VLA82-004", state="close", any_gripper_contact=False,
        ))
        self.assertFalse(expert.should_lock_vla004_wrist_for_close(
            "VLA82-017", state="close", any_gripper_contact=True,
        ))

    def test_verified_close_grasp_promotes_to_secure_lift(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_existing_grasp_transition(
            state="close", raw_grasp=True, two_pad_contact=True,
        ), "secure")

    def test_cabinet_spray_holds_aperture_after_native_grasp(self) -> None:
        from tools.vla82_full_sim import expert

        command = expert.pick_place_bottle_aperture_command(
            "VLA82-004", aperture=.030, previous_aperture=.031, grasped=True,
        )
        self.assertEqual(command, 0.0)

    def test_cabinet_spray_uses_gentle_lift_after_contact_grasp(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertLessEqual(expert.pick_place_initial_lift_limit("VLA82-004"), .25)
        self.assertLessEqual(expert.pick_place_lift_limit("VLA82-004", 5), .25)

    def test_cabinet_spray_keeps_torso_fixed_during_the_initial_pinch_lift(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_lift_torso_command("VLA82-004", grasped=True), 0.0)

    def test_cabinet_spray_transitions_to_extraction_before_grasp_slip(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_lift_proof_height("VLA82-004", .045), .060)
        self.assertEqual(expert.pick_place_lift_proof_height("VLA82-020", .045), .045)

    def test_cabinet_spray_stabilizes_after_extraction_before_transit(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.cabinet_extract_next_state("VLA82-004"), "cabinet_settle")
        self.assertEqual(expert.cabinet_settle_step_requirement("VLA82-004"), 12)
        self.assertEqual(expert.cabinet_extract_next_state("VLA82-017"), "transit")

    def test_cabinet_spray_uses_low_acceleration_transit(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_transit_limit("VLA82-004"), .30)

    def test_cabinet_spray_counter_release_uses_live_rotated_height(self) -> None:
        from tools.vla82_full_sim import expert

        height = expert.pick_place_counter_release_height(
            support_top=.925,
            geom_size=(.03, .08, .025),
            geom_rotation=np.array(((0., 0., 1.), (0., 1., 0.), (1., 0., 0.))),
            clearance=0.0,
        )
        self.assertAlmostEqual(height, .955)

    def test_cabinet_spray_uses_high_grasp_target(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.pick_place_high_grasp_target(
            "VLA82-004",
            object_center=(1., 2., 3.),
            object_rotation=np.eye(3),
            object_half_length=.085,
        )
        np.testing.assert_allclose(target, (1., 2., 3.045))
        np.testing.assert_allclose(
            expert.pick_place_high_grasp_target(
                "VLA82-019",
                object_center=(1., 2., 3.),
                object_rotation=np.eye(3),
                object_half_length=.085,
            ),
            (1., 2., 3.),
        )

    def test_cabinet_spray_high_grasp_is_selection_scoped(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_high_grasp_target("VLA82-004"))
        self.assertFalse(expert.pick_place_uses_high_grasp_target("VLA82-017"))

    def test_cabinet_spray_levels_its_long_axis_before_counter_descent(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_counter_upright_stage_required("VLA82-004"))
        self.assertFalse(expert.pick_place_counter_upright_stage_required("VLA82-019"))
        self.assertEqual(expert.pick_place_post_transit_state("VLA82-004"), "upright_for_counter")
        self.assertEqual(expert.pick_place_post_transit_state("VLA82-019"), "lower")
        self.assertEqual(
            expert.pick_place_post_upright_state("VLA82-004"),
            "counter_lower_base_reposition",
        )
        self.assertAlmostEqual(expert.pick_place_counter_upright_base_reposition_distance("VLA82-004"), .08)
        self.assertEqual(expert.pick_place_counter_upright_base_reposition_direction("VLA82-004"), 1.0)
        self.assertEqual(
            expert.pick_place_post_counter_base_reposition_state("VLA82-004"),
            "upright_recenter_for_counter",
        )
        self.assertEqual(expert.pick_place_lower_limit("VLA82-004"), .45)
        command, error = expert.upright_axis_rotation_command(
            object_axis=(.6, 0., .8), base_rotation=np.eye(3),
        )
        self.assertAlmostEqual(error, np.arccos(.8))
        self.assertLess(command[1], -.2)
        np.testing.assert_allclose(command[[0, 2]], 0.0, atol=1e-9)
        stronger, _ = expert.upright_axis_rotation_command(
            object_axis=(.6, 0., .8), base_rotation=np.eye(3), maximum=.45,
        )
        self.assertLess(stronger[1], -.4)

    def test_cabinet_spray_uses_slow_extended_extraction(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.cabinet_extraction_limit("VLA82-004"), .15)
        self.assertEqual(expert.cabinet_extraction_step_budget("VLA82-004"), 160)

    def test_boxed_drink_uses_low_acceleration_cabinet_extraction(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(
            expert.cabinet_extraction_limit(
                "VLA82-004", actual_selection_id="VLA82-018",
            ),
            .12,
        )

    def test_water_bottle_extraction_budget_reaches_unchanged_strict_distance(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.cabinet_extraction_threshold("VLA82-017"), .22)
        self.assertEqual(
            expert.cabinet_extraction_limit(
                "VLA82-004", actual_selection_id="VLA82-017",
            ),
            .35,
        )
        self.assertGreaterEqual(
            expert.cabinet_extraction_step_budget(
                "VLA82-004", actual_selection_id="VLA82-017",
            ),
            96,
        )

    def test_cabinet_spray_uses_mobile_base_for_the_remaining_counter_carry(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_base_transport_threshold("VLA82-004"), .20)
        self.assertEqual(expert.pick_place_base_transport_threshold("VLA82-017"), .45)
        self.assertFalse(expert.pick_place_base_transport_required_for_selection(
            "VLA82-004", source_fixture="counter", target_fixture="table", horizontal_distance=.24,
        ))

    def test_cabinet_spray_routes_its_twenty_two_cm_y_carry_through_the_base(self) -> None:
        from tools.vla82_full_sim import expert

        tolerance = expert.pick_place_cabinet_arm_finish_y_tolerance("VLA82-004")
        self.assertEqual(tolerance, .05)
        self.assertEqual(expert.pick_place_cabinet_transport_stage(
            base_x=.76, initial_base_x=.76, object_y=-4.298, release_y=-4.078,
            arm_finish_y_tolerance=tolerance,
        ), "clear_x")

    def test_cabinet_spray_keeps_base_transport_latched_until_base_returns(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_base_transport_latched(
            "VLA82-004", horizontal_distance=.17, initial_base_x=.76, base_x=1.02,
        ))
        self.assertFalse(expert.pick_place_base_transport_latched(
            "VLA82-004", horizontal_distance=.17, initial_base_x=.76, base_x=.80,
        ))

    def test_cabinet_spray_limits_mobile_base_carry_speed(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_base_transport_magnitude("VLA82-004"), .20)
        self.assertEqual(expert.pick_place_base_transport_magnitude("VLA82-017"), .60)

    def test_cabinet_spray_has_budget_for_its_low_acceleration_base_route(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_step_budget("VLA82-004"), 900)

    def test_trash_deposit_audit_camera_targets_receptacle(self) -> None:
        from tools.vla82_full_sim import environment

        target_id = getattr(environment, "audit_camera_target_object_id", None)
        self.assertIsNotNone(target_id)
        self.assertEqual(target_id("VLA82TrashCanDeposit"), "trash_can")
        self.assertEqual(target_id("PickPlaceCounterToSink"), "obj")

    def test_trash_deposit_audit_camera_uses_wide_view(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = getattr(environment, "audit_camera_offsets", None)
        self.assertIsNotNone(offsets)
        self.assertGreater(float(np.linalg.norm(offsets("VLA82TrashCanDeposit")[0])), 1.0)

    def test_cabinet_cup_audit_camera_uses_unobstructed_wide_view(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("PickPlaceCabinetToCounter", "VLA82-036")
        self.assertGreater(float(np.linalg.norm(offsets[0])), 1.5)

    def test_vla82_041_drawer_transfer_camera_covers_source_and_destination(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("PickPlaceCounterToDrawer", "VLA82-041")
        self.assertGreater(float(np.linalg.norm(offsets[0])), 1.5)
        whisk_offsets = environment.audit_camera_offsets("PickPlaceCounterToDrawer", "VLA82-005")
        self.assertGreater(float(np.linalg.norm(whisk_offsets[0])), 1.5)
        np.testing.assert_allclose(
            environment.audit_camera_focus(
                "VLA82-041",
                target=(2.0, -0.5, 0.9),
                eef=(2.0, -0.5, 1.1),
                destination=(2.0, -0.8, 0.7),
            ),
            (2.0, -0.65, 0.8),
        )
        np.testing.assert_allclose(
            environment.audit_camera_focus(
                "VLA82-005",
                target=(2.0, -0.5, 0.9),
                eef=(2.0, -0.5, 1.1),
                destination=(2.0, -0.8, 0.7),
            ),
            (2.0, -0.65, 0.8),
        )

    def test_cabinet_spray_audit_camera_uses_unobstructed_wide_view(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("PickPlaceCabinetToCounter", "VLA82-004")
        # The cabinet-side camera must stay inside the room shell and look
        # across the open cabinet rather than through its exterior wall.
        self.assertEqual(offsets[0], (-.75, .35, .70))

    def test_dispenser_audit_camera_uses_unobstructed_wide_view(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("PickPlaceCabinetToCounter", "VLA82-060")
        self.assertGreater(float(np.linalg.norm(offsets[0])), 1.5)

    def test_toaster_oven_open_audit_camera_uses_unobstructed_wide_view(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("OpenFixture", "VLA82-035")
        self.assertGreater(float(np.linalg.norm(offsets[0])), 1.5)

    def test_fridge_drawer_close_audit_camera_uses_unobstructed_wide_view(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("CloseDrawer", "VLA82-027")
        self.assertGreater(float(np.linalg.norm(offsets[0])), 1.5)
        self.assertLess(offsets[0][0], 0.0)

    def test_stove_knob_audit_camera_uses_a_high_wide_view(self) -> None:
        from tools.vla82_full_sim import environment

        offsets = environment.audit_camera_offsets("TurnKnob", "VLA82-007")
        self.assertGreater(float(np.linalg.norm(offsets[0])), 1.2)
        self.assertGreater(offsets[0][2], .7)

    def test_video_environment_supports_hd_render_dimensions(self) -> None:
        from tools.vla82_full_sim import environment

        request = type("Request", (), {"selection_id": "VLA82-036", "camera_names": ("robot0_agentview_left",), "primary_asset": type("Asset", (), {"kind": "fixture_part"})()})()
        kwargs = environment.robocasa_environment_kwargs(request, 2000, camera_width=1280, camera_height=720)
        self.assertEqual(kwargs["camera_widths"], 1280)
        self.assertEqual(kwargs["camera_heights"], 720)
        self.assertGreaterEqual(kwargs["horizon"], 200)

    def test_recorded_mujoco_frame_is_converted_to_top_left_image_origin(self) -> None:
        from tools.vla82_full_sim import expert

        bottom_left_origin = np.array(
            [[[1, 2, 3]], [[4, 5, 6]]],
            dtype=np.uint8,
        )
        upright = expert._frame(
            {"robot0_agentview_left_image": bottom_left_origin},
            "robot0_agentview_left",
        )
        np.testing.assert_array_equal(upright[:, 0, :], [[4, 5, 6], [1, 2, 3]])

    def test_obscured_operation_camera_focuses_between_robot_and_target(self) -> None:
        from tools.vla82_full_sim import environment

        focus = getattr(environment, "audit_camera_focus", None)
        self.assertIsNotNone(focus)
        np.testing.assert_allclose(
            focus("VLA82-017", target=(.2, -4.3, 1.5), eef=(.5, -4.3, 1.3)),
            (.35, -4.3, 1.4),
        )

    def test_rack_front_contact_point_selects_face_opposite_push(self) -> None:
        from tools.vla82_full_sim import expert

        contact_point = getattr(expert, "rack_front_contact_point", None)
        self.assertIsNotNone(contact_point)
        point = contact_point(
            center=(2.48, -.34, .99),
            rotation=np.eye(3),
            half_size=(.15, .12, .002),
            push_direction=(0., 1., 0.),
        )
        np.testing.assert_allclose(point, (2.48, -.46, .99), atol=1e-9)

    def test_rack_seat_contact_stays_on_the_exterior_side_of_the_front_face(self) -> None:
        from tools.vla82_full_sim import expert

        point = expert.rack_seat_contact_point(
            face=(2.48, -.46, .99), push_direction=(0., 1., 0.),
        )
        np.testing.assert_allclose(point, (2.48, -.472, .99), atol=1e-9)

    def test_rack_contact_target_converts_pad_point_to_wrist_target(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.rack_wrist_target_for_pad_contact(
            eef=(.50, -.50, 1.05),
            pad_center=(.50, -.46, 1.05),
            desired_pad_center=(.50, -.40, 1.05),
        )
        np.testing.assert_allclose(target, (.50, -.44, 1.05))

    def test_rack_precontact_transition_uses_pad_distance(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.rack_contact_stage(
            joint_closed=False, exact_contact=False, pad_precontact_distance=.010,
        ), "seat_contact")
        self.assertEqual(expert.rack_contact_stage(
            joint_closed=False, exact_contact=False, pad_precontact_distance=.030,
        ), "preapproach")

    def test_rack_contact_stage_keeps_seating_after_precontact_is_reached(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.rack_contact_stage(
            joint_closed=False, exact_contact=False, pad_precontact_distance=.030,
            seat_latched=True,
        ), "seat_contact")

    def test_rack_contact_stage_keeps_pushing_after_a_verified_contact(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.rack_contact_stage(
            joint_closed=False, exact_contact=False, pad_precontact_distance=.030,
            seat_latched=True, push_latched=True,
        ), "push_close")

    def test_short_travel_rack_requires_most_of_its_real_stroke(self) -> None:
        from tools.vla82_full_sim import predicates

        self.assertAlmostEqual(
            predicates.fixture_motion_threshold("push", .08088), .06066, places=5,
        )
        self.assertTrue(predicates.fixture_target_reached("push", (.08088, .00156)))
        self.assertFalse(predicates.fixture_target_reached("push", (.08088, .01230)))

    def test_knob_requires_a_visible_fraction_of_its_physical_turn_range(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertAlmostEqual(expert.knob_required_turn_delta((0.0, 3.665191429)), .366519143, places=7)

    def test_knob_operation_rejects_a_displaced_cookware_object(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertFalse(expert.knob_collateral_is_stable(
            initial_position=(3.09, -2.891, .933), final_position=(3.152, -2.949, .933),
        ))

    def test_knob_cookware_avoidance_points_from_cookware_toward_selected_knob(self) -> None:
        from tools.vla82_full_sim import expert

        vector = expert.knob_cookware_avoidance_vector(
            knob_center=(3.362, -2.660, .930), cookware_center=(3.426, -2.787, 1.028),
        )
        self.assertLess(float(vector[0]), 0.0)
        self.assertGreater(float(vector[1]), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=7)

    def test_knob_completion_retreat_is_upward_and_away_from_cookware(self) -> None:
        from tools.vla82_full_sim import expert

        vector = expert.knob_post_turn_retreat_vector(
            knob_center=(3.362, -2.660, .930), cookware_center=(3.426, -2.787, 1.028),
        )
        self.assertLess(float(vector[0]), 0.0)
        self.assertGreater(float(vector[1]), 0.0)
        self.assertGreater(float(vector[2]), 0.0)

    def test_runtime_rack_spec_uses_specific_real_video_operation(self) -> None:
        from tools import run_vla82_full_simulation as runner
        from tools.vla82_full_sim.annotations import OperationSpec

        raw = OperationSpec(
            selection_id="VLA82-009", task="烹饪与加热辅助", object_name="烤面包机下层烤盘",
            operation_text="柜内→台面放置", source_kind="operation_json", source_path="operation.json",
            phases=("place",), manipulated_objects=("烤面包机下层烤盘",),
            predicate_names=("place_completed",), source_sha256="source", source_table="5-8-2",
            source_differences=(("ledger_operation_label", "将烤箱下层烤盘完全推入"),),
        )
        resolver = getattr(runner, "_runtime_spec_for_execution", None)
        self.assertIsNotNone(resolver)
        effective = resolver(raw)
        self.assertEqual(effective.phases, ("push",))
        self.assertEqual(effective.predicate_names, ("push_completed",))
        self.assertEqual(effective.operation_text, "将烤箱下层烤盘完全推入")

    def test_sink_to_counter_preserves_reward_required_container_only(self) -> None:
        from tools.vla82_full_sim import environment

        required = getattr(environment, "required_task_auxiliary_object_names", None)
        self.assertIsNotNone(required)
        self.assertEqual(required("PickPlaceSinkToCounter"), ("container",))
        self.assertEqual(required("PickPlaceCabinetToCounter"), ())

    def test_cabinet_front_entry_has_convergence_budget(self) -> None:
        self.assertGreaterEqual(PickPlaceExpert.CABINET_ALIGN_STEPS, 64)

    def test_transport_grasp_dropout_requires_three_consecutive_frames(self) -> None:
        from tools.vla82_full_sim import expert

        count = 0
        count = expert.update_transport_lost_grasp_frames(hold_valid=True, count=count)
        count = expert.update_transport_lost_grasp_frames(hold_valid=False, count=count)
        self.assertFalse(expert.transport_hold_recovery_required(count))
        count = expert.update_transport_lost_grasp_frames(hold_valid=True, count=count)
        self.assertEqual(count, 0)
        for expected in (1, 2, 3):
            count = expert.update_transport_lost_grasp_frames(hold_valid=False, count=count)
            self.assertEqual(count, expected)
        self.assertTrue(expert.transport_hold_recovery_required(count))

    def test_mouse_transport_accepts_opposed_finger_shell_contact(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_transport_hold_valid(
            selection_id="VLA82-053", drawer_pick_place=False,
            raw_grasp=False, two_finger_contact=True,
        ))
        self.assertFalse(expert.pick_place_transport_hold_valid(
            selection_id="VLA82-053", drawer_pick_place=False,
            raw_grasp=False, two_finger_contact=False,
        ))
        self.assertFalse(expert.pick_place_transport_hold_valid(
            selection_id="VLA82-036", drawer_pick_place=False,
            raw_grasp=False, two_finger_contact=True,
        ))

    def test_thin_stapler_secure_and_transport_accept_opposed_finger_contact(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-050"), "top")
        self.assertEqual(expert.pick_place_initial_state("VLA82-050"), "align_yaw")
        self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-050"), "approach")
        self.assertEqual(expert.pick_place_release_clearance("VLA82-050"), 0.0)
        self.assertEqual(expert.pick_place_secure_transition(
            "VLA82-050", raw_grasp=False, two_finger_contact=True,
            secure_steps=expert.pick_place_secure_step_requirement("VLA82-050"),
        ), "lift")
        self.assertTrue(expert.pick_place_transport_hold_valid(
            selection_id="VLA82-050", drawer_pick_place=False,
            raw_grasp=False, two_finger_contact=True,
        ))
        self.assertEqual(expert.pick_place_secure_transition(
            "VLA82-036", raw_grasp=False, two_finger_contact=True,
            secure_steps=expert.pick_place_secure_step_requirement("VLA82-036"),
        ), "secure")

    def test_smooth_cup_uses_faster_bounded_transport_and_descent(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_transit_limit("VLA82-036"), 0.60)
        self.assertEqual(expert.pick_place_transit_limit("VLA82-020"), 0.60)
        self.assertTrue(expert.pick_place_uses_direct_transit("VLA82-004"))
        self.assertTrue(expert.pick_place_uses_direct_transit("VLA82-020"))
        self.assertTrue(expert.pick_place_uses_direct_transit("VLA82-036"))
        self.assertFalse(expert.pick_place_uses_direct_transit("VLA82-019"))
        self.assertEqual(expert.pick_place_lower_limit("VLA82-036"), 0.80)
        self.assertEqual(expert.pick_place_lower_limit("VLA82-020"), 1.00)

        self.assertLess(expert.pick_place_transit_limit("VLA82-036"), 0.8)
        self.assertLessEqual(expert.pick_place_lower_limit("VLA82-036"), 0.8)

    def test_surface_release_retreats_until_gripper_contact_clears(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_post_release_retreat_required(
            raw_grasp=False, target_contact=True, exact_gripper_contact=True,
        ))
        self.assertFalse(expert.pick_place_post_release_retreat_required(
            raw_grasp=False, target_contact=True, exact_gripper_contact=False,
        ))

    def test_post_release_retreat_target_is_latched(self) -> None:
        from tools.vla82_full_sim import expert

        first = expert.pick_place_latched_retreat_target(
            previous=None, eef=(0.5, 0.5, 1.0), object_center=(0.5, 0.45, 0.9),
            released_supported=True,
        )
        second = expert.pick_place_latched_retreat_target(
            previous=first, eef=(0.51, 0.52, 1.01), object_center=(0.5, 0.45, 0.9),
            released_supported=True,
        )
        np.testing.assert_allclose(second, first)
        self.assertGreater(float(first[2]), 1.0)

    def test_release_requires_three_consecutive_clear_supported_frames(self) -> None:
        from tools.vla82_full_sim import expert

        count = 0
        for expected in (1, 2):
            count = expert.update_release_clear_frames(
                raw_grasp=False, target_contact=True,
                exact_gripper_contact=False, count=count,
            )
            self.assertEqual(count, expected)
            self.assertFalse(expert.release_clearance_ready(count))
        count = expert.update_release_clear_frames(
            raw_grasp=False, target_contact=True,
            exact_gripper_contact=True, count=count,
        )
        self.assertEqual(count, 0)
        for _ in range(3):
            count = expert.update_release_clear_frames(
                raw_grasp=False, target_contact=True,
                exact_gripper_contact=False, count=count,
            )
        self.assertTrue(expert.release_clearance_ready(count))

    def test_released_object_on_surface_settles_above_target_top(self) -> None:
        from tools.vla82_full_sim import expert

        kwargs = dict(
            raw_grasp=False,
            target_contact=True,
            object_center=(0.5, 0.5, 0.98),
            target_low=(0.0, 0.0, 0.885),
            target_high=(1.0, 1.0, 0.925),
        )
        self.assertEqual(
            expert.pick_place_released_target_transition(**kwargs, spatial_relation="on"),
            "settle",
        )
        self.assertIsNone(
            expert.pick_place_released_target_transition(**kwargs, spatial_relation="inside"),
        )
        self.assertIsNone(
            expert.pick_place_released_target_transition(
                **kwargs, spatial_relation="on", release_started=False,
            ),
        )
        self.assertIsNone(
            expert.pick_place_released_target_transition(
                **kwargs, spatial_relation="on", release_started=True,
                clearance_ready=False,
            ),
        )
        self.assertIsNone(
            expert.pick_place_released_target_transition(
                **kwargs, spatial_relation="on", exact_gripper_contact=True,
            ),
        )

    def test_knob_turn_reseats_without_rotation_after_contact_dropout(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.knob_turn_control_stage(
            exact_contact=False, contact_observed=False, goal_reached=False,
        ), "preapproach")
        self.assertEqual(expert.knob_turn_control_stage(
            exact_contact=True, contact_observed=True, goal_reached=False,
        ), "rotate")
        self.assertEqual(expert.knob_turn_control_stage(
            exact_contact=False, contact_observed=True, goal_reached=False,
        ), "reseat")
        self.assertEqual(expert.knob_turn_control_stage(
            exact_contact=True, contact_observed=True, goal_reached=True,
        ), "stop")

    def test_knob_turn_goal_matches_strict_predicate_and_requires_live_contact(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.knob_turn_should_stop(
            initial_joint=0.0, current_joint=0.20, required_delta=0.20,
            contact_observed=True, exact_contact=True,
        ))
        self.assertFalse(expert.knob_turn_should_stop(
            initial_joint=0.0, current_joint=0.21, required_delta=0.20,
            contact_observed=True, exact_contact=False,
        ))

    def test_knob_controller_has_one_frame_to_record_contacted_goal(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertGreaterEqual(expert.KnobPrimitive.MAX_STEPS, 145)

    def test_knob_reseat_returns_to_measured_contact_pose(self) -> None:
        from tools.vla82_full_sim import expert

        np.testing.assert_allclose(
            expert.knob_reseat_target(
                knob_center=(1.0, 2.0, 3.0),
                measured_contact_offset=(0.02, -0.03, 0.01),
            ),
            (1.02, 1.97, 3.01),
        )

    def test_cabinet_to_counter_uses_front_entry_controller(self) -> None:
        self.assertEqual(pick_place_initial_state("VLA82-017"), "cabinet_front_stage")

    def test_cabinet_to_counter_long_transit_uses_mobile_base(self) -> None:
        from tools.vla82_full_sim import expert

        required = getattr(expert, "pick_place_base_transport_required", None)
        self.assertIsNotNone(required)
        self.assertTrue(required(
            source_fixture="cabinet",
            target_fixture="counter",
            horizontal_distance=1.50,
        ))
        self.assertFalse(required(
            source_fixture="cabinet",
            target_fixture="counter",
            horizontal_distance=.30,
        ))
        self.assertFalse(required(
            source_fixture="counter",
            target_fixture="drawer",
            horizontal_distance=1.50,
        ))

    def test_mobile_base_transport_command_points_from_object_to_release(self) -> None:
        from tools.vla82_full_sim import expert

        command = getattr(expert, "pick_place_base_transport_local_direction", None)
        self.assertIsNotNone(command)
        local = command(
            object_center=np.array((.45, -4.29, 1.50)),
            release=np.array((.32, -2.45, .99)),
            base_rotation=np.eye(3),
            magnitude=.6,
        )
        self.assertLess(float(local[0]), 0.0)
        self.assertGreater(float(local[1]), 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(local))), .6, places=6)

    def test_counter_release_uses_a_real_solid_patch_not_union_bbox_center(self) -> None:
        from tools.vla82_full_sim import expert

        choose = getattr(expert, "solid_counter_release_point", None)
        self.assertIsNotNone(choose)
        # The source is in a sink cut-out.  The union center is therefore not
        # a physical support point; the nearest actual slab begins at x=2.10.
        point = choose(
            patches=(
                ("counter_front_0", (1.36, -4.63, .89), (1.56, -3.80, .92)),
                ("counter_back_0", (2.10, -4.63, .89), (3.16, -3.80, .92)),
            ),
            source=(2.04, -4.21, 1.03),
            robot_base=(1.00, -4.21, 0.00),
            object_half_size_xy=(.03, .04),
            margin=.015,
        )
        self.assertGreaterEqual(float(point[0]), 2.10 + .03 + .015)
        self.assertLessEqual(float(point[0]), 3.16 - .03 - .015)
        self.assertGreaterEqual(float(point[1]), -4.63 + .04 + .015)
        self.assertLessEqual(float(point[1]), -3.80 - .04 - .015)
        self.assertAlmostEqual(float(point[2]), .92, places=6)

    def test_sink_object_is_lifted_above_counter_rim_before_transit(self) -> None:
        from tools.vla82_full_sim import expert

        required = getattr(expert, "pick_place_required_lift_height", None)
        self.assertIsNotNone(required)
        lift = required(
            source_fixture="sink",
            source_z=.776,
            support_top=.920,
            object_half_height=.115,
            nominal=.045,
        )
        self.assertGreaterEqual(lift, .920 + .115 + .03 - .776)
        self.assertEqual(required(
            source_fixture="counter",
            source_z=.900,
            support_top=.920,
            object_half_height=.030,
            nominal=.045,
        ), .045)

    def test_submicron_base_door_contact_is_not_a_material_collision(self) -> None:
        from tools.vla82_full_sim import expert

        material = getattr(expert, "is_material_mobile_base_fixture_contact", None)
        self.assertIsNotNone(material)
        names = ("mobilebase0_pedestal_feet_col", "stack_2_left_group_1_right_door_g1")
        self.assertFalse(material(names, distance=-5.5e-7))
        self.assertTrue(material(names, distance=-.003))
        self.assertFalse(material(("mobilebase0_pedestal_feet_col", "floor_2_room_g0"), distance=-.003))

    def test_cabinet_transport_clears_door_before_aisle_translation(self) -> None:
        from tools.vla82_full_sim import expert

        stage = getattr(expert, "pick_place_cabinet_transport_stage", None)
        self.assertIsNotNone(stage)
        self.assertEqual(stage(
            base_x=.76, initial_base_x=.76, object_y=-4.29, release_y=-2.45,
        ), "clear_x")
        self.assertEqual(stage(
            base_x=1.12, initial_base_x=.76, object_y=-4.29, release_y=-2.45,
        ), "translate_y")
        self.assertEqual(stage(
            base_x=1.12, initial_base_x=.76, object_y=-2.70, release_y=-2.45,
        ), "return_x")
        self.assertEqual(stage(
            base_x=.87, initial_base_x=.76, object_y=-2.70, release_y=-2.45,
        ), "arm_finish")

    def test_cabinet_to_counter_release_uses_nearby_solid_worktop_patch(self) -> None:
        from tools.vla82_full_sim import expert

        point = getattr(expert, "cabinet_to_counter_release_point", None)
        self.assertIsNotNone(point)
        release = point(
            target_low=np.array((-.83, -5.10, .885)),
            target_high=np.array((1.48, .20, .925)),
            source=np.array((.228, -4.294, 1.465)),
            robot_base=np.array((.76, -4.294, .70)),
        )
        self.assertAlmostEqual(float(release[1]), -4.074, places=3)
        self.assertAlmostEqual(float(release[0]), .56, places=3)
        self.assertGreater(float(release[2]), .92)

    def test_water_bottle_release_keeps_reachable_counter_point(self) -> None:
        from tools.vla82_full_sim import expert

        point = expert.pick_place_counter_release_adjustment(
            "VLA82-017", release=np.array((.56, -4.074, .925)),
        )
        self.assertTrue(np.allclose(point, np.array((.56, -4.074, .925))))

    def test_all_authoritative_cabinet_sources_share_front_alignment(self) -> None:
        from tools.vla82_full_sim import environment, expert

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-036")
        request = _scene_for_spec(spec)
        alignment = getattr(environment, "uses_cabinet_source_alignment", None)
        initial_state = getattr(expert, "pick_place_initial_state_for_request", None)
        profile = getattr(expert, "pick_place_controller_profile_id", None)
        self.assertIsNotNone(alignment)
        self.assertIsNotNone(initial_state)
        self.assertIsNotNone(profile)
        self.assertEqual(request.source_fixture, "cabinet")
        self.assertTrue(alignment(request))
        self.assertEqual(initial_state(request), "cabinet_front_stage")
        self.assertEqual(profile(request), "VLA82-004")

    def test_cabinet_to_counter_anchor_aligns_to_source_object(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-017")
        environment = make_environment(_scene_for_spec(spec), seed=2000)
        try:
            environment.reset(seed=2000)
            raw = _raw_environment(environment)
            base, _ = raw.robots[0].composite_controller.part_controllers["base"].get_base_pose()
            object_body = raw.sim.model.body_name2id(raw.objects["obj"].root_body)
            source = np.asarray(raw.sim.data.body_xpos[object_body], dtype=float)
            # The source cabinet opens along world X in the seeded scene; the
            # reset base must align on its world-Y tangent so the wrist can
            # enter through the door rather than drive diagonally into it.
            self.assertLessEqual(abs(float(source[1] - np.asarray(base, dtype=float)[1])), 0.03)
        finally:
            environment.close()

    def test_counter_to_cabinet_source_is_within_base_reach_budget(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-021")
        environment = make_environment(_scene_for_spec(spec), seed=2000)
        try:
            environment.reset(seed=2000)
            raw = _raw_environment(environment)
            base, _ = raw.robots[0].composite_controller.part_controllers["base"].get_base_pose()
            object_body = raw.sim.model.body_name2id(raw.objects["obj"].root_body)
            source = np.asarray(raw.sim.data.body_xpos[object_body], dtype=float)
            self.assertLessEqual(float(np.linalg.norm(source[:2] - np.asarray(base, dtype=float)[:2])), 0.55)
        finally:
            environment.close()

    def test_broad_finger_shell_contact_cannot_start_secure_grasp(self) -> None:
        self.assertEqual(
            pick_place_close_transition(
                raw_grasp=False, two_pad_contact=False, broad_two_finger_contact=True,
            ),
            "close",
        )

    def test_finger_shell_contact_can_start_closing_without_becoming_a_grasp(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_descent_transition
        self.assertEqual(pick_place_descent_transition(distance=.09, any_gripper_contact=True), "close")

    def test_cleaning_descent_stops_at_first_physical_tool_contact(self) -> None:
        from tools.vla82_full_sim import expert

        transition = getattr(expert, "cleaning_descent_transition", None)
        self.assertIsNotNone(transition)
        self.assertEqual(transition(distance=.09, exact_gripper_contact=True), "close")
        self.assertEqual(transition(distance=.09, exact_gripper_contact=False), "descend")

    def test_counter_to_cabinet_allows_verified_arm_reach_threshold(self) -> None:
        from tools.vla82_full_sim.expert import (
            PickPlaceExpert, pick_place_base_reach_step_budget,
            pick_place_base_reach_threshold,
        )
        self.assertEqual(pick_place_base_reach_threshold("VLA82-021"), .35)
        # The counter-side base reaches a real furniture boundary at 0.445 m.
        # Its public arm trajectory is valid from the collision-free 0.45 m
        # standoff; requiring a closer base pose only creates a dead loop.
        self.assertEqual(pick_place_base_reach_threshold("VLA82-012"), .45)
        self.assertEqual(
            pick_place_base_reach_step_budget("VLA82-012"),
            PickPlaceExpert.BASE_REACH_STEPS,
        )
        self.assertEqual(pick_place_base_reach_threshold("VLA82-013"), .31)
        self.assertGreaterEqual(pick_place_base_reach_threshold("VLA82-042"), .65)
        # This open-table source is deliberately reset in direct arm reach.
        # Any mobile-base pulse is counterproductive in ObjectPlay's table
        # frame, so its reach threshold must suppress base navigation.
        self.assertEqual(pick_place_base_reach_threshold("VLA82-046"), .65)
        self.assertGreaterEqual(pick_place_base_reach_threshold("VLA82-055"), .42)
        self.assertGreaterEqual(pick_place_base_reach_threshold("VLA82-056"), .52)
        self.assertGreaterEqual(pick_place_base_reach_threshold("VLA82-037"), .39)
        self.assertGreaterEqual(pick_place_base_reach_threshold("VLA82-015"), .46)
        self.assertGreaterEqual(pick_place_base_reach_threshold("VLA82-052"), .44)
        self.assertEqual(pick_place_base_reach_threshold("VLA82-020"), .515)
        # The bathroom counter has a physical stop at roughly 0.393 m.  The
        # arm can reach the sanitary-box source from a 0.41 m standoff, so the
        # base controller must hand off there instead of pushing indefinitely.
        self.assertEqual(pick_place_base_reach_threshold("VLA82-054"), .41)
        # The desk edge stops the mobile base at 0.349 m while the slim pencil
        # remains in the Panda arm workspace.
        self.assertEqual(pick_place_base_reach_threshold("VLA82-051"), .65)
        self.assertGreaterEqual(pick_place_base_reach_step_budget("VLA82-021"), 220)
        self.assertGreaterEqual(pick_place_base_reach_step_budget("VLA82-020"), 120)

    def test_sink_brush_uses_torso_to_clear_the_rim(self) -> None:
        from tools.vla82_full_sim import expert

        command = getattr(expert, "pick_place_lower_torso_command", None)
        self.assertIsNotNone(command)
        self.assertEqual(command(
            "VLA82-042", eef_z=.92, target_z=1.14, torso_qpos=.00,
        ), .5)
        self.assertEqual(command(
            "VLA82-042", eef_z=1.13, target_z=1.14, torso_qpos=.10,
        ), 0.0)
        self.assertEqual(command(
            "VLA82-020", eef_z=.92, target_z=1.14, torso_qpos=.00,
        ), 0.0)

    def test_wide_tupperware_starts_with_narrow_axis_yaw_alignment(self) -> None:
        self.assertEqual(pick_place_initial_state("VLA82-021"), "reach_base")

    def test_vla82_056_uses_side_entry_after_base_reach(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-056"), "side_orient")
        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-012"), "approach")
        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-056"))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-056"), "side")

    def test_a5_folder_uses_top_grasp_across_its_short_edge(self) -> None:
        from tools.vla82_full_sim import expert

        # A desk-supported folder leaves no clearance for the lower finger of
        # a horizontal side grasp.  The compact A5 prop is instead pinched
        # from above across its gripper-compatible short edge.
        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-046"))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-046"), "top")
        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-046"), "approach")
        self.assertEqual(expert.pick_place_secure_step_requirement("VLA82-046"), 1)
        self.assertGreaterEqual(expert.pick_place_initial_lift_limit("VLA82-046"), .60)
        self.assertEqual(
            expert.pick_place_descent_transition(
                selection_id="VLA82-046", distance=.045, any_gripper_contact=False,
            ),
            "descend",
        )
        self.assertEqual(
            expert.pick_place_descent_transition(
                selection_id="VLA82-046", distance=.015, any_gripper_contact=True,
            ),
            "descend",
        )
        self.assertEqual(
            expert.pick_place_descent_transition(
                selection_id="VLA82-046", distance=.005, any_gripper_contact=True,
            ),
            "close",
        )

    def test_wide_tupperware_uses_native_pad_center_for_descent(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_uses_pad_center_alignment
        self.assertTrue(pick_place_uses_pad_center_alignment("VLA82-021"))

    def test_wide_tupperware_uses_three_finger_side_entry(self) -> None:
        self.assertEqual(pick_place_grasp_strategy("VLA82-021"), "side")

    def test_toothbrush_cup_insert_uses_completion_safe_lower_speed(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertGreaterEqual(expert.pick_place_lower_limit("VLA82-058"), .8)

    def test_toothbrush_cup_insert_releases_at_physical_contact_height(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_release_clearance("VLA82-058"), 0.0)
        self.assertEqual(
            expert.pick_place_inside_support_top(
                "VLA82-058", target_low_z=.816, target_high_z=.928,
            ),
            .816,
        )

    def test_toothbrush_cup_insert_refreshes_the_live_grasp_offset_before_lowering(self) -> None:
        from tools.vla82_full_sim import expert

        refreshes = getattr(expert, "pick_place_refreshes_grasp_offset_before_lower", lambda _id: False)
        self.assertTrue(refreshes("VLA82-058"))

    def test_counter_bottles_use_geometry_matched_native_pad_entry(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-013"))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-013"), "top")
        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-013"), "approach")
        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-059"))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-059"), "horizontal_side")
        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-059"), "side_orient")
        self.assertEqual(expert.pick_place_post_side_orient_state("VLA82-059"), "side_preapproach")

    def test_box_drink_selects_fixed_base_top_extraction(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-018"))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-018"), "top")
        self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-018"), "approach")

    def test_flat_sanitary_box_aligns_the_native_pad_center(self) -> None:
        from tools.vla82_full_sim import expert

        # The 30-mm-tall box sits below the wrist origin.  Wrist-centre
        # alignment closes the fingers in free space above it; the physical
        # pad midpoint must be aligned with the box instead.
        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-054"))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-054"), "horizontal_side")
        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-054"), "side_orient")
        self.assertEqual(expert.pick_place_post_side_orient_state("VLA82-054"), "side_preapproach")

    def test_slim_pencil_uses_gentle_lift_after_verified_two_pad_grasp(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-051"))
        self.assertLessEqual(expert.pick_place_initial_lift_limit("VLA82-051"), .15)
        self.assertLessEqual(expert.pick_place_lift_limit("VLA82-051", 5), .20)

    def test_small_mouse_uses_top_grasp_instead_of_pushing_from_the_side(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-053"), "top")
        self.assertEqual(expert.pick_place_initial_state("VLA82-053"), "align_yaw")
        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-053"), "align_yaw")
        self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-053"), "approach")
        self.assertEqual(expert.pick_place_release_clearance("VLA82-053"), 0.0)

    def test_bar_soap_uses_top_grasp_instead_of_pushing_from_the_side(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-056"), "top")
        self.assertEqual(expert.pick_place_initial_state("VLA82-056"), "align_yaw")
        self.assertEqual(expert.pick_place_post_base_reach_state("VLA82-056"), "align_yaw")
        self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-056"), "approach")
        self.assertEqual(expert.pick_place_secure_transition(
            "VLA82-056", raw_grasp=False, two_finger_contact=True,
            secure_steps=expert.pick_place_secure_step_requirement("VLA82-056"),
        ), "lift")
        self.assertTrue(expert.pick_place_transport_hold_valid(
            selection_id="VLA82-056", drawer_pick_place=False,
            raw_grasp=False, two_finger_contact=True,
        ))
        np.testing.assert_allclose(
            expert.pick_place_secure_hold_target(
                "VLA82-056",
                eef=np.array((1.0, 2.0, 3.0)),
                obj=np.array((0.5, 1.5, 2.5)),
                grasp_offset=np.array((0.1, -0.2, 0.3)),
            ),
            np.array((0.6, 1.3, 2.8)),
        )

    def test_vla82_053_asset_has_a_recognizable_mouse_shell_and_wheel(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-053")
        request = _scene_for_spec(spec)
        asset = request.primary_asset
        geoms = ET.fromstring(_xml_for_asset(asset, "source_texture.png")).findall(".//geom")
        named = {str(geom.get("name")): geom for geom in geoms}

        self.assertEqual(named["visual"].get("type"), "ellipsoid")
        self.assertIn("mouse_wheel", named)
        self.assertEqual(named["collision"].get("type"), "box")

        keyboard = next(item for item in request.object_assets if item.semantic_class == "键盘")
        keyboard_geoms = {
            str(geom.get("name")): geom
            for geom in ET.fromstring(_xml_for_asset(keyboard, "source_texture.png")).findall(".//geom")
        }
        self.assertIn("keyboard_base", keyboard_geoms)
        self.assertTrue(any(name.startswith("keyboard_key_") for name in keyboard_geoms))

    def test_vla82_050_asset_has_a_recognizable_stapler_body(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-050")
        request = _scene_for_spec(spec)
        geoms = ET.fromstring(
            _xml_for_asset(request.primary_asset, "source_texture.png")
        ).findall(".//geom")
        named = {str(geom.get("name")): geom for geom in geoms}

        self.assertIn("stapler_base", named)
        self.assertIn("stapler_upper_arm", named)
        self.assertIn("stapler_hinge", named)
        self.assertIn("stapler_metal_channel", named)
        self.assertEqual(named["collision"].get("type"), "box")

    def test_vla82_053_scene_contains_the_physical_keyboard_landmark(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-053")
        request = _scene_for_spec(spec)
        environment = make_environment(request, seed=2000)
        try:
            raw = environment.env
            self.assertIn("obj", raw.objects)
            self.assertIn("keyboard", raw.objects)
            keyboard = raw.objects["keyboard"]
            keyboard_geoms = tuple(
                str(raw.sim.model.geom_id2name(index) or "")
                for index in range(int(raw.sim.model.ngeom))
                if str(raw.sim.model.geom_id2name(index) or "").startswith(keyboard.naming_prefix)
            )
            self.assertTrue(any("keyboard_base" in name for name in keyboard_geoms))
            self.assertTrue(any("keyboard_key_" in name for name in keyboard_geoms))
        finally:
            environment.close()

    def test_mouse_starts_in_the_proven_continuous_carry_patch(self) -> None:
        from tools.vla82_full_sim import environment

        self.assertEqual(
            environment.work_study_source_ranges("VLA82-053"),
            ((.05, .06), (.035, .045)),
        )
        self.assertEqual(
            environment.work_study_source_ranges("VLA82-050"),
            ((.13, .15), (.10, .12)),
        )

    def test_open_support_lost_hold_opens_gripper_when_object_is_supported_at_target(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_lost_hold_transition(
            drawer_pick_place=False, selection_id="VLA82-053",
            target_contact=True, inside_xy=True, release_pose_ready=False,
        ), "release")
        self.assertEqual(expert.pick_place_lost_hold_transition(
            drawer_pick_place=False, selection_id="VLA82-053",
            target_contact=False, inside_xy=True, release_pose_ready=False,
        ), "close")
        self.assertEqual(expert.pick_place_lost_hold_transition(
            drawer_pick_place=False, selection_id="VLA82-053",
            target_contact=False, inside_xy=True, release_pose_ready=True,
        ), "release")

    def test_pencil_case_route_keeps_the_authoritative_insert_predicate(self) -> None:
        from tools.vla82_full_sim import expert

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-051")
        scene = _scene_for_spec(spec)
        routed = expert.pick_place_evaluation_spec_for_scene(spec, scene)
        self.assertEqual(spec.phases, ("insert",))
        self.assertEqual(routed.phases, ("insert",))
        self.assertEqual(routed.predicate_names, spec.predicate_names)

    def test_pencil_case_release_uses_the_cavity_floor_not_the_rim(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(
            expert.pick_place_inside_support_top(
                "VLA82-051", target_low_z=.816, target_high_z=.878,
            ),
            .816,
        )

    def test_pencil_release_height_ignores_visual_only_geometries(self) -> None:
        from tools.vla82_full_sim import expert

        names = ("pencil_body", "pencil_graphite", "legacy_visual_mass_carrier", "collision")
        self.assertEqual(
            expert.pick_place_physical_geom_names(
                names,
                contypes=(0, 0, 0, 1),
                conaffinities=(0, 0, 0, 1),
            ),
            ("collision",),
        )

    def test_soup_spoon_can_release_after_a_short_bounded_drawer_drop(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_maximum_gravity_drop("VLA82-041"), .05)
        ceiling = expert.drawer_gravity_release_ceiling(
            floor_top_z=.736,
            vertical_half_extent=.028,
            maximum_drop=expert.pick_place_maximum_gravity_drop("VLA82-041"),
        )
        self.assertAlmostEqual(ceiling, .819)

    def test_soup_spoon_release_moves_the_gripper_clear_of_the_drawer_front(self) -> None:
        from tools.vla82_full_sim import expert

        coordinate = expert.drawer_release_front_bias(
            selection_id="VLA82-041",
            coordinate=-.818,
            robot_coordinate=-1.40,
            active=True,
        )
        self.assertAlmostEqual(coordinate, -.748)

    def test_pencil_release_finishes_once_the_wrist_is_visibly_clear(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.pick_place_retreat_complete(
            "VLA82-051",
            eef=(.1439, -.1032, .9190),
            object_center=(.1372, -.1317, .8220),
            retreat_target=(.1438, -.0450, .9160),
        ))

    def test_box_drink_uses_direct_cabinet_front_insertion(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_post_cabinet_stage_state("VLA82-018"), "cabinet_front_insert")
        self.assertEqual(expert.pick_place_post_cabinet_stage_state("VLA82-017"), "cabinet_front_insert")

    def test_box_drink_keeps_the_default_top_grasp_transition(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_post_yaw_alignment_state("VLA82-018"), "approach")
        self.assertEqual(expert.pick_place_post_yaw_alignment_state("VLA82-021"), "side_preapproach")

    def test_box_drink_grasp_target_is_in_upper_body_band(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.pick_place_grasp_target(
            "VLA82-018",
            object_center=np.array((.20, -4.20, 1.49)),
            object_rotation=np.eye(3),
        )
        np.testing.assert_allclose(target, np.array((.20, -4.20, 1.465)), atol=1e-6)

    def test_box_drink_uses_short_cabinet_clearance_lift(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_extraction_lift_height("VLA82-018", nominal=.22), .01)
        self.assertEqual(expert.pick_place_lift_proof_height("VLA82-018", .22), .01)
        actions_source = inspect.getsource(expert.PickPlaceExpert.actions)
        self.assertIn("pick_place_lift_proof_height(\n                    selection_id,", actions_source)
        self.assertIn("pick_place_extraction_lift_height(selection_id,", actions_source)



    def test_box_drink_close_locks_wrist_after_first_side_contact(self) -> None:
        from tools.vla82_full_sim import expert

        eef = np.array((.50, -4.00, .95))
        tracking = np.array((.40, -4.00, .95))
        np.testing.assert_allclose(
            expert.pick_place_close_target("VLA82-018", eef=eef, tracking_target=tracking),
            eef,
        )

    def test_box_drink_secure_gate_requires_native_grasp_and_two_pads(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_close_transition_for_selection(
            "VLA82-018", raw_grasp=True, two_pad_contact=False,
            broad_two_finger_contact=True,
        ), "close")
        self.assertEqual(expert.pick_place_close_transition_for_selection(
            "VLA82-018", raw_grasp=True, two_pad_contact=True,
            broad_two_finger_contact=True,
        ), "secure")
        self.assertEqual(expert.pick_place_close_transition_for_selection(
            "VLA82-017", raw_grasp=True, two_pad_contact=False,
            broad_two_finger_contact=True,
        ), "secure")

    def test_horizontal_side_tool_command_rotates_downward_tool_toward_box_side(self) -> None:
        from tools.vla82_full_sim import expert

        command, error = expert.horizontal_side_grasp_tool_command(
            tool_axis_world=(0., 0., -1.),
            object_center=(0., 0., 0.),
            side_entry=(1., 0., 0.),
            base_rotation=np.eye(3),
        )
        self.assertGreater(error, 1.5)
        self.assertGreater(command[1], .49)

    def test_horizontal_side_orientation_gate_requires_both_axes(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.horizontal_side_orientation_transition(.20, .02), "side_orient")
        self.assertEqual(expert.horizontal_side_orientation_transition(.02, .20), "side_axis_align")
        self.assertEqual(expert.horizontal_side_orientation_transition(.02, .02), "side_descend")

    def test_flat_folder_rejects_palm_only_contact_before_closing(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertFalse(expert.pick_place_side_center_ready(
            "VLA82-046", distance=.08, exact_contact=True, two_pad_contact=False,
        ))
        self.assertTrue(expert.pick_place_side_center_ready(
            "VLA82-046", distance=.08, exact_contact=True, two_pad_contact=True,
        ))

    def test_horizontal_side_roll_aligns_opening_about_tool_axis(self) -> None:
        from tools.vla82_full_sim import expert

        command, error = expert.horizontal_side_grasp_roll_command(
            opening_axis_world=(0., 0., 1.),
            target_opening_axis_world=(0., 1., 0.),
            tool_axis_world=(-1., 0., 0.),
            base_rotation=np.eye(3),
        )
        self.assertGreater(error, 1.5)
        self.assertAlmostEqual(float(np.linalg.norm(command)), .45)
        self.assertLess(command[0], -.44)
        aligned_command, aligned_error = expert.horizontal_side_grasp_roll_command(
            opening_axis_world=(0., 1., 0.),
            target_opening_axis_world=(0., 1., 0.),
            tool_axis_world=(-1., 0., 0.),
            base_rotation=np.eye(3),
        )
        self.assertLess(aligned_error, 1e-6)
        np.testing.assert_allclose(aligned_command, np.zeros(3), atol=1e-6)

    def test_named_pad_opening_axis_preserves_world_vertical_component(self) -> None:
        from tools.vla82_full_sim import expert

        axis = expert.gripper_opening_axis_world_from_named_pads({
            "gripper_f1_pad_collision": np.array((0., 0., -.10)),
            "gripper_f2_pad_collision": np.array((0., 0., -.12)),
            "gripper_finger_middle_pad_collision": np.array((0., 0., .11)),
        })
        np.testing.assert_allclose(axis, np.array((0., 0., 1.)), atol=1e-6)

    def test_horizontal_side_target_opening_axis_uses_box_narrow_horizontal_axis(self) -> None:
        from tools.vla82_full_sim import expert

        axis = expert.horizontal_side_target_opening_axis(
            geom_rotation=np.eye(3),
            geom_size=np.array((.035, .025, .080)),
        )
        np.testing.assert_allclose(axis, np.array((0., 1., 0.)), atol=1e-6)

    def test_cabinet_water_bottle_freezes_wrist_at_first_contact(self) -> None:
        from tools.vla82_full_sim import expert

        eef = np.array((.25, -4.29, 1.51))
        tracked = np.array((.22, -4.29, 1.51))
        self.assertTrue(np.allclose(
            expert.pick_place_close_target(
                "VLA82-017", eef=eef, tracking_target=tracked,
            ),
            eef,
        ))
        self.assertEqual(expert.pick_place_grasp_strategy("VLA82-017"), "top")
        self.assertTrue(np.allclose(
            expert.pick_place_pad_alignment_offset("VLA82-017"),
            np.array((0.0, -0.018, 0.0)),
        ))
        self.assertEqual(pick_place_initial_state("VLA82-021"), "reach_base")

    def test_side_entry_stays_at_object_height_and_approaches_along_long_axis(self) -> None:
        point = pick_place_side_entry_point(
            object_center=np.array((.40, -3.99, .95)),
            eef=np.array((.40, -4.20, 1.10)),
            geom_rotation=np.eye(3),
            geom_size=np.array((.05, .12, .03)),
        )
        self.assertAlmostEqual(float(point[2]), .95)
        self.assertAlmostEqual(float(point[0]), .40)
        self.assertLess(float(point[1]), -3.99)
        self.assertAlmostEqual(abs(float(point[1] + 3.99)), .17)

    def test_wide_gripper_retry_returns_to_side_entry(self) -> None:
        self.assertEqual(pick_place_retry_grasp_state("VLA82-021"), "side_preapproach")
        self.assertEqual(pick_place_retry_grasp_state("VLA82-005"), "approach")

    def test_three_finger_opening_axis_uses_opposing_pad_groups(self) -> None:
        from tools.vla82_full_sim import expert
        function = getattr(expert, "gripper_opening_axis_from_named_pads", None)
        self.assertIsNotNone(function)
        axis = function({
            "gripper_f1_pad_collision": np.array((-1.0, 0.0, 0.0)),
            "gripper_f2_pad_collision": np.array((-1.0, 0.2, 0.0)),
            "gripper_finger_middle_pad_collision": np.array((1.0, 0.1, 0.0)),
        })
        self.assertTrue(np.allclose(axis, np.array((1.0, 0.0))))

    def test_side_contact_locks_wrist_during_close(self) -> None:
        from tools.vla82_full_sim import expert
        function = getattr(expert, "pick_place_close_target", None)
        self.assertIsNotNone(function)
        eef = np.array((.50, -4.00, .95))
        tracking = np.array((.40, -4.00, .95))
        self.assertTrue(np.allclose(function("VLA82-021", eef=eef, tracking_target=tracking), eef))
        self.assertTrue(np.allclose(function("VLA82-005", eef=eef, tracking_target=tracking), tracking))

    def test_wide_container_uses_official_three_finger_gripper(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-021")
        self.assertEqual(gripper_type_for_request(_scene_for_spec(spec)), "RobotiqThreeFingerGripper")

    def test_gripper_pad_detection_supports_official_three_finger_pad_names(self) -> None:
        self.assertTrue(is_gripper_pad_geometry_name("gripper0_right_f1_pad_collision"))
        self.assertTrue(is_gripper_pad_geometry_name("gripper0_right_finger_middle_pad_collision"))
        self.assertFalse(is_gripper_pad_geometry_name("obj_collision"))

    def test_contact_normal_is_oriented_from_gripper_toward_object(self) -> None:
        forward = make_selected_gripper_contact_detail(
            names=("gripper0_right_f1_pad_collision", "obj_collision"),
            normal=(1.0, 0.0, 0.0), distance=-0.001, object_geoms=("obj_collision",),
        )
        reversed_pair = make_selected_gripper_contact_detail(
            names=("obj_collision", "gripper0_right_f1_pad_collision"),
            normal=(1.0, 0.0, 0.0), distance=-0.001, object_geoms=("obj_collision",),
        )
        self.assertEqual(forward["normal_gripper_to_object"], [1.0, 0.0, 0.0])
        self.assertEqual(reversed_pair["normal_gripper_to_object"], [-1.0, -0.0, -0.0])

    def test_three_finger_grasp_lifts_after_two_consecutive_native_grasp_frames(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_gripper_hold_command, pick_place_initial_lift_limit, pick_place_secure_step_requirement, pick_place_secure_transition
        # The close-to-secure transition is the first observed grasp frame;
        # one additional secure frame makes the required consecutive total two.
        self.assertEqual(pick_place_secure_step_requirement("VLA82-021"), 1)
        self.assertGreaterEqual(pick_place_secure_step_requirement("VLA82-005"), 8)
        self.assertEqual(pick_place_secure_transition("VLA82-021", raw_grasp=True, secure_steps=1), "lift")
        self.assertEqual(pick_place_secure_transition("VLA82-021", raw_grasp=False, secure_steps=10), "close")
        self.assertLess(pick_place_initial_lift_limit("VLA82-021"), .70)
        self.assertEqual(pick_place_gripper_hold_command("VLA82-021"), 0.0)
        self.assertEqual(pick_place_gripper_hold_command("VLA82-005"), 1.0)

    def test_lift_target_preserves_live_grasp_xy_instead_of_spawn_xy(self) -> None:
        from tools.vla82_full_sim import expert

        target = getattr(expert, "pick_place_lift_target", None)
        self.assertIsNotNone(target)
        actual = target(
            source=(.23, -4.29, 1.50),
            grasp_source=(.06, -4.34, 1.49),
            grasp_offset=(.02, .01, .06),
            lift_height=.12,
        )
        self.assertTrue(np.allclose(actual, np.array((.08, -4.33, 1.68))))

    def test_cabinet_outward_points_from_object_to_live_robot_base(self) -> None:
        from tools.vla82_full_sim import expert

        outward = expert.cabinet_outward_from_positions(
            object_xy=(.053, -4.340), robot_base_xy=(.788, -4.285),
        )
        self.assertGreater(float(outward[0]), .99)

    def test_tall_water_bottle_brakes_accumulated_gripper_closure(self) -> None:
        from tools.vla82_full_sim import expert

        feedback = getattr(expert, "pick_place_bottle_aperture_command", None)
        self.assertIsNotNone(feedback)
        self.assertEqual(feedback(
            "VLA82-017", aperture=.075, previous_aperture=.076,
        ), 1.0)
        self.assertEqual(feedback(
            "VLA82-017", aperture=.063, previous_aperture=.064,
        ), 1.0)
        self.assertEqual(feedback(
            "VLA82-017", aperture=.060, previous_aperture=.061,
        ), -1.0)
        self.assertEqual(feedback(
            "VLA82-017", aperture=.060, previous_aperture=.060,
        ), 0.0)
        self.assertEqual(feedback(
            "VLA82-005", aperture=.067, previous_aperture=.068,
        ), 1.0)

    def test_cabinet_spray_bottle_holds_verified_grasp_aperture(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_bottle_aperture_command

        self.assertEqual(pick_place_bottle_aperture_command(
            "VLA82-004", aperture=.034, previous_aperture=.035,
        ), 1.0)
        self.assertEqual(pick_place_bottle_aperture_command(
            "VLA82-004", aperture=.030, previous_aperture=.031,
        ), -1.0)
        self.assertEqual(pick_place_bottle_aperture_command(
            "VLA82-004", aperture=.030, previous_aperture=.030,
        ), 0.0)

    def test_cabinet_extraction_progress_uses_current_grasp_baseline(self) -> None:
        from tools.vla82_full_sim import expert

        distance = getattr(expert, "cabinet_extraction_distance", None)
        self.assertIsNotNone(distance)
        self.assertGreater(distance(
            object_xy=(.440, -4.300),
            extraction_start_xy=(.150, -4.307),
            outward=(1.0, 0.0),
        ), .22)

    def test_strict_pick_place_trace_rejects_regrasp_and_missing_controlled_release(self) -> None:
        from tools.vla82_full_sim import expert

        trace = [
            {"state": "close", "raw_grasp": False},
            *({"state": "secure", "raw_grasp": True} for _ in range(4)),
            *({"state": "cabinet_extract", "raw_grasp": False} for _ in range(4)),
            *({"state": "secure", "raw_grasp": True} for _ in range(4)),
            {"state": "cabinet_front_stage", "raw_grasp": False},
        ]

        self.assertEqual(
            expert.pick_place_strict_trace_errors(("place",), trace),
            (
                "pick_place:grasp_reacquired_after_sustained_loss",
                "pick_place:controlled_release_stage_not_observed",
                "pick_place:settle_stage_not_observed",
            ),
        )

    def test_strict_pick_place_trace_accepts_one_continuous_grasp_and_release(self) -> None:
        from tools.vla82_full_sim import expert

        trace = [
            {"state": "close", "raw_grasp": False},
            *({"state": "secure", "raw_grasp": True} for _ in range(4)),
            *({"state": "transit", "raw_grasp": True} for _ in range(5)),
            {"state": "release", "raw_grasp": True},
            {"state": "settle", "raw_grasp": False},
        ]

        self.assertEqual(expert.pick_place_strict_trace_errors(("place",), trace), ())

    def test_strict_pick_place_trace_rejects_gripper_target_penetration(self) -> None:
        from tools.vla82_full_sim import expert

        trace = [
            {"state": "secure", "raw_grasp": True},
            {"state": "transit", "raw_grasp": True, "exact_gripper_target_contact": True},
            {"state": "release", "raw_grasp": True},
            {"state": "settle", "raw_grasp": False},
        ]

        self.assertEqual(
            expert.pick_place_strict_trace_errors(("place",), trace),
            ("pick_place:gripper_target_collision_before_release",),
        )

    def test_strict_pick_place_trace_rejects_a_crooked_final_shelf_pose(self) -> None:
        from tools.vla82_full_sim import expert

        trace = [
            {"state": "secure", "raw_grasp": True},
            {"state": "release", "raw_grasp": True, "support_normal_alignment_deg": 13.2},
            {"state": "settle", "raw_grasp": False, "support_normal_alignment_deg": 13.2},
        ]

        self.assertEqual(
            expert.pick_place_strict_trace_errors(("place",), trace),
            ("pick_place:crooked_release_pose", "pick_place:crooked_final_support_pose"),
        )

    def test_flat_support_alignment_measures_local_top_normal(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertAlmostEqual(expert.flat_support_normal_alignment_degrees(np.eye(3)), 0.0)
        angle = np.deg2rad(13.0)
        rotation = np.array(((1., 0., 0.), (0., np.cos(angle), -np.sin(angle)), (0., np.sin(angle), np.cos(angle))))
        self.assertAlmostEqual(expert.flat_support_normal_alignment_degrees(rotation), 13.0, places=5)
        yaw = np.deg2rad(56.0)
        yawed = np.array(((np.cos(yaw), -np.sin(yaw), 0.), (np.sin(yaw), np.cos(yaw), 0.), (0., 0., 1.)))
        self.assertAlmostEqual(expert.flat_support_yaw_alignment_degrees(yawed), 56.0, places=5)
        command, tilt_error, yaw_error = expert.shelf_package_rotation_command(
            object_rotation=yawed, base_rotation=np.eye(3),
        )
        self.assertAlmostEqual(tilt_error, 0.0, places=5)
        self.assertAlmostEqual(yaw_error, 56.0, places=5)
        self.assertLess(command[2], 0.0)

    def test_strict_pick_place_trace_rejects_a_yawed_shelf_package(self) -> None:
        from tools.vla82_full_sim import expert

        trace = [
            {"state": "secure", "raw_grasp": True},
            {"state": "release", "raw_grasp": True, "support_normal_alignment_deg": 2.0, "support_yaw_alignment_deg": 18.0},
            {"state": "settle", "raw_grasp": False, "support_normal_alignment_deg": 2.0, "support_yaw_alignment_deg": 18.0},
        ]
        self.assertEqual(
            expert.pick_place_strict_trace_errors(("place",), trace),
            ("pick_place:crooked_release_pose", "pick_place:crooked_final_support_pose"),
        )

    def test_vla82_055_levels_the_box_before_shelf_descent(self) -> None:
        from tools.vla82_full_sim import environment, expert

        np.testing.assert_allclose(
            environment.fixed_support_target_position("bathroom_shelf"),
            (.14, -.06, .815),
        )
        self.assertEqual(expert.pick_place_release_clearance("VLA82-055"), 0.0)

        self.assertEqual(expert.pick_place_post_transit_state("VLA82-055"), "level_for_shelf")
        self.assertEqual(expert.pick_place_post_lift_state("VLA82-055"), "level_for_shelf")
        self.assertTrue(expert.pick_place_transit_preserves_shelf_orientation("VLA82-055"))
        self.assertFalse(expert.pick_place_transit_preserves_shelf_orientation("VLA82-036"))
        self.assertEqual(
            expert.shelf_leveling_transition(hold_valid=True, alignment_error_deg=1.5, yaw_error_deg=1.0, xy_distance=.01, steps=12),
            "lower",
        )
        self.assertEqual(
            expert.shelf_leveling_transition(hold_valid=True, alignment_error_deg=3.5, yaw_error_deg=3.0, xy_distance=.01, steps=12),
            "lower",
        )
        self.assertEqual(
            expert.shelf_leveling_transition(hold_valid=True, alignment_error_deg=8.0, yaw_error_deg=1.0, xy_distance=.01, steps=12),
            "level_for_shelf",
        )
        self.assertEqual(
            expert.shelf_leveling_transition(hold_valid=True, alignment_error_deg=1.0, yaw_error_deg=8.0, xy_distance=.01, steps=12),
            "level_for_shelf",
        )
        self.assertEqual(
            expert.shelf_leveling_transition(hold_valid=True, alignment_error_deg=1.0, yaw_error_deg=1.0, xy_distance=.08, steps=12),
            "transit",
        )
        self.assertEqual(
            expert.shelf_leveling_transition(hold_valid=False, alignment_error_deg=1.0, yaw_error_deg=1.0, xy_distance=.01, steps=12),
            "close",
        )
        self.assertFalse(expert.pick_place_target_contact_release_ready(
            "VLA82-055", target_contact=True, inside_xy=True,
            eef_distance=.02, support_alignment_deg=8.0, support_yaw_alignment_deg=1.0,
        ))
        self.assertTrue(expert.pick_place_target_contact_release_ready(
            "VLA82-055", target_contact=True, inside_xy=True,
            eef_distance=.02, support_alignment_deg=2.0, support_yaw_alignment_deg=2.0,
        ))
        self.assertFalse(expert.pick_place_target_contact_release_ready(
            "VLA82-055", target_contact=True, inside_xy=True,
            eef_distance=.02, support_alignment_deg=2.0, support_yaw_alignment_deg=12.0,
        ))
        self.assertFalse(expert.pick_place_transit_ready_for_selection(
            "VLA82-055", eef=(.14, -.16, 1.0), desired=(.14, -.16, 1.0),
            object_center=(.14, -.13, .95), release=(.14, -.16, .863),
        ))
        self.assertTrue(expert.pick_place_transit_ready_for_selection(
            "VLA82-055", eef=(.14, -.16, 1.0), desired=(.14, -.16, 1.0),
            object_center=(.14, -.155, .95), release=(.14, -.16, .863),
        ))
        self.assertTrue(expert.pick_place_transit_ready_for_selection(
            "VLA82-055", eef=(.14, -.16, 1.0), desired=(.14, -.16, 1.0),
            object_center=(.14, -.146, .95), release=(.14, -.16, .863),
        ))
        self.assertFalse(expert.pick_place_transit_ready_for_selection(
            "VLA82-058", eef=(.14, -.16, 1.0), desired=(.14, -.16, 1.0),
            object_center=(.14, -.146, .95), release=(.14, -.16, .863),
        ))
        self.assertFalse(expert.pick_place_transit_ready_for_selection(
            "VLA82-055", eef=(.14, -.16, .92), desired=(.14, -.16, 1.0),
            object_center=(.14, -.155, .89), release=(.14, -.16, .863),
        ))
        self.assertTrue(expert.pick_place_uses_height_controlled_transit("VLA82-055"))
        self.assertFalse(expert.pick_place_uses_height_controlled_transit("VLA82-036"))
        np.testing.assert_allclose(
            expert.shelf_leveling_position_target(
                release=(.14, -.16, .863), eef=(.13, -.08, .94),
                object_center=(.12, -.07, .90), lift_height=.14,
            ),
            (.13, -.08, .94),
        )

    def test_vla82_055_routes_above_and_toward_the_front_of_the_shelf(self) -> None:
        from tools.vla82_full_sim import expert

        adjusted = expert.pick_place_release_adjustment(
            "VLA82-055",
            release=(.14, -.12, .863),
            target_low=(.02, -.19, .80),
            target_high=(.26, -.05, .83),
        )
        np.testing.assert_allclose(adjusted, (.14, -.16, .863))
        self.assertEqual(expert.pick_place_lift_proof_height("VLA82-055", .045), .12)
        self.assertEqual(expert.pick_place_post_transit_state("VLA82-055"), "level_for_shelf")
        np.testing.assert_allclose(
            expert.pick_place_release_retreat_target(
                "VLA82-055", eef=(.14, -.14, .89), object_center=(.14, -.14, .86),
            ),
            (.14, -.14, 1.01),
        )

    def test_cabinet_extract_transition_debounces_brief_grasp_loss(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.cabinet_extract_transition(
            hold_valid=False, lost_frames=1, extracted=.04,
            threshold=.22, next_state="transit",
        ), "cabinet_extract")
        self.assertEqual(expert.cabinet_extract_transition(
            hold_valid=False, lost_frames=3, extracted=.04,
            threshold=.22, next_state="transit",
        ), "retry")
        self.assertEqual(expert.cabinet_extract_transition(
            hold_valid=True, lost_frames=0, extracted=.23,
            threshold=.22, next_state="transit",
        ), "transit")

    def test_dispenser_accepts_measured_safe_cabinet_extraction(self) -> None:
        from tools.vla82_full_sim import expert

        threshold = getattr(expert, "cabinet_extraction_threshold", None)
        self.assertIsNotNone(threshold)
        self.assertEqual(threshold("VLA82-060"), .218)
        self.assertEqual(threshold("VLA82-019"), .22)

    def test_counter_support_height_comes_from_collision_slab_not_union_bbox(self) -> None:
        from tools.vla82_full_sim import expert

        support_top = getattr(expert, "counter_collision_support_top_at_point", None)
        self.assertIsNotNone(support_top)
        self.assertEqual(support_top(
            patches=(
                ("counter_top_0", (0.30, -4.53, .89), (0.95, -4.03, .92)),
                ("counter_top_1", (0.30, -4.03, .89), (0.95, -3.53, .92)),
            ),
            point_xy=(.56, -4.08),
            fallback=.925,
        ), .92)

    def test_drawer_descent_uses_available_torso_when_arm_z_is_saturated(self) -> None:
        from tools.vla82_full_sim import expert

        command = getattr(expert, "drawer_vertical_descent_torso_command", None)
        self.assertIsNotNone(command)
        self.assertLess(command(
            object_z=.939, release_z=.768, torso_qpos=.136,
        ), 0.0)
        self.assertEqual(command(
            object_z=.916, release_z=.768, torso_qpos=.020,
        ), 0.0)
        self.assertEqual(command(
            object_z=.810, release_z=.768, torso_qpos=.136,
        ), 0.0)

    def test_retry_approach_recovers_an_already_verified_native_grasp(self) -> None:
        from tools.vla82_full_sim import expert

        recover = getattr(expert, "pick_place_existing_grasp_transition", None)
        self.assertIsNotNone(recover)
        self.assertEqual(recover(
            state="approach", raw_grasp=True, two_pad_contact=True,
        ), "secure")
        self.assertEqual(recover(
            state="descend", raw_grasp=True, two_pad_contact=True,
        ), "secure")
        self.assertEqual(recover(
            state="approach", raw_grasp=False, two_pad_contact=True,
        ), "approach")

    def test_regular_object_keeps_default_panda_gripper(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-005")
        self.assertEqual(gripper_type_for_request(_scene_for_spec(spec)), "PandaGripper")

    def test_knob_turn_stops_after_contact_and_required_joint_motion(self) -> None:
        from tools.vla82_full_sim import expert

        should_stop = getattr(expert, "knob_turn_should_stop", None)
        self.assertIsNotNone(should_stop)
        self.assertFalse(should_stop(
            initial_joint=0.0,
            current_joint=0.31,
            required_delta=0.30,
            contact_observed=False,
        ))
        self.assertFalse(should_stop(
            initial_joint=0.0,
            current_joint=0.29,
            required_delta=0.30,
            contact_observed=True,
        ))
        self.assertTrue(should_stop(
            initial_joint=0.0,
            current_joint=0.31,
            required_delta=0.30,
            contact_observed=True,
        ))

    def test_short_pizza_cutter_allows_drawer_yaw_hysteresis(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertGreaterEqual(
            expert.drawer_lower_yaw_tolerance("VLA82-038"),
            float(np.deg2rad(27.0)),
        )
        step_budget = getattr(expert, "pick_place_step_budget", None)
        self.assertIsNotNone(step_budget)
        self.assertGreaterEqual(step_budget("VLA82-038"), 900)
        self.assertGreaterEqual(step_budget("VLA82-041"), 1300)
        self.assertTrue(expert.drawer_locked_descent_enabled("VLA82-038"))
        self.assertTrue(expert.drawer_locked_descent_enabled("VLA82-041"))
        self.assertTrue(expert.drawer_locked_descent_enabled("VLA82-005"))
        self.assertTrue(expert.drawer_locked_descent_keeps_pose_correction("VLA82-005"))
        self.assertFalse(expert.drawer_locked_descent_keeps_pose_correction("VLA82-041"))
        self.assertFalse(expert.drawer_lower_base_recenter_enabled("VLA82-038"))
        self.assertEqual(expert.pick_place_lower_limit("VLA82-038"), .6)
        self.assertEqual(expert.drawer_lost_hold_transition(
            "VLA82-038", target_contact=True, inside_xy=True,
        ), "settle")
        self.assertEqual(expert.drawer_lost_hold_transition(
            "VLA82-038", target_contact=False, inside_xy=True,
        ), "close")

    def test_pizza_cutter_uses_smooth_pregrasp_motion_limits(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertLessEqual(expert.pick_place_pregrasp_yaw_limit("VLA82-038"), .25)
        self.assertLessEqual(expert.pick_place_approach_limit("VLA82-038"), .35)
        self.assertEqual(expert.pick_place_pregrasp_yaw_limit("VLA82-019"), .5)
        self.assertEqual(expert.pick_place_approach_limit("VLA82-019"), .8)

    def test_pizza_cutter_release_is_biased_away_from_drawer_front(self) -> None:
        from tools.vla82_full_sim import expert

        coordinate = expert.drawer_release_front_bias(
            selection_id="VLA82-038",
            coordinate=-.80,
            robot_coordinate=-1.00,
        )
        self.assertAlmostEqual(coordinate, -.73)

    def test_drawer_tools_use_selection_matched_leveling_authority(self) -> None:
        from tools.vla82_full_sim import expert

        magnitude = getattr(expert, "drawer_orientation_control_magnitude", None)
        self.assertIsNotNone(magnitude)
        self.assertEqual(magnitude("VLA82-006"), .8)
        self.assertEqual(magnitude("VLA82-005"), .25)
        self.assertEqual(magnitude("VLA82-041"), .8)
        self.assertEqual(magnitude("VLA82-038"), .8)
        self.assertEqual(magnitude("VLA82-045"), .25)

    def test_vla82_006_avoids_the_base_retreat_collision_route(self) -> None:
        from tools.vla82_full_sim.expert import drawer_base_retreat_distance

        self.assertEqual(drawer_base_retreat_distance("VLA82-006"), .30)

    def test_vla82_006_transit_does_not_enter_the_xy_dead_zone(self) -> None:
        from tools.vla82_full_sim.expert import drawer_transit_xy_reached

        self.assertFalse(drawer_transit_xy_reached("VLA82-006", distance=.033))
        self.assertTrue(drawer_transit_xy_reached("VLA82-006", distance=.011))

    def test_drawer_leveling_latch_keeps_the_first_measured_rotation_direction(self) -> None:
        from tools.vla82_full_sim.expert import drawer_leveling_rotation_latch

        first = np.array((.25, 0., 0.))
        opposite = np.array((-.25, 0., 0.))
        np.testing.assert_allclose(drawer_leveling_rotation_latch(None, first), first)
        np.testing.assert_allclose(drawer_leveling_rotation_latch(first, opposite), first)

    def test_whisk_requires_compact_horizontal_pose_before_drawer_release(self) -> None:
        from tools.vla82_full_sim.expert import drawer_release_orientation_ready

        self.assertTrue(drawer_release_orientation_ready("VLA82-005", vertical_half_extent=.054))
        self.assertFalse(drawer_release_orientation_ready("VLA82-005", vertical_half_extent=.056))
        self.assertTrue(drawer_release_orientation_ready("VLA82-005", vertical_half_extent=.030))
        self.assertTrue(drawer_release_orientation_ready("VLA82-014", vertical_half_extent=.054))

    def test_whisk_uses_tip_first_drawer_insertion(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.drawer_tip_first_insertion_enabled("VLA82-005"))
        self.assertFalse(expert.drawer_tip_first_insertion_enabled("VLA82-014"))
        self.assertEqual(
            expert.pick_place_post_transit_state("VLA82-005", front_insert=False),
            "tilt_drawer_entry",
        )
        self.assertEqual(
            expert.pick_place_post_transit_state("VLA82-014", front_insert=False),
            "lower",
        )

    def test_whisk_tip_tilt_requires_gripper_above_object(self) -> None:
        from tools.vla82_full_sim.expert import drawer_tip_tilt_ready

        self.assertFalse(drawer_tip_tilt_ready(
            eef=(1.94, -.74, .99), object_center=(1.93, -.66, .95),
        ))
        self.assertTrue(drawer_tip_tilt_ready(
            eef=(1.94, -.74, 1.03), object_center=(1.93, -.66, .95),
        ))

    def test_whisk_tip_release_requires_inner_bottom_support_without_fixture_collision(self) -> None:
        from tools.vla82_full_sim.expert import drawer_tip_release_ready

        self.assertTrue(drawer_tip_release_ready(
            inner_bottom_contact=True, inside_xy=True, fixture_collision=False,
        ))
        self.assertFalse(drawer_tip_release_ready(
            inner_bottom_contact=True, inside_xy=True, fixture_collision=True,
        ))
        self.assertFalse(drawer_tip_release_ready(
            inner_bottom_contact=False, inside_xy=True, fixture_collision=False,
        ))

    def test_whisk_tip_advance_moves_from_door_toward_drawer_interior(self) -> None:
        from tools.vla82_full_sim.expert import drawer_tip_advance_target

        target = drawer_tip_advance_target(
            object_center=(2.014, -.778, .876),
            target_low=(1.778, -.857, .701),
            target_high=(2.072, -.317, .819),
            robot_base=(1.897, -1.395, .700),
            distance=.12,
        )
        self.assertAlmostEqual(float(target[0]), 1.925)
        self.assertAlmostEqual(float(target[1]), -.658)
        self.assertAlmostEqual(float(target[2]), .876)

    def test_exact_object_support_contact_distinguishes_drawer_floor_from_door(self) -> None:
        from types import SimpleNamespace
        from tools.vla82_full_sim.expert import selected_geoms_have_contact

        names = ("obj_collision", "stack_2_main_group_2_door_g1", "stack_2_main_group_2_inner_bottom")
        raw = SimpleNamespace(sim=SimpleNamespace(
            data=SimpleNamespace(
                ncon=1,
                contact=[SimpleNamespace(geom1=0, geom2=1)],
            ),
            model=SimpleNamespace(geom_id2name=lambda index: names[index]),
        ))
        self.assertFalse(selected_geoms_have_contact(
            raw, ("obj_collision",), ("stack_2_main_group_2_inner_bottom",),
        ))
        raw.sim.data.contact[0].geom2 = 2
        self.assertTrue(selected_geoms_have_contact(
            raw, ("obj_collision",), ("stack_2_main_group_2_inner_bottom",),
        ))

    def test_whisk_tip_descent_adds_small_floor_contact_preload(self) -> None:
        from tools.vla82_full_sim.expert import drawer_tip_contact_height

        self.assertAlmostEqual(drawer_tip_contact_height(.838), .830)

    def test_single_insert_uses_the_pick_place_controller_and_keeps_insert_evidence(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_execution_phases

        self.assertEqual(pick_place_execution_phases(("insert",)), (("grasp", "place"), "insert"))
        self.assertEqual(pick_place_execution_phases(("place",)), (("grasp", "place"), "place"))
        self.assertEqual(pick_place_execution_phases(("grasp", "place")), (("grasp", "place"), "place"))
        self.assertIsNone(pick_place_execution_phases(("insert", "insert")))


    def test_stable_tall_bottle_descent_has_bounded_completion_budget(self) -> None:
        from tools.vla82_full_sim import expert

        clearance = getattr(expert, "pick_place_release_clearance", None)
        self.assertIsNotNone(clearance)
        self.assertGreaterEqual(expert.pick_place_step_budget("VLA82-006"), 700)
        self.assertGreaterEqual(expert.pick_place_step_budget("VLA82-017"), 700)
        self.assertEqual(clearance("VLA82-004"), 0.0)
        self.assertEqual(clearance("VLA82-017"), 0.0)
        self.assertEqual(clearance("VLA82-042"), 0.0)
        self.assertEqual(clearance("VLA82-020"), .008)

    def test_fragile_cabinet_objects_use_fast_controlled_descent(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.pick_place_lower_limit("VLA82-019"), .8)
        self.assertEqual(expert.pick_place_lower_limit("VLA82-020"), 1.0)

    def test_foil_box_vertical_gate_clears_orientation_controller_deadband(self) -> None:
        from tools.vla82_full_sim import expert

        threshold = expert.drawer_vertical_transport_threshold("VLA82-045")
        self.assertGreaterEqual(threshold, .036)
        self.assertEqual(
            expert.drawer_transit_control_stage(
                vertical_half_extent=.0359,
                depth_alignment_error=np.deg2rad(1.0),
                xy_reached=True,
                joint5=-.08,
                depth_alignment_latched=True,
                horizontal_latched=True,
                vertical_threshold=threshold,
            ),
            "ready",
        )

    def test_whisk_release_moves_gripper_clear_of_drawer_front(self) -> None:
        from tools.vla82_full_sim import expert

        coordinate = expert.drawer_release_front_bias(
            selection_id="VLA82-005", coordinate=-.74,
            robot_coordinate=-1.40, active=True,
        )
        self.assertGreaterEqual(coordinate, -.68)
        self.assertEqual(expert.drawer_release_front_bias(
            selection_id="VLA82-045", coordinate=-.74,
            robot_coordinate=-1.40, active=True,
        ), -.74)

    def test_measuring_cup_release_clears_drawer_front_lip(self) -> None:
        from tools.vla82_full_sim import expert

        coordinate = expert.drawer_release_front_bias(
            selection_id="VLA82-014", coordinate=-.784,
            robot_coordinate=-1.40, active=True,
        )
        self.assertGreaterEqual(coordinate, -.715)

    def test_drawer_release_retreats_toward_interior_not_robot(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.drawer_post_release_interior_retreat_target(
            eef=(1.93, -.728, .854),
            object_center=(1.926, -.715, .792),
            target_low=(1.778, -.857, .701),
            target_high=(2.072, -.317, .819),
        )
        self.assertGreater(target[1], -.66)
        self.assertGreater(target[2], .90)

    def test_opposing_finger_contact_accepts_wide_object_small_closure(self) -> None:
        from types import SimpleNamespace
        from tools.vla82_full_sim.environment import ContactEvidence
        from tools.vla82_full_sim import predicates

        snapshot = SimpleNamespace(contact_evidence=(
            ContactEvidence("obj_collision", "gripper0_right_finger1_collision", "obj", None, None, None, -.0002),
            ContactEvidence("obj_collision", "gripper0_right_finger2_collision", "obj", None, None, None, -.0002),
        ))
        self.assertTrue(predicates.opposing_finger_grasp_sufficient(
            snapshot, object_id="obj", closure=.0013,
        ))
        one_sided = SimpleNamespace(contact_evidence=snapshot.contact_evidence[:1])
        self.assertFalse(predicates.opposing_finger_grasp_sufficient(
            one_sided, object_id="obj", closure=.0013,
        ))

    def test_released_object_inside_target_enters_settle_without_regrasp(self) -> None:
        from tools.vla82_full_sim import expert

        transition = getattr(expert, "pick_place_released_target_transition", None)
        self.assertIsNotNone(transition)
        self.assertEqual(
            transition(
                raw_grasp=False,
                target_contact=True,
                object_center=np.array((1.90, -0.73, 0.79)),
                target_low=np.array((1.78, -0.86, 0.70)),
                target_high=np.array((2.07, -0.32, 0.82)),
            ),
            "settle",
        )
        self.assertIsNone(
            transition(
                raw_grasp=False,
                target_contact=False,
                object_center=np.array((1.90, -0.73, 0.90)),
                target_low=np.array((1.78, -0.86, 0.70)),
                target_high=np.array((2.07, -0.32, 0.82)),
            )
        )

    def test_native_fixture_uses_authoritative_operation_phases(self) -> None:
        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-008")
        request = _scene_for_spec(spec)
        self.assertEqual(tuple(request.primary_asset.affordances), tuple(spec.phases))

    def test_hinged_fixture_close_direction_reduces_joint_angle(self) -> None:
        from types import SimpleNamespace
        from tools.vla82_full_sim.expert import (
            drawer_close_stage,
            fixture_close_contact_geometries,
            fixture_close_push_direction,
        )

        self.assertEqual(
            drawer_close_stage(exact_handle_contact=False, joint_position=1.15),
            "approach_handle",
        )

        direction = fixture_close_push_direction(
            joint_type=3,
            joint_position=1.15,
            joint_axis_world=np.array((1.0, 0.0, 0.0)),
            joint_anchor_world=np.array((0.8, -1.87, 0.40)),
            handle_world=np.array((1.27, -1.87, 0.55)),
        )
        # Positive q must decrease toward zero.  For axis +X and lever +Z,
        # cross(axis, lever) is -Y, so the closing tangent is +Y.
        self.assertGreater(float(direction[1]), 0.99)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        contract = SimpleNamespace(fixture_contact_joints={
            "oven_door_panel": "oven_door_joint",
            "oven_door_handle": "oven_door_joint",
            "other_fixture": "other_joint",
        })
        self.assertEqual(
            fixture_close_contact_geometries(contract, "oven_door_joint"),
            ("oven_door_panel", "oven_door_handle"),
        )

    def test_flat_desktop_keyboard_uses_reachable_approach_height(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_approach_height

        self.assertAlmostEqual(pick_place_approach_height("VLA82-052", default=.15), .07)
        self.assertAlmostEqual(pick_place_approach_height("VLA82-014", default=.15), .15)

    def test_hinged_fixture_open_direction_increases_joint_angle(self) -> None:
        from tools.vla82_full_sim.expert import fixture_open_pull_direction

        direction = fixture_open_pull_direction(
            joint_type=3,
            joint_position=0.0,
            joint_axis_world=np.array((1.0, 0.0, 0.0)),
            joint_anchor_world=np.array((0.8, -1.87, 0.40)),
            handle_world=np.array((1.27, -1.87, 0.55)),
        )
        # d(handle)/dq is -Y, the physical tangent for increasing a closed
        # hinged door's positive joint coordinate.
        self.assertLess(float(direction[1]), -0.99)

    def test_fixture_open_requires_full_presentation_angle(self) -> None:
        from tools.vla82_full_sim.expert import fixture_open_presentation_complete

        self.assertFalse(fixture_open_presentation_complete(contact_seen=True, joint_position=.36))
        self.assertTrue(fixture_open_presentation_complete(contact_seen=True, joint_position=.80))

    def test_fixture_open_hold_target_tracks_the_moving_handle(self) -> None:
        from tools.vla82_full_sim.expert import fixture_open_contact_hold_target

        target = fixture_open_contact_hold_target(
            handle_world=(1.0, 2.0, .8), contact_offset=(.02, -.01, .03),
            pull_direction=(0.0, -1.0, 0.0),
        )
        np.testing.assert_allclose(target, (1.02, 1.95, .83), atol=1e-9)

    def test_negative_microwave_hinge_close_direction_increases_joint_toward_zero(self) -> None:
        from tools.vla82_full_sim.expert import fixture_close_push_direction

        direction = fixture_close_push_direction(
            joint_type=3,
            joint_position=-1.47,
            joint_axis_world=np.array((0.0, 0.0, 1.0)),
            joint_anchor_world=np.array((0.0, 0.0, 0.0)),
            handle_world=np.array((1.0, 0.0, 0.0)),
        )
        # Negative q must increase toward zero; d(handle)/dq is +Y.
        np.testing.assert_allclose(direction, np.array((0.0, 1.0, 0.0)), atol=1e-6)

    def test_hinge_close_reacquires_lost_handle_before_another_push(self) -> None:
        from tools.vla82_full_sim import expert

        stage = getattr(expert, "fixture_close_active_stage", None)
        push_distance = getattr(expert, "fixture_close_push_distance", None)
        target = getattr(expert, "fixture_close_target", None)
        self.assertIsNotNone(stage)
        self.assertIsNotNone(push_distance)
        self.assertIsNotNone(target)
        self.assertEqual(stage(
            exact_handle_contact=False,
            contact_seen=True,
            joint_position=1.15,
        ), "approach_handle")
        self.assertEqual(stage(
            exact_handle_contact=False,
            contact_seen=True,
            joint_position=1.15,
            selection_id="VLA82-034",
        ), "push_close")
        self.assertEqual(push_distance(joint_type=3), .01)
        self.assertEqual(push_distance(joint_type=2), .24)
        handle = np.array((1.27, -1.87, .55))
        eef = np.array((1.25, -1.94, .57))
        tangent = np.array((0., 0., 1.))
        self.assertTrue(np.allclose(
            target(
                stage="approach_handle", handle_world=handle,
                eef_world=eef, push_direction=tangent, push_distance=.05,
                contact_offset=None, joint_type=3,
            ),
            handle,
        ))
        contact_offset = eef - handle
        self.assertTrue(np.allclose(
            target(
                stage="approach_handle", handle_world=handle,
                eef_world=eef, push_direction=tangent, push_distance=.05,
                contact_offset=contact_offset, joint_type=3,
            ),
            handle + .9 * contact_offset,
        ))
        self.assertTrue(np.allclose(
            target(
                stage="push_close", handle_world=handle,
                eef_world=eef, push_direction=tangent, push_distance=.05,
                contact_offset=contact_offset, joint_type=3,
            ),
            eef + np.array((0., 0., .05)),
        ))
        self.assertTrue(np.allclose(
            target(
                stage="push_close", handle_world=handle,
                eef_world=eef, push_direction=tangent, push_distance=.24,
                contact_offset=contact_offset, joint_type=2,
            ),
            handle + np.array((0., 0., .24)),
        ))

    def test_long_slide_close_follows_with_mobile_base_only_during_contact(self) -> None:
        from tools.vla82_full_sim import expert

        pulse = expert.fixture_close_base_follow_local(
            joint_type=2, exact_contact=True, joint_position=.23,
            push_direction_world=(1., 0., 0.), base_rotation=np.eye(3),
        )
        np.testing.assert_allclose(pulse, (.18, 0.))
        self.assertIsNone(expert.fixture_close_base_follow_local(
            joint_type=3, exact_contact=True, joint_position=1.1,
            push_direction_world=(1., 0., 0.), base_rotation=np.eye(3),
        ))

    def test_unlabelled_drawer_uses_nearest_joint_bound_contact_geom(self) -> None:
        from tools.vla82_full_sim import expert

        selected = expert.fixture_close_contact_target_name(
            labelled_geoms=(),
            joint_bound_geoms=("drawer_panel", "drawer_front_edge"),
            geom_positions={
                "drawer_panel": (1., 0., 0.),
                "drawer_front_edge": (.2, 0., 0.),
            },
            eef_world=(0., 0., 0.),
        )
        self.assertEqual(selected, "drawer_front_edge")

    def test_fixture_close_targets_contactable_handle_not_semantic_region(self) -> None:
        from tools.vla82_full_sim import expert

        selected = expert.fixture_close_contact_target_name(
            labelled_geoms=("door_handle_reg_main", "door_handle_g0"),
            joint_bound_geoms=("door_panel", "door_handle_reg_main", "door_handle_g0"),
            contactable_geoms=("door_handle_g0",),
            geom_positions={
                "door_handle_reg_main": (.1, 0., 0.),
                "door_handle_g0": (.2, 0., 0.),
                "door_panel": (.3, 0., 0.),
            },
            eef_world=(0., 0., 0.),
        )
        self.assertEqual(selected, "door_handle_g0")

    def test_handle_grasp_prefers_the_largest_contactable_bar_over_a_near_endcap(self) -> None:
        from tools.vla82_full_sim import expert

        selected = expert.fixture_close_contact_target_name(
            labelled_geoms=("door_handle_g1", "door_handle_g9"),
            joint_bound_geoms=("door_panel", "door_handle_g1", "door_handle_g9"),
            contactable_geoms=("door_handle_g1", "door_handle_g9"),
            geom_positions={
                "door_handle_g1": (.20, 0., 0.),
                "door_handle_g9": (.01, 0., 0.),
                "door_panel": (.30, 0., 0.),
            },
            geom_half_sizes={
                "door_handle_g1": (.008, .003, .090),
                "door_handle_g9": (.008, .003, .006),
            },
            eef_world=(0., 0., 0.),
            prefer_grasp_bar=True,
        )
        self.assertEqual(selected, "door_handle_g1")

    def test_microwave_close_targets_nearest_contactable_door_panel(self) -> None:
        from tools.vla82_full_sim import expert

        selected = expert.fixture_close_contact_target_name(
            labelled_geoms=("microwave_door_handle_main",),
            joint_bound_geoms=("microwave_door_panel", "microwave_door_handle_main"),
            contactable_geoms=("microwave_door_panel", "microwave_door_handle_main"),
            geom_positions={
                "microwave_door_panel": (.03, 0., 0.),
                "microwave_door_handle_main": (.15, 0., 0.),
            },
            eef_world=(0., 0., 0.),
            prefer_nearest_joint_bound=True,
        )
        self.assertEqual(selected, "microwave_door_panel")

    def test_microwave_bypass_uses_edge_farthest_from_hinge(self) -> None:
        from tools.vla82_full_sim import expert

        points = expert.fixture_hinge_free_edge_waypoints(
            panel_center=(0., 0., 0.), panel_rotation=np.eye(3),
            panel_half_size=(.25, .01, .15), hinge_anchor=(.25, 0., 0.),
            eef_world=(0., -.20, 0.), push_direction=(0., -1., 0.),
            edge_clearance=.09, side_clearance=.06,
        )

        self.assertLess(points["free_edge"][0], -.249)
        self.assertLess(points["outside_current_side"][0], -.33)
        self.assertGreater(points["outside_closing_side"][1], .05)
        self.assertGreater(points["contact_prepoint"][1], .05)

    def test_microwave_bypass_rotates_with_the_live_panel(self) -> None:
        from tools.vla82_full_sim import expert

        yaw = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
        points = expert.fixture_hinge_free_edge_waypoints(
            panel_center=(1., 2., .8), panel_rotation=yaw,
            panel_half_size=(.25, .01, .15), hinge_anchor=(1., 2.25, .8),
            eef_world=(1.2, 2., .8), push_direction=(1., 0., 0.),
            edge_clearance=.09, side_clearance=.06,
        )

        self.assertGreater(
            float(np.linalg.norm(points["free_edge"] - np.array((1., 2., .8)))),
            .24,
        )
        self.assertLess(float(np.dot(
            points["contact_prepoint"] - np.array((1., 2., .8)),
            np.array((1., 0., 0.)),
        )), -.05)

    def test_free_edge_bypass_is_microwave_only(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.fixture_close_requires_free_edge_bypass(
            "microwave_microjoint",
        ))
        self.assertFalse(expert.fixture_close_requires_free_edge_bypass(
            "microwave_microjoint", initial_joint_position=-.50,
        ))
        self.assertFalse(expert.fixture_close_requires_free_edge_bypass(
            "fridge_door_joint",
        ))
        self.assertFalse(expert.fixture_close_requires_free_edge_bypass(
            "drawer_slide_joint",
        ))

    def test_microwave_close_uses_the_observed_closing_tangent(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(np.allclose(
            expert.fixture_close_selection_push_direction("VLA82-029", (1., -2., 0.)),
            (-1., 2., 0.),
        ))

    def test_microwave_close_contract_uses_its_fixture_namespace(self) -> None:
        from tools.vla82_full_sim.predicates import _expected_fixture_token
        from tools.run_vla82_full_simulation import _load_compiled_specs

        spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-029")
        self.assertEqual(_expected_fixture_token(spec, "close"), "microwave")

    def test_microwave_bypass_stage_orders_waypoints_and_gates_push_on_contact(self) -> None:
        from tools.vla82_full_sim import expert

        cases = (
            ((False, .20, .20, .20, False), "bypass_to_free_edge"),
            ((False, .01, .20, .20, False), "bypass_cross_plane"),
            ((False, .01, .01, .20, False), "bypass_return_to_panel"),
            ((False, .01, .01, .01, False), "seat_contact"),
            ((False, .01, .01, .01, True), "push_close"),
            ((True, .01, .01, .01, True), "settle_closed"),
        )
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(expert.fixture_close_bypass_stage(
                    joint_closed=arguments[0],
                    free_edge_distance=arguments[1],
                    cross_plane_distance=arguments[2],
                    precontact_distance=arguments[3],
                    exact_contact=arguments[4],
                    tolerance=.025,
                ), expected)

        self.assertEqual(expert.fixture_close_bypass_stage(
            joint_closed=False,
            free_edge_distance=.01,
            cross_plane_distance=.01,
            precontact_distance=.01,
            exact_contact=False,
            tolerance=.025,
        ), "seat_contact")
        self.assertEqual(expert.fixture_close_bypass_stage(
            joint_closed=False,
            free_edge_distance=.01,
            cross_plane_distance=.01,
            precontact_distance=.20,
            exact_contact=True,
            tolerance=.025,
        ), "push_close")
        self.assertEqual(expert.fixture_close_bypass_stage(
            joint_closed=False,
            free_edge_distance=.01,
            cross_plane_distance=.01,
            precontact_distance=.20,
            exact_contact=False,
            contact_seen_after_cross=True,
            tolerance=.025,
        ), "seat_contact")

    def test_fixture_close_contact_normal_aligns_and_smooths(self) -> None:
        from tools.vla82_full_sim import expert

        filtered = expert.fixture_close_filtered_contact_normal(
            previous=(1., 0., 0.), current=(-.8, -.6, 0.),
        )

        self.assertGreater(filtered[0], .9)
        self.assertGreater(filtered[1], 0.)
        self.assertAlmostEqual(float(np.linalg.norm(filtered)), 1., places=6)

    def test_fixture_close_direction_adds_only_bounded_inward_bias(self) -> None:
        from tools.vla82_full_sim import expert

        tangent = np.array((.05, -.99875, 0.))
        normal = np.array((.95, .31225, 0.))
        direction, bias = expert.fixture_close_contact_preserving_direction(
            tangent=tangent, contact_normal=normal,
            minimum_inward=.08, max_normal_bias=.45,
        )

        self.assertGreaterEqual(float(np.dot(direction, normal)), .079)
        self.assertGreater(float(np.dot(direction, tangent)), .8)
        self.assertGreater(bias, 0.)
        self.assertLessEqual(bias, .45)

    def test_fixture_close_direction_does_not_bias_an_inward_tangent(self) -> None:
        from tools.vla82_full_sim import expert

        direction, bias = expert.fixture_close_contact_preserving_direction(
            tangent=(1., 0., 0.), contact_normal=(1., 0., 0.),
        )

        np.testing.assert_allclose(direction, (1., 0., 0.), atol=1e-8)
        self.assertEqual(bias, 0.)

    def test_knob_contact_targets_the_finger_pad_not_the_wrist_center(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.knob_wrist_target_for_pad_contact(
            eef=(1.0, 2.0, 3.0),
            pad_center=(1.0, 2.04, 3.0),
            knob_center=(4.0, 5.0, 6.0),
        )

        np.testing.assert_allclose(target, (4.0, 4.96, 6.0), atol=1e-8)

    def test_knob_reseat_target_rotates_with_the_knob_body(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.knob_rotating_surface_target(
            knob_center=(1.0, 2.0, 3.0),
            initial_contact_offset=(.04, 0.0, 0.0),
            initial_rotation=np.eye(3),
            current_rotation=np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        )

        np.testing.assert_allclose(target, (1.0, 2.04, 3.0), atol=1e-8)

    def test_fixture_close_contact_control_rejects_degenerate_vectors(self) -> None:
        from tools.vla82_full_sim import expert

        with self.assertRaisesRegex(ValueError, "normal"):
            expert.fixture_close_filtered_contact_normal(
                previous=None, current=(0., 0., 0.),
            )
        with self.assertRaisesRegex(ValueError, "tangent"):
            expert.fixture_close_contact_preserving_direction(
                tangent=(0., 0., 0.), contact_normal=(1., 0., 0.),
            )

    def test_fixture_close_selects_deepest_target_contact_normal(self) -> None:
        from tools.vla82_full_sim import expert

        normal = expert.fixture_close_deepest_contact_normal((
            {"normal_gripper_to_object": (1., 0., 0.), "distance": -.0001},
            {"normal_gripper_to_object": (0., 1., 0.), "distance": -.0006},
        ))

        np.testing.assert_allclose(normal, (0., 1., 0.))
        self.assertIsNone(expert.fixture_close_deepest_contact_normal(()))

    def test_fixture_close_reseat_uses_small_last_normal_step(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.fixture_close_contact_reseat_target(
            contact_reference=(1., 2., 3.), last_contact_normal=(0., 1., 0.),
            distance=.006,
        )

        np.testing.assert_allclose(target, (1., 2.006, 3.))
        with self.assertRaisesRegex(ValueError, "normal"):
            expert.fixture_close_contact_reseat_target(
                contact_reference=(1., 2., 3.), last_contact_normal=(0., 0., 0.),
            )

    def test_fixture_close_selects_thin_broad_non_handle_panel(self) -> None:
        from tools.vla82_full_sim import expert

        selected = expert.fixture_close_panel_geom_name(
            candidate_names=("door_frame", "door_handle_main", "door_panel"),
            geom_types={"door_frame": 6, "door_handle_main": 6, "door_panel": 6},
            geom_sizes={
                "door_frame": (.25, .04, .18),
                "door_handle_main": (.18, .01, .015),
                "door_panel": (.234, .0028, .165),
            },
        )

        self.assertEqual(selected, "door_panel")

    def test_fixture_close_upper_panel_waypoints_clear_handle_and_top(self) -> None:
        from tools.vla82_full_sim import expert

        points = expert.fixture_close_upper_panel_waypoints(
            panel_center=(0., 0., 1.3), panel_rotation=np.eye(3),
            panel_half_size=(.234, .0028, .165),
            hinge_anchor=(.234, 0., 1.3),
            handle_world=(-.234, 0., 1.3), push_direction=(0., -1., 0.),
            outside_closing_side=(-.33, .06, 1.3),
        )

        self.assertGreaterEqual(points["face"][2] - 1.3, .07)
        self.assertGreaterEqual(1.465 - points["face"][2], .08)
        self.assertAlmostEqual(points["precontact"][1] - points["face"][1], .06)
        self.assertAlmostEqual(points["raise"][2], points["precontact"][2])
        np.testing.assert_allclose(points["free_edge_direction"], (-1., 0., 0.))
        self.assertAlmostEqual(float(np.linalg.norm(points["horizontal_offset"])), .65 * .234)
        self.assertAlmostEqual(points["edge_clearance"], .35 * .234)
        self.assertAlmostEqual(points["face"][0], -.65 * .234)
        self.assertLess(float(np.linalg.norm(points["precontact"] - points["raise"])), .20)

    def test_fixture_close_upper_panel_waypoints_follow_panel_rotation(self) -> None:
        from tools.vla82_full_sim import expert

        yaw = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
        points = expert.fixture_close_upper_panel_waypoints(
            panel_center=(1., 2., 1.3), panel_rotation=yaw,
            panel_half_size=(.234, .0028, .165),
            hinge_anchor=(1., 2.234, 1.3),
            handle_world=(1., 1.766, 1.3), push_direction=(1., 0., 0.),
            outside_closing_side=(.94, 1.67, 1.3),
        )

        self.assertGreater(points["face"][2], 1.37)
        np.testing.assert_allclose(points["vertical_axis"], (0., 0., 1.))

    def test_fixture_close_upper_panel_waypoints_reject_unsafe_clearance(self) -> None:
        from tools.vla82_full_sim import expert

        with self.assertRaisesRegex(ValueError, "handle clearance"):
            expert.fixture_close_upper_panel_waypoints(
                panel_center=(0., 0., 1.3), panel_rotation=np.eye(3),
                panel_half_size=(.234, .0028, .165),
                hinge_anchor=(.234, 0., 1.3),
                handle_world=(-.234, 0., 1.35), push_direction=(0., -1., 0.),
                outside_closing_side=(-.33, .06, 1.3),
            )
        with self.assertRaisesRegex(ValueError, "edge clearance"):
            expert.fixture_close_upper_panel_waypoints(
                panel_center=(0., 0., 1.3), panel_rotation=np.eye(3),
                panel_half_size=(.234, .0028, .165),
                hinge_anchor=(.234, 0., 1.3),
                handle_world=(-.234, 0., 1.3), push_direction=(0., -1., 0.),
                outside_closing_side=(-.33, .06, 1.3),
                horizontal_free_edge_fraction=.80,
            )

    def test_fixture_close_upper_panel_stage_requires_ordered_panel_contact(self) -> None:
        from tools.vla82_full_sim import expert

        cases = (
            ((False, False, False, False, False), "raise_above_handle"),
            ((False, True, False, False, False), "traverse_to_upper_panel"),
            ((False, True, True, False, False), "seat_upper_panel"),
            ((False, True, True, True, True), "push_close"),
            ((False, True, True, False, True), "reseat_upper_panel"),
            ((True, True, True, True, True), "settle_closed"),
        )
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(expert.fixture_close_upper_panel_stage(
                    joint_closed=arguments[0], raised=arguments[1],
                    traversed=arguments[2], exact_panel_contact=arguments[3],
                    panel_contact_seen=arguments[4],
                ), expected)

        self.assertEqual(expert.fixture_close_upper_panel_stage(
            joint_closed=False, raised=True, traversed=True,
            exact_panel_contact=False, panel_contact_seen=False,
        ), "seat_upper_panel")

    def test_fixture_raise_base_assist_requires_a_safe_plateau(self) -> None:
        from tools.vla82_full_sim import expert

        plateau = np.linspace(.0290, .0278, 40)
        common = dict(
            stage="raise_above_handle", stage_frames=60,
            distance_history=plateau, target_delta_world=(1., 0., .2),
            base_rotation=np.eye(3), cumulative_displacement=.01,
            collision=False, locked=False,
        )
        command = expert.fixture_raise_base_assist_local(**common)
        np.testing.assert_allclose(command, (.08, 0.), atol=1e-8)

        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "stage_frames": 59},
        ))
        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "distance_history": np.linspace(.04, .03, 40)},
        ))
        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "distance_history": np.full(40, .024)},
        ))
        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "cumulative_displacement": .05},
        ))
        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "collision": True},
        ))
        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "locked": True},
        ))
        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "stage": "traverse_to_upper_panel"},
        ))
        self.assertIsNone(expert.fixture_raise_base_assist_local(
            **{**common, "target_delta_world": (0., 0., .2)},
        ))

    def test_fixture_raise_base_assist_converts_world_to_local_direction(self) -> None:
        from tools.vla82_full_sim import expert

        yaw = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
        command = expert.fixture_raise_base_assist_local(
            stage="raise_above_handle", stage_frames=60,
            distance_history=np.linspace(.0290, .0278, 40),
            target_delta_world=(1., 0., .2), base_rotation=yaw,
            cumulative_displacement=0., collision=False, locked=False,
        )

        np.testing.assert_allclose(command, (0., -.08), atol=1e-8)

    def test_fixture_close_prefers_central_handle_bar_over_end_caps(self) -> None:
        from tools.vla82_full_sim import expert

        selected = expert.fixture_close_contact_target_name(
            labelled_geoms=("fridge_handle_main", "fridge_handle_2", "fridge_handle_1"),
            joint_bound_geoms=("door_panel", "fridge_handle_main", "fridge_handle_2", "fridge_handle_1"),
            contactable_geoms=("fridge_handle_main", "fridge_handle_2", "fridge_handle_1"),
            geom_positions={
                "fridge_handle_main": (1., 0., 0.),
                "fridge_handle_2": (.2, 0., 0.),
                "fridge_handle_1": (1.8, 0., 0.),
                "door_panel": (1., .1, 0.),
            },
            eef_world=(0., 0., 0.),
        )
        self.assertEqual(selected, "fridge_handle_main")

    def test_labelled_fixture_requires_handle_contact_before_close_push(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.fixture_close_exact_contact_geometries(
            labelled_handle_geoms=("door_handle_g0", "door_handle_reg_main"),
            joint_bound_geoms=("door_panel", "door_handle_g0", "door_handle_reg_main"),
        ), ("door_handle_g0", "door_handle_reg_main"))
        self.assertEqual(expert.fixture_close_exact_contact_geometries(
            labelled_handle_geoms=(),
            joint_bound_geoms=("drawer_front", "drawer_edge"),
        ), ("drawer_front", "drawer_edge"))
        self.assertEqual(expert.fixture_close_exact_contact_geometries(
            labelled_handle_geoms=("door_handle_main",),
            joint_bound_geoms=("door_panel", "door_handle_main"),
            allow_panel_contact=True,
        ), ("door_panel", "door_handle_main"))

    def test_vla028_close_requires_two_pad_handle_grasp(self) -> None:
        from tools.vla82_full_sim import expert

        requirement = getattr(expert, "fixture_close_requires_two_pad_handle_grasp", None)
        self.assertIsNotNone(requirement)
        self.assertTrue(requirement("VLA82-028"))
        self.assertFalse(requirement("VLA82-027"))

    def test_microwave_door_close_requires_a_two_pad_handle_grasp(self) -> None:
        """A door-close result cannot be accepted from a panel-only push."""
        from tools.vla82_full_sim import expert

        requirement = getattr(expert, "fixture_close_requires_two_pad_handle_grasp", None)
        gate = getattr(expert, "fixture_close_handle_grasp_observed", None)
        self.assertIsNotNone(requirement)
        self.assertIsNotNone(gate)
        self.assertTrue(requirement("VLA82-029"))
        self.assertFalse(gate("VLA82-029", [{
            "stage": "push_close",
            "exact_two_pad_handle_contact": False,
            "exact_two_finger_handle_contact": False,
        }]))

    def test_vla028_rejects_closed_door_without_two_pad_handle_grasp(self) -> None:
        from tools.vla82_full_sim import expert

        gate = getattr(expert, "fixture_close_handle_grasp_observed", None)
        self.assertIsNotNone(gate)
        self.assertFalse(gate("VLA82-028", [{
            "stage": "approach_handle",
            "exact_two_pad_handle_contact": False,
        }]))
        self.assertTrue(gate("VLA82-028", [{
            "stage": "push_close",
            "exact_two_pad_handle_contact": True,
        }]))
        self.assertTrue(gate("VLA82-028", [{
            "stage": "push_close",
            "exact_two_finger_handle_contact": True,
        }]))
        self.assertTrue(gate("VLA82-027", []))

    def test_vla028_closes_gripper_before_handle_body_collision(self) -> None:
        from tools.vla82_full_sim import expert

        command = getattr(expert, "fixture_close_gripper_command", None)
        self.assertIsNotNone(command)
        self.assertEqual(command(
            handle_grasp_required=True, bypass_required=False,
            contact_seen=False, pad_handle_distance=.20,
        ), .8)
        self.assertEqual(command(
            handle_grasp_required=True, bypass_required=False,
            contact_seen=False, pad_handle_distance=.30,
        ), -1.0)
        self.assertEqual(command(
            handle_grasp_required=True, bypass_required=False,
            contact_seen=False, pad_handle_distance=.20,
            orientation_ready=False,
        ), -1.0)

    def test_handle_grasp_surface_normal_faces_the_robot(self) -> None:
        from tools.vla82_full_sim import expert

        normal = getattr(expert, "fixture_handle_grasp_surface_normal", None)
        self.assertIsNotNone(normal)
        np.testing.assert_allclose(normal(
            handle_rotation=np.eye(3), handle_half_size=(.30, .01, .01),
            handle_world=(0., 0., 0.), eef_world=(0., -1., 0.),
        ), (0., -1., 0.))
        np.testing.assert_allclose(normal(
            handle_rotation=np.array(((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.))),
            handle_half_size=(.01, .30, .01),
            handle_world=(0., 0., 0.), eef_world=(1., -1., .3),
        ), (0., -1., 0.))
        np.testing.assert_allclose(normal(
            handle_rotation=np.array(((.07, 1., 0.), (-1., .07, 0.), (0., 0., 1.))),
            handle_half_size=(.01, .30, .01),
            handle_world=(0., 0., 0.), eef_world=(.47, .02, .07),
        ), (.07, -1., 0.), atol=.01)

    def test_handle_pregrasp_point_stays_outside_the_door_face(self) -> None:
        from tools.vla82_full_sim import expert

        point = getattr(expert, "fixture_handle_pregrasp_point", None)
        self.assertIsNotNone(point)
        np.testing.assert_allclose(point(
            handle_world=(1., 2., 3.), outward_normal=(0., -1., 0.), clearance=.10,
        ), (1., 1.9, 3.))

    def test_handle_pregrasp_tolerance_accounts_for_the_open_gripper(self) -> None:
        from tools.vla82_full_sim import expert

        reached = getattr(expert, "fixture_handle_pregrasp_reached", None)
        self.assertIsNotNone(reached)
        self.assertTrue(reached(.052))
        self.assertFalse(reached(.061))

    def test_handle_safe_approach_stays_above_the_pregrasp_corridor(self) -> None:
        from tools.vla82_full_sim import expert

        point = getattr(expert, "fixture_handle_safe_approach_point", None)
        self.assertIsNotNone(point)
        safe = point(
            pregrasp_target=(.93, -1.87, .87),
            eef_world=(1.14, -2.38, 1.30),
        )
        np.testing.assert_allclose(safe[:2], (.93, -1.87))
        self.assertGreaterEqual(float(safe[2]), 1.30)
        self.assertGreater(float(safe[2]), .87)

    def test_handle_safe_lift_moves_only_upward_before_crossing_the_door(self) -> None:
        from tools.vla82_full_sim import expert

        point = getattr(expert, "fixture_handle_safe_lift_point", None)
        self.assertIsNotNone(point)
        lift = point(
            eef_world=(1.14, -2.38, 1.30),
            pregrasp_target=(.93, -1.87, .87),
        )
        np.testing.assert_allclose(lift[:2], (1.14, -2.38))
        self.assertGreater(float(lift[2]), 1.30)

    def test_handle_pregrasp_state_is_revoked_after_the_wrist_drifts_away(self) -> None:
        from tools.vla82_full_sim import expert

        state = getattr(expert, "fixture_handle_pregrasp_state", None)
        self.assertIsNotNone(state)
        self.assertTrue(state(previously_reached=False, distance=.052))
        self.assertTrue(state(previously_reached=True, distance=.090))
        self.assertFalse(state(previously_reached=True, distance=.116))

    def test_handle_pregrasp_translates_without_rotating_away_from_the_handle(self) -> None:
        from tools.vla82_full_sim import expert

        command = getattr(expert, "fixture_handle_pregrasp_rotation_command", None)
        self.assertIsNotNone(command)
        np.testing.assert_allclose(
            command("pregrasp_handle", (.3, -.2, .1)), (0., 0., 0.),
        )
        np.testing.assert_allclose(
            command("safe_approach_above_handle", (.3, -.2, .1)), (0., 0., 0.),
        )
        np.testing.assert_allclose(
            command("orient_handle", (.3, -.2, .1)), (.3, -.2, .1),
        )

    def test_vla031_keeps_orientation_control_during_safe_handle_approach(self) -> None:
        from tools.vla82_full_sim import expert

        np.testing.assert_allclose(
            expert.fixture_handle_pregrasp_rotation_command(
                "safe_approach_above_handle", (.3, -.2, .1), selection_id="VLA82-031",
            ),
            (.3, -.2, .1),
        )

    def test_handle_safe_waypoints_remain_complete_after_descent_starts(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertTrue(expert.fixture_handle_waypoint_state(
            previously_reached=True, distance=.20, tolerance=.055,
        ))
        self.assertTrue(expert.fixture_handle_waypoint_state(
            previously_reached=False, distance=.05, tolerance=.055,
        ))
        self.assertFalse(expert.fixture_handle_waypoint_state(
            previously_reached=False, distance=.06, tolerance=.055,
        ))

    def test_vla031_safe_lift_first_retreats_to_the_front_clearance(self) -> None:
        from tools.vla82_full_sim import expert

        target = expert.fixture_handle_safe_lift_target_for_selection(
            "VLA82-031",
            eef_world=(.602, -4.243, 1.292),
            pregrasp_target=(.585, -4.162, 1.569),
            outward_normal=(.3825, .923956, 0.0),
        )
        np.testing.assert_allclose(target[:2], (.604125, -4.1158022), atol=1e-6)
        self.assertGreater(float(target[2]), 1.7)

    def test_handle_required_close_cannot_finish_before_a_handle_grasp(self) -> None:
        from tools.vla82_full_sim import expert

        self.assertEqual(expert.fixture_close_active_stage(
            exact_handle_contact=False,
            contact_seen=False,
            joint_position=.001,
            completion_allowed=False,
        ), "approach_handle")


    def test_handle_tool_axis_uses_the_stable_negative_eef_z_axis(self) -> None:
        from tools.vla82_full_sim import expert

        axis = getattr(expert, "fixture_handle_tool_axis_from_eef_rotation", None)
        self.assertIsNotNone(axis)
        np.testing.assert_allclose(axis(np.eye(3)), (0., 0., -1.))

    def test_vla028_handle_close_uses_verified_hinge_direction(self) -> None:
        from tools.vla82_full_sim import expert

        direction = getattr(expert, "fixture_close_handle_grasp_direction", None)
        self.assertIsNotNone(direction)
        np.testing.assert_allclose(direction("VLA82-028", (.9, -.2, 0.)), (-.9, .2, 0.))
        np.testing.assert_allclose(direction("VLA82-027", (.9, -.2, 0.)), (.9, -.2, 0.))

    def test_latched_handle_grasp_keeps_closing_only_while_on_handle(self) -> None:
        from tools.vla82_full_sim import expert

        contact = getattr(expert, "fixture_close_latched_handle_contact", None)
        self.assertIsNotNone(contact)
        self.assertFalse(contact(grasp_latched=False, exact_handle_contact=True))
        self.assertTrue(contact(grasp_latched=True, exact_handle_contact=True))
        self.assertFalse(contact(grasp_latched=True, exact_handle_contact=False))

    def test_vla028_handle_close_uses_a_long_enough_physical_rollout(self) -> None:
        from tools.vla82_full_sim import expert

        budget = getattr(expert, "fixture_close_step_budget", None)
        distance = getattr(expert, "fixture_close_effective_push_distance", None)
        self.assertIsNotNone(budget)
        self.assertIsNotNone(distance)
        self.assertGreaterEqual(budget("VLA82-028", default_steps=700), 1600)
        self.assertGreater(distance("VLA82-028", joint_type=3), .01)

    def test_fixture_close_converts_desired_pad_point_to_wrist_target(self) -> None:
        from tools.vla82_full_sim import expert

        np.testing.assert_allclose(expert.fixture_close_wrist_target_for_pad_contact(
            eef=(0., 0., 1.), pad_center=(.1, 0., 1.),
            desired_pad_center=(.5, 0., 1.),
        ), (.4, 0., 1.))
        self.assertGreaterEqual(expert.DrawerClosePrimitive.MAX_STEPS, 700)

    def test_handle_roll_aligns_opening_axis_perpendicular_before_approach(self) -> None:
        from tools.vla82_full_sim import expert

        command, error = expert.fixture_handle_roll_command(
            opening_axis_world=(0., -1., 0.),
            handle_axis_world=(0., 1., 0.),
            tool_axis_world=(0., 0., -1.),
            base_rotation=np.eye(3),
        )
        self.assertGreater(error, 1.5)
        self.assertAlmostEqual(float(np.linalg.norm(command)), .6)
        aligned_command, aligned_error = expert.fixture_handle_roll_command(
            opening_axis_world=(1., 0., 0.),
            handle_axis_world=(0., 1., 0.),
            tool_axis_world=(0., 0., -1.),
            base_rotation=np.eye(3),
        )
        self.assertLess(aligned_error, 1e-6)
        np.testing.assert_allclose(aligned_command, np.zeros(3), atol=1e-6)

    def test_handle_tool_alignment_points_fingers_toward_fixture_face(self) -> None:
        from tools.vla82_full_sim import expert

        command, error = expert.fixture_handle_tool_alignment_command(
            tool_axis_world=(0., 0., -1.),
            outward_normal_world=(1., 0., 0.),
            base_rotation=np.eye(3),
        )
        self.assertGreater(error, 1.5)
        self.assertAlmostEqual(float(np.linalg.norm(command)), .6)
        self.assertGreater(command[1], .59)
        aligned_command, aligned_error = expert.fixture_handle_tool_alignment_command(
            tool_axis_world=(-1., 0., 0.),
            outward_normal_world=(1., 0., 0.),
            base_rotation=np.eye(3),
        )
        self.assertLess(aligned_error, 1e-6)
        np.testing.assert_allclose(aligned_command, np.zeros(3), atol=1e-6)

    def test_handle_outward_normal_uses_hinge_plane_not_nearest_minor_axis(self) -> None:
        from tools.vla82_full_sim import expert

        normal = expert.fixture_handle_outward_normal(
            joint_axis_world=(0., 0., 1.),
            handle_world=(1., 0., 1.),
            joint_anchor_world=(0., 0., 1.),
            eef_world=(1., -1., 2.),
        )
        np.testing.assert_allclose(normal, (0., -1., 0.), atol=1e-6)

    def test_fixture_approach_base_follows_only_outside_arm_workspace(self) -> None:
        from tools.vla82_full_sim import expert

        command = expert.fixture_approach_base_local(
            pad_handle_distance=.37,
            handle_delta_world=(1., 0., .2),
            base_rotation=np.eye(3),
        )
        np.testing.assert_allclose(command, (.16, 0.), atol=1e-6)
        self.assertIsNone(expert.fixture_approach_base_local(
            pad_handle_distance=.12,
            handle_delta_world=(1., 0., .2),
            base_rotation=np.eye(3),
        ))

    def test_handle_safe_approach_enables_bounded_base_follow(self) -> None:
        from tools.vla82_full_sim import expert

        command = expert.fixture_handle_stage_base_follow(
            stage="safe_approach_above_handle", bypass_required=False,
            target_distance=.37, target_delta_world=(0., -1., 0.),
            base_rotation=np.eye(3),
        )
        np.testing.assert_allclose(command, (0., -.16), atol=1e-6)
        self.assertIsNone(expert.fixture_handle_stage_base_follow(
            stage="safe_approach_above_handle", bypass_required=True,
            target_distance=.37, target_delta_world=(0., -1., 0.),
            base_rotation=np.eye(3),
        ))

    def test_close_fridge_binding_excludes_closed_freezer_door(self) -> None:
        from tools.vla82_full_sim.environment import preferred_fixture_joint_names

        discovered = (
            "fridge_freezer0_door_joint",
            "fridge_fridge_left_door_joint",
            "fridge_fridge_right_door_joint",
        )
        fridge_doors = (
            "fridge_fridge_left_door_joint",
            "fridge_fridge_right_door_joint",
        )
        self.assertEqual(preferred_fixture_joint_names(
            task_class="CloseFridge",
            discovered_joint_names=discovered,
            fridge_door_joint_names=fridge_doors,
        ), fridge_doors)

    def test_cabinet_bottle_targets_pad_center_not_wrist_origin(self) -> None:
        eef = np.array((0.240, -4.294, 1.517))
        pads = np.array((0.241, -4.294, 1.517))
        obj = np.array((0.197, -4.296, 1.514))
        target = pick_place_pad_center_target(eef=eef, pad_center=pads, obj=obj)
        self.assertTrue(np.allclose(target, np.array((0.196, -4.296, 1.514))))

    def test_water_bottle_release_retreat_is_vertical_only(self) -> None:
        from tools.vla82_full_sim.expert import pick_place_release_retreat_target

        eef = np.array((0.555, -4.080, 1.090))
        obj = np.array((0.562, -4.080, 1.024))
        target = pick_place_release_retreat_target(
            "VLA82-017", eef=eef, object_center=obj,
        )
        np.testing.assert_allclose(target, np.array((0.555, -4.080, 1.210)), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
