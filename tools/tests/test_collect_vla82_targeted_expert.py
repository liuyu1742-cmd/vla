import numpy as np

from tools.collect_vla82_targeted_expert import (
    apply_rotation_template,
    box_waypoints,
    configure_oracle_for_mapping,
    public_rotation_template_path,
)


def test_box_waypoints_for_open_container_descends_inside():
    points = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([0.0, 2.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    front, center, retreat = box_waypoints(points, interior_height=0.6)
    np.testing.assert_allclose(center, [1.0, 1.0, 0.6])
    assert front[2] > center[2]
    assert retreat[2] > center[2]


def test_box_waypoints_for_counter_uses_top_surface():
    points = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([0.0, 2.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    _, center, _ = box_waypoints(points, interior_height=1.02)
    np.testing.assert_allclose(center, [1.0, 1.0, 1.02])


class DummyOracle:
    LIFT_HEIGHT = 0.14
    APPROACH_HEIGHT = 0.18
    APPROACH_TOLERANCE = 0.025
    CONTACT_TOLERANCE = 0.028
    GRASP_HEIGHT_OFFSET = 0.010
    CLOSED_TRANSLATION_LIMIT = 0.50


def test_boxed_food_uses_reachable_contact_threshold():
    oracle = DummyOracle()
    configure_oracle_for_mapping(
        oracle, "PickPlaceCounterToCabinet", "boxed_food"
    )
    assert oracle.GRASP_HEIGHT_OFFSET == 0.045
    assert oracle.CONTACT_TOLERANCE == 0.025
    assert oracle.LIFT_HEIGHT == 0.070
    assert oracle.CLOSED_TRANSLATION_LIMIT == 0.08


def test_cabinet_source_accepts_front_reachable_grasp_pose():
    oracle = DummyOracle()
    configure_oracle_for_mapping(
        oracle, "PickPlaceCabinetToCounter", "canned_food"
    )
    assert oracle.APPROACH_TOLERANCE == 0.110
    assert oracle.CONTACT_TOLERANCE == 0.110


def test_bowl_grasps_at_rim_instead_of_inside_cavity():
    oracle = DummyOracle()
    configure_oracle_for_mapping(oracle, "PickPlaceCounterToSink", "bowl")
    assert oracle.GRASP_HEIGHT_OFFSET == 0.045
    assert oracle.CONTACT_TOLERANCE == 0.040


def test_fragile_grasps_move_slowly_and_lift_less():
    for task_class, object_group in [
        ("PickPlaceCounterToCabinet", "bar_soap"),
        ("PickPlaceFridgeDrawerToShelf", "apple"),
    ]:
        oracle = DummyOracle()
        configure_oracle_for_mapping(oracle, task_class, object_group)
        assert oracle.CLOSED_TRANSLATION_LIMIT == 0.08
        assert oracle.LIFT_HEIGHT <= 0.08


def test_tongs_use_slow_short_transit():
    oracle = DummyOracle()
    configure_oracle_for_mapping(oracle, "PickPlaceCounterToSink", "tongs")
    assert oracle.CLOSED_TRANSLATION_LIMIT == 0.08
    assert oracle.LIFT_HEIGHT == 0.06


def test_public_rotation_template_only_replaces_wrist_axes():
    action = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0])
    template = np.asarray(
        [[9.0, 9.0, 9.0, -0.4, 0.5, -0.6, 1.0]], dtype=np.float32
    )
    actual = apply_rotation_template(action, 0, template)
    np.testing.assert_allclose(actual[:3], action[:3])
    np.testing.assert_allclose(actual[3:6], template[0, 3:6])
    assert actual[6] == action[6]
    np.testing.assert_allclose(apply_rotation_template(action, 1, template), action)


def test_matching_public_rotation_template_is_selected_by_task_and_object():
    assert public_rotation_template_path(
        "PickPlaceCabinetToCounter", "canned_food"
    ).name == "episode_003740_actions.npy"
    assert public_rotation_template_path(
        "PickPlaceCounterToCabinet", "tupperware"
    ).name == "episode_004021_actions.npy"
    assert public_rotation_template_path(
        "PickPlaceCounterToSink", "tongs"
    ).name == "episode_004410_actions.npy"
