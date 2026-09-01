"""Fixture joint evidence must include the physical handle used by the controller."""

from tools.vla82_full_sim.environment import fixture_contact_geometry_names


def test_drawer_contact_binding_includes_selected_handle_parts():
    geoms = fixture_contact_geometry_names(
        semantic_geom="stack_2_main_group_2_door_g0",
        all_geom_names=(
            "stack_2_main_group_2_door_g0",
            "stack_2_main_group_2_door_g1",
            "stack_2_main_group_2_door_handle_g3",
            "stack_2_main_group_2_inner_bottom",
        ),
    )
    assert geoms == (
        "stack_2_main_group_2_door_g0",
        "stack_2_main_group_2_door_g1",
        "stack_2_main_group_2_door_handle_g3",
    )


def test_named_rack_semantic_does_not_expand_to_whole_appliance():
    geoms = fixture_contact_geometry_names(
        semantic_geom="toaster_oven_main_group_reg_rack0",
        all_geom_names=(
            "toaster_oven_main_group_reg_main",
            "toaster_oven_main_group_door_main",
            "toaster_oven_main_group_reg_rack0",
            "toaster_oven_main_group_reg_tray0",
        ),
    )
    # A named semantic geom has no generated sibling stem to expand.  The
    # capture builder will retain all collision geoms on this exact body.
    assert geoms == ()
