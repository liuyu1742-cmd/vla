"""Closed-drawer control must be contact-gated and use only physics actions."""

from tools.vla82_full_sim.expert import drawer_close_stage


def test_drawer_close_requires_handle_contact_before_push():
    assert drawer_close_stage(exact_handle_contact=False, joint_position=-0.2877) == "approach_handle"


def test_drawer_close_pushes_after_handle_contact_until_closed():
    assert drawer_close_stage(exact_handle_contact=True, joint_position=-0.2877) == "push_close"


def test_drawer_close_keeps_pushing_until_strict_motion_requirement_is_met():
    assert drawer_close_stage(exact_handle_contact=True, joint_position=-0.10) == "push_close"


def test_drawer_close_stops_only_near_the_true_closed_limit():
    assert drawer_close_stage(exact_handle_contact=True, joint_position=-0.08) == "settle_closed"
