from tools.check_openvla_oft_environment import (
    OFFICIAL_FLAGS,
    required_checkpoint,
)


def test_official_suite_checkpoint_mapping():
    assert required_checkpoint("libero_spatial").endswith("libero-spatial")
    assert required_checkpoint("libero_object").endswith("libero-object")
    assert required_checkpoint("libero_goal").endswith("libero-goal")
    assert required_checkpoint("libero_10").endswith("libero-10")


def test_official_flags_keep_oft_recipe():
    assert OFFICIAL_FLAGS["use_l1_regression"] is True
    assert OFFICIAL_FLAGS["use_diffusion"] is False
    assert OFFICIAL_FLAGS["num_images_in_input"] == 2
    assert OFFICIAL_FLAGS["use_proprio"] is True
    assert OFFICIAL_FLAGS["num_open_loop_steps"] == 8

