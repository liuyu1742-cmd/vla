from tools.vla82_pure_closed_loop import stable_distractor_group


def test_stable_distractor_avoids_excluded_fruit_and_is_fridgable():
    assert stable_distractor_group(
        {"exclude_obj_groups": ("vegetable", "fruit"), "fridgable": True}
    ) == "milk"


def test_stable_distractor_keeps_simple_apple_default():
    assert stable_distractor_group({}) == "apple"
