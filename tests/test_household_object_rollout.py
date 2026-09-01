import pytest


def test_build_instruction_uses_a_real_object_name_and_canonical_task():
    from tools.openvla_household_object_rollout import HouseholdObject, build_instruction

    spec = HouseholdObject(group="glass_cup", english_name="glass cup", chinese_name="glass cup")

    assert build_instruction(spec) == "pick up the glass cup and place it in the cabinet"


def test_household_object_rejects_blank_names():
    from tools.openvla_household_object_rollout import HouseholdObject

    with pytest.raises(ValueError, match="english_name"):
        HouseholdObject(group="glass_cup", english_name=" ", chinese_name="glass cup")


def test_validate_object_scale_preserves_verified_cup_scale_and_rejects_zero():
    from tools.openvla_household_object_rollout import validate_object_scale

    assert validate_object_scale(0.7) == 0.7
    with pytest.raises(ValueError, match="object_scale"):
        validate_object_scale(0.0)

def test_validate_grasp_height_offset_accepts_expert_height_and_rejects_negative():
    from tools.openvla_household_object_rollout import validate_grasp_height_offset

    assert validate_grasp_height_offset(0.08) == 0.08
    with pytest.raises(ValueError, match="grasp_height_offset"):
        validate_grasp_height_offset(-0.01)