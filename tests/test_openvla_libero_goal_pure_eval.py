def test_build_openvla_prompt_uses_official_instruction_template():
    from tools.openvla_libero_goal_pure_eval import build_openvla_prompt

    assert build_openvla_prompt("Put the bowl on the stove") == (
        "In: What action should the robot take to put the bowl on the stove?\nOut:"
    )
