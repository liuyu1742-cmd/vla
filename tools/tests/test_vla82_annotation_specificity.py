"""The executable phase list must not broaden a concrete source instruction."""

from tools.vla82_full_sim.annotations import parse_ordered_phases, select_source_annotation


def test_concrete_instruction_overrides_ambiguous_operation_label():
    text, kind = select_source_annotation(
        operation="拉开/关闭", instruction="Close the left drawer.",
    )
    assert text == "Close the left drawer."
    assert kind == "instruction_json_specific"


def test_operation_remains_authoritative_when_it_is_not_ambiguous():
    text, kind = select_source_annotation(
        operation="清洁表面", instruction="", 
    )
    assert text == "清洁表面"
    assert kind == "operation_json"


def test_left_drawer_uses_runtime_slide_joint_predicate_token():
    from tools.vla82_full_sim.predicates import _FIXTURE_TOKENS
    assert _FIXTURE_TOKENS["VLA82-030"] == "slidejoint"


def test_overlapping_chinese_place_synonyms_compile_to_one_physical_phase():
    assert parse_ordered_phases("摆放至键盘旁") == ("place",)


def test_two_non_overlapping_place_occurrences_remain_two_phases():
    assert parse_ordered_phases("先摆放鼠标，再放至键盘旁") == ("place", "place")
