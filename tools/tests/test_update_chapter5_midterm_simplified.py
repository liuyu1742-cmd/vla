import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "update_chapter5_midterm_simplified.py"
SPEC = importlib.util.spec_from_file_location("update_chapter5_midterm_simplified", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_midterm_selection_excludes_complex_cooking_and_tool_sequences():
    selected = {obj for objects in MOD.MIDTERM_OBJECTS.values() for obj in objects}

    assert "削皮刀" not in selected
    assert "平底锅" not in selected
    assert "工具箱" not in selected
    assert "量杯" in selected
    assert "水瓶" in selected


def test_all_fifteen_section_texts_are_within_requested_length_range():
    assert len(MOD.SECTION_TEXT) == 15
    lengths = [MOD.section_length(text) for text in MOD.SECTION_TEXT.values()]
    assert all(350 <= length <= 500 for length in lengths)


def test_figure_plan_has_one_source_sample_for_each_task():
    assert set(MOD.FIGURE_SOURCES) == set(MOD.SECTION_TEXT)
