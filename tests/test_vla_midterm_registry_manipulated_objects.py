from tools.build_vla_midterm_registry import build_registry


def test_selected_objects_are_the_manipulated_objects() -> None:
    registry = build_registry()
    object_groups = {row["object_group"] for row in registry["relations"]}

    assert all(row["manipulated_object_text"] for row in registry["relations"])
    assert {"spoon", "pan", "plate", "tray"}.isdisjoint(object_groups)
    assert {"ladle", "dish_brush", "onion", "squash"} <= object_groups
