"""Map validated human-video object semantics to supported RoboCasa tasks."""

from __future__ import annotations


_PICK_PLACE_OBJECTS = {
    "cup": {"task": "PickPlaceCounterToCabinet", "object_group": "glass_cup"},
}


def map_pick_place(object_name: str) -> dict[str, str]:
    """Return a simulator task mapping for one supported pick-and-place object."""
    key = object_name.strip().lower()
    if key not in _PICK_PLACE_OBJECTS:
        raise ValueError(f"unsupported pick-and-place object: {object_name}")
    return dict(_PICK_PLACE_OBJECTS[key])
