"""Finite-state planning for transferred household skills."""

from __future__ import annotations


PROMPTS = {
    "pick": "pick up the glass cup",
    "place": "place the glass cup in the cabinet",
}


def next_phase(current: str, predicates: dict[str, bool]) -> str:
    """Advance only when the simulator supplies the required predicate."""
    if current == "pick" and predicates.get("grasped", False):
        return "place"
    return current
