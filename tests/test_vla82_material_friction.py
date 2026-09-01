from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_first_batch_profiles_are_material_specific() -> None:
    from tools.vla82_full_sim.friction import PROFILE_VERSION, profile_for

    expected = {
        "VLA82-006": ("wood", (0.82, 0.008, 0.0003)),
        "VLA82-014": ("smooth_ceramic_glass", (0.40, 0.004, 0.0001)),
        "VLA82-018": ("paper_cardboard", (0.65, 0.006, 0.0002)),
        "VLA82-037": ("metal", (0.45, 0.004, 0.0001)),
        "VLA82-041": ("metal", (0.45, 0.004, 0.0001)),
        "VLA82-045": ("paper_cardboard", (0.65, 0.006, 0.0002)),
    }

    for selection_id, (material, values) in expected.items():
        profile = profile_for(selection_id, "")
        assert profile.version == PROFILE_VERSION
        assert profile.material == material
        assert profile.source == "selection_id"
        assert profile.values == values


def test_registry_has_explicit_material_for_all_sixty_objects() -> None:
    from tools.vla82_full_sim.friction import MATERIAL_BY_SELECTION

    assert set(MATERIAL_BY_SELECTION) == {f"VLA82-{index:03d}" for index in range(1, 61)}


def test_unknown_object_uses_declared_semantic_fallback() -> None:
    from tools.vla82_full_sim.friction import profile_for

    profile = profile_for("VLA82-999", "unknown utensil")

    assert profile.material == "plastic"
    assert profile.source == "semantic_fallback"
    assert profile.values == (0.55, 0.005, 0.0002)


def test_apply_object_friction_changes_only_selected_contact_geoms() -> None:
    from tools.vla82_full_sim.friction import apply_object_friction, profile_for

    class FakeModel:
        def __init__(self) -> None:
            self._ids = {"obj_visual": 0, "obj_collision": 1, "robot_collision": 2}
            self.geom_contype = np.asarray((0, 1, 1), dtype=int)
            self.geom_conaffinity = np.asarray((0, 1, 1), dtype=int)
            self.geom_friction = np.asarray(
                ((0.95, 0.3, 0.1), (0.95, 0.3, 0.1), (1.0, 0.005, 0.0001)),
                dtype=float,
            )

        def geom_name2id(self, name: str) -> int:
            return self._ids.get(name, -1)

    model = FakeModel()
    raw = SimpleNamespace(sim=SimpleNamespace(model=model))

    evidence = apply_object_friction(
        raw,
        ("obj_visual", "obj_collision"),
        profile_for("VLA82-006", "rolling pin"),
    )

    assert evidence["geom_names"] == ["obj_collision"]
    assert evidence["before"]["obj_collision"] == [0.95, 0.3, 0.1]
    assert evidence["after"]["obj_collision"] == [0.82, 0.008, 0.0003]
    assert model.geom_friction[0].tolist() == [0.95, 0.3, 0.1]
    assert model.geom_friction[2].tolist() == [1.0, 0.005, 0.0001]


def test_apply_object_friction_rejects_missing_collision_geom() -> None:
    import pytest

    from tools.vla82_full_sim.friction import apply_object_friction, profile_for

    model = SimpleNamespace(
        geom_name2id=lambda _name: 0,
        geom_contype=np.asarray((0,), dtype=int),
        geom_conaffinity=np.asarray((0,), dtype=int),
        geom_friction=np.asarray(((0.95, 0.3, 0.1),), dtype=float),
    )
    raw = SimpleNamespace(sim=SimpleNamespace(model=model))

    with pytest.raises(ValueError, match="no contact-enabled collision geom"):
        apply_object_friction(raw, ("obj_visual",), profile_for("VLA82-006", "rolling pin"))
