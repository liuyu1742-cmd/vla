"""Deterministic material-aware friction for VLA82 collision geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


PROFILE_VERSION = "vla82-material-friction-v2"


@dataclass(frozen=True)
class FrictionProfile:
    version: str
    material: str
    source: str
    sliding: float
    torsional: float
    rolling: float

    @property
    def values(self) -> tuple[float, float, float]:
        return (self.sliding, self.torsional, self.rolling)


MATERIAL_FRICTION: dict[str, tuple[float, float, float]] = {
    "smooth_ceramic_glass": (0.40, 0.004, 0.0001),
    "metal": (0.45, 0.004, 0.0001),
    "plastic": (0.55, 0.005, 0.0002),
    "textured_plastic": (0.70, 0.006, 0.0002),
    "paper_cardboard": (0.65, 0.006, 0.0002),
    "wood": (0.82, 0.008, 0.0003),
    "rubber_sponge": (1.00, 0.012, 0.0005),
}


MATERIAL_BY_SELECTION: dict[str, str] = {
    "VLA82-001": "plastic",
    "VLA82-002": "rubber_sponge",
    "VLA82-003": "plastic",
    "VLA82-004": "textured_plastic",
    "VLA82-005": "metal",
    "VLA82-006": "wood",
    "VLA82-007": "metal",
    "VLA82-008": "metal",
    "VLA82-009": "metal",
    "VLA82-010": "metal",
    "VLA82-011": "smooth_ceramic_glass",
    "VLA82-012": "smooth_ceramic_glass",
    "VLA82-013": "smooth_ceramic_glass",
    "VLA82-014": "smooth_ceramic_glass",
    "VLA82-015": "metal",
    "VLA82-016": "paper_cardboard",
    "VLA82-017": "plastic",
    "VLA82-018": "paper_cardboard",
    "VLA82-019": "metal",
    "VLA82-020": "smooth_ceramic_glass",
    "VLA82-021": "plastic",
    "VLA82-022": "wood",
    "VLA82-023": "plastic",
    "VLA82-024": "plastic",
    "VLA82-025": "paper_cardboard",
    "VLA82-026": "smooth_ceramic_glass",
    "VLA82-027": "plastic",
    "VLA82-028": "metal",
    "VLA82-029": "metal",
    "VLA82-030": "wood",
    "VLA82-031": "wood",
    "VLA82-032": "metal",
    "VLA82-033": "metal",
    "VLA82-034": "metal",
    "VLA82-035": "metal",
    "VLA82-036": "smooth_ceramic_glass",
    "VLA82-037": "metal",
    "VLA82-038": "metal",
    "VLA82-039": "wood",
    "VLA82-040": "smooth_ceramic_glass",
    "VLA82-041": "metal",
    "VLA82-042": "plastic",
    "VLA82-043": "metal",
    "VLA82-044": "smooth_ceramic_glass",
    "VLA82-045": "paper_cardboard",
    "VLA82-046": "paper_cardboard",
    "VLA82-047": "plastic",
    "VLA82-048": "paper_cardboard",
    "VLA82-049": "metal",
    "VLA82-050": "metal",
    "VLA82-051": "wood",
    "VLA82-052": "plastic",
    "VLA82-053": "plastic",
    "VLA82-054": "plastic",
    "VLA82-055": "paper_cardboard",
    "VLA82-056": "plastic",
    "VLA82-057": "plastic",
    "VLA82-058": "plastic",
    "VLA82-059": "plastic",
    "VLA82-060": "plastic",
}

if set(MATERIAL_BY_SELECTION) != {f"VLA82-{index:03d}" for index in range(1, 61)}:
    raise RuntimeError("VLA82 material registry must cover the fixed 60 selections")


def profile_for(selection_id: str, semantic_class: str) -> FrictionProfile:
    """Return one deterministic profile; unknown diagnostics use plastic."""
    del semantic_class
    material = MATERIAL_BY_SELECTION.get(selection_id, "plastic")
    source = "selection_id" if selection_id in MATERIAL_BY_SELECTION else "semantic_fallback"
    return FrictionProfile(PROFILE_VERSION, material, source, *MATERIAL_FRICTION[material])


def format_friction(profile: FrictionProfile) -> str:
    """Format a profile for a MuJoCo MJCF ``friction`` attribute."""
    return " ".join(f"{value:.6g}" for value in profile.values)


def apply_object_friction(
    raw: Any,
    geom_names: Sequence[str],
    profile: FrictionProfile,
) -> dict[str, object]:
    """Apply a profile only to selected contact-enabled collision geoms."""
    model = raw.sim.model
    before: dict[str, list[float]] = {}
    after: dict[str, list[float]] = {}
    applied: list[str] = []
    values = np.asarray(profile.values, dtype=float)
    for name in geom_names:
        geom_id = int(model.geom_name2id(name))
        if geom_id < 0:
            continue
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        old = np.asarray(model.geom_friction[geom_id], dtype=float).copy()
        model.geom_friction[geom_id] = values
        before[name] = old.tolist()
        after[name] = values.tolist()
        applied.append(name)
    if not applied:
        raise ValueError("selected object has no contact-enabled collision geom")
    return {
        "version": profile.version,
        "material": profile.material,
        "material_source": profile.source,
        "geom_names": applied,
        "before": before,
        "after": after,
        "application_stage": "post_load_pre_step",
    }
