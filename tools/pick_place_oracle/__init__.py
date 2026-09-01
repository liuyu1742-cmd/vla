"""Calibrated compatibility API for solid-object top grasps.

The state machine remains in the original module. This package applies the
RoboCasa-tested solid-object grasp height while the Windows editor helper cannot
update that module in place.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "pick_place_oracle.py"
_MODULE_NAME = "tools._pick_place_oracle_base"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load pick/place oracle implementation from {_SOURCE}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, _BASE)
_SPEC.loader.exec_module(_BASE)

PickPlaceDecision = _BASE.PickPlaceDecision
PickPlaceSnapshot = _BASE.PickPlaceSnapshot


class PickPlaceOracle(_BASE.PickPlaceOracle):
    """Generic oracle calibrated to the reachable center of a solid 6 cm block."""

    GRASP_HEIGHT_OFFSET = 0.025
    CONTACT_TOLERANCE = 0.028


__all__ = ["PickPlaceDecision", "PickPlaceOracle", "PickPlaceSnapshot"]
