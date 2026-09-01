"""Active formal expert collector with the explicit Panda gripper contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import tools.pick_place_oracle as _oracle_api
from tools.pick_place_oracle.open_gripper import OpenGripperPickPlaceOracle


# The source collector imports the public class once at module load. Bind its
# dependency to the corrected expert before loading it under a private name.
_oracle_api.PickPlaceOracle = OpenGripperPickPlaceOracle
_SOURCE = Path(__file__).resolve().parents[1] / "collect_formal_skill_expert.py"
_MODULE_NAME = "tools._collect_formal_skill_expert_base"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load formal expert collector from {_SOURCE}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, _BASE)
_SPEC.loader.exec_module(_BASE)

EXPECTED_ACTIONS = _BASE.EXPECTED_ACTIONS
EXPECTED_RELATION = _BASE.EXPECTED_RELATION
build_formal_report = _BASE.build_formal_report
build_parser = _BASE.build_parser
cabinet_waypoints = _BASE.cabinet_waypoints
main = _BASE.main
save_formal_episode = _BASE.save_formal_episode
validate_skill_ir_for_collection = _BASE.validate_skill_ir_for_collection


__all__ = [
    "EXPECTED_ACTIONS",
    "EXPECTED_RELATION",
    "build_formal_report",
    "build_parser",
    "cabinet_waypoints",
    "main",
    "save_formal_episode",
    "validate_skill_ir_for_collection",
]
