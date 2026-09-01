"""Windows-safe phase-balanced post-training pipeline."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from tools.formal_skill_posttrain_pipeline import process_exists


_SOURCE = Path(__file__).resolve().parents[1] / "formal_skill_posttrain_pipeline.py"
_TEXT = _SOURCE.read_text(encoding="utf-8")
_OLD = '"tools.openvla_formal_skill_tcp_server",'
_NEW = '"tools.openvla_formal_skill_tcp_server_balanced",'
if _TEXT.count(_OLD) != 1:
    raise ImportError("formal posttrain server substitution drifted")
_TEXT = _TEXT.replace(_OLD, _NEW)

_MODULE_NAME = "tools._formal_skill_phase_balanced_posttrain_pipeline_active"
_IMPLEMENTATION = types.ModuleType(_MODULE_NAME)
_IMPLEMENTATION.__file__ = str(_SOURCE)
_IMPLEMENTATION.__package__ = "tools"
sys.modules.setdefault(_MODULE_NAME, _IMPLEMENTATION)
exec(compile(_TEXT, str(_SOURCE), "exec"), _IMPLEMENTATION.__dict__)
_IMPLEMENTATION._process_exists = process_exists

build_pipeline_summary = _IMPLEMENTATION.build_pipeline_summary
main = _IMPLEMENTATION.main


__all__ = ["build_pipeline_summary", "main"]
