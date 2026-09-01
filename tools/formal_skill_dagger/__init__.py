"""JSON-safe active API for formal DAgger mixing and episode persistence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


_SOURCE = Path(__file__).resolve().parents[1] / "formal_skill_dagger.py"
_MODULE_NAME = "tools._formal_skill_dagger_base"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load formal DAgger source from {_SOURCE}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, _BASE)
_SPEC.loader.exec_module(_BASE)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_dagger_episode(*args: Any, **kwargs: Any):
    prepared = dict(kwargs)
    prepared["report"] = json_safe(prepared["report"])
    return _BASE.save_dagger_episode(*args, **prepared)


MixingResult = _BASE.MixingResult
choose_executed_action = _BASE.choose_executed_action
validate_training_seed = _BASE.validate_training_seed


__all__ = [
    "MixingResult",
    "choose_executed_action",
    "json_safe",
    "save_dagger_episode",
    "validate_training_seed",
]
