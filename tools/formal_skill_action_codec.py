"""Install the exact action normalization contract saved by formal training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ACTION_NORM_KEY = "task2_organizing_toy"
ACTION_DIMENSION = 7
STATS_FILENAME = "formal_skill_action_stats.json"


def load_formal_skill_action_stats(adapter_dir: Path) -> tuple[str, dict[str, Any]]:
    path = Path(adapter_dir) / STATS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"formal action statistics are missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = payload.get("key")
    if key != ACTION_NORM_KEY:
        raise ValueError(
            f"unexpected formal action normalization key: {key!r}; "
            f"expected {ACTION_NORM_KEY!r}"
        )
    action = payload.get("action")
    if not isinstance(action, dict):
        raise ValueError("formal action statistics must contain an action object")
    low = np.asarray(action.get("q01"), dtype=np.float64)
    high = np.asarray(action.get("q99"), dtype=np.float64)
    mask = np.asarray(action.get("mask"))
    expected = (ACTION_DIMENSION,)
    if low.shape != expected or high.shape != expected or mask.shape != expected:
        raise ValueError("formal action statistics must contain seven values per field")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("formal action quantiles must be finite")
    mask_bool = mask.astype(bool)
    if np.any(high[mask_bool] <= low[mask_bool]):
        raise ValueError("masked formal action quantiles must have positive ranges")
    validated = {
        "q01": low.tolist(),
        "q99": high.tolist(),
        "mask": mask_bool.tolist(),
    }
    return ACTION_NORM_KEY, validated


def install_formal_skill_action_stats(model: Any, adapter_dir: Path) -> str:
    key, action = load_formal_skill_action_stats(adapter_dir)
    model.norm_stats = {key: {"action": action}}
    return key


__all__ = [
    "ACTION_DIMENSION",
    "ACTION_NORM_KEY",
    "STATS_FILENAME",
    "install_formal_skill_action_stats",
    "load_formal_skill_action_stats",
]
