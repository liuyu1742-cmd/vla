"""Windows compatibility loader for optional MuJoCo bundled plugins."""

from __future__ import annotations

import ctypes
import importlib
import os
import warnings
from pathlib import Path


def preload_mujoco():
    """Import MuJoCo while tolerating only failing bundled optional plugins."""
    if os.name != "nt":
        return importlib.import_module("mujoco")

    original_cdll = ctypes.CDLL
    skipped = []

    def guarded_cdll(name, *args, **kwargs):
        try:
            return original_cdll(name, *args, **kwargs)
        except OSError as error:
            path = Path(str(name))
            if path.parent.name == "plugin" and path.suffix.lower() == ".dll":
                skipped.append((str(path), str(error)))
                return None
            raise

    ctypes.CDLL = guarded_cdll
    try:
        module = importlib.import_module("mujoco")
    finally:
        ctypes.CDLL = original_cdll

    if skipped:
        warnings.warn(
            "Skipped optional MuJoCo plugin DLLs that failed to initialize: "
            + ", ".join(Path(path).name for path, _ in skipped),
            RuntimeWarning,
        )
    return module
