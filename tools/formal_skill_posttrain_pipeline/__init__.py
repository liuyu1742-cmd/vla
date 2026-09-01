"""Active post-training pipeline with a signal-free Windows PID probe."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path


_source = Path(__file__).resolve().parents[1] / "formal_skill_posttrain_pipeline.py"
_spec = importlib.util.spec_from_file_location(
    "tools._formal_skill_posttrain_pipeline_legacy", _source
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load formal post-training pipeline: {_source}")
_implementation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_implementation)


def process_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(process_query_limited_information, 0, int(pid))
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return int(exit_code.value) == still_active
    finally:
        kernel32.CloseHandle(handle)


_implementation._process_exists = process_exists
build_pipeline_summary = _implementation.build_pipeline_summary
main = _implementation.main


__all__ = ["build_pipeline_summary", "main", "process_exists"]
