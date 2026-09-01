"""Active formal rollout implementation with deployment-only fingerprinting."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


_source = Path(__file__).resolve().parents[1] / "formal_skill_rollout.py"
_spec = importlib.util.spec_from_file_location(
    "tools._formal_skill_rollout_legacy", _source
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load formal rollout implementation: {_source}")
_implementation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_implementation)


def adapter_fingerprint(adapter_dir: Path) -> str:
    root = Path(adapter_dir)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (not path.relative_to(root).parts or path.relative_to(root).parts[0] != "checkpoints")
    )
    if not files:
        raise FileNotFoundError(f"formal adapter has no deployment files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest()


_implementation._adapter_fingerprint = adapter_fingerprint
guarded_action = _implementation.guarded_action
next_canonical_phase = _implementation.next_canonical_phase
main = _implementation.main


__all__ = [
    "adapter_fingerprint",
    "guarded_action",
    "main",
    "next_canonical_phase",
]
