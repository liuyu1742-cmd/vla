"""Runtime-complete entry point for continued formal DAgger training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import tools.finetune_formal_skill_openvla_dagger as _dagger


def adapter_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_dagger._IMPLEMENTATION._sha256 = adapter_sha256


def implementation_has_adapter_hash() -> bool:
    return _dagger._IMPLEMENTATION.__dict__.get("_sha256") is adapter_sha256


for _name in dir(_dagger._IMPLEMENTATION):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_dagger._IMPLEMENTATION, _name)

_IMPLEMENTATION = _dagger._IMPLEMENTATION
main = _IMPLEMENTATION.main


if __name__ == "__main__":
    raise SystemExit(main())
