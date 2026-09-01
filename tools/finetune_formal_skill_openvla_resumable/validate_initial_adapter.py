"""Callable compatibility export for the DAgger initial-adapter validator."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from tools.finetune_formal_skill_openvla_dagger import (
    validate_initial_adapter as _validate,
)


class _CallableModule(types.ModuleType):
    def __call__(
        self,
        initial_adapter: Path | None,
        *,
        resume_checkpoint: Path | None,
    ) -> Path | None:
        return _validate(
            initial_adapter, resume_checkpoint=resume_checkpoint
        )


sys.modules[__name__].__class__ = _CallableModule
