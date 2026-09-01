"""Active resumable trainer with deterministic cross-episode shuffling.

The original module remains intact for provenance. This compatibility package
loads it with the three audited ordering substitutions required on Windows.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "finetune_formal_skill_openvla_resumable.py"
_TEXT = _SOURCE.read_text(encoding="utf-8")
_IMPORT_OLD = "from tools.formal_skill_dataset import FormalSkillDataset\n"
_IMPORT_NEW = (
    _IMPORT_OLD
    + "from tools.formal_skill_training_order import epoch_sample_order\n"
)
_SUMMARY_OLD = '            "trainer": "checkpointed_resumable_v1",\n'
_SUMMARY_NEW = (
    '            "trainer": "checkpointed_resumable_shuffled_v2",\n'
    '            "data_order": "deterministic cross-episode permutation per epoch",\n'
)
_ORDER_OLD = """        sample_start = min(first_update * args.batch_size, len(dataset))
        epoch_data = Subset(demonstrations, range(sample_start, len(dataset)))
"""
_ORDER_NEW = """        sample_start = min(first_update * args.batch_size, len(dataset))
        epoch_data = Subset(
            demonstrations,
            epoch_sample_order(
                len(dataset),
                seed=args.seed,
                epoch=epoch,
                start=sample_start,
            ),
        )
"""
for needle in (_IMPORT_OLD, _SUMMARY_OLD, _ORDER_OLD):
    if _TEXT.count(needle) != 1:
        raise ImportError(f"resumable trainer compatibility substitution drifted: {needle!r}")
_TEXT = _TEXT.replace(_IMPORT_OLD, _IMPORT_NEW)
_TEXT = _TEXT.replace(_SUMMARY_OLD, _SUMMARY_NEW)
_TEXT = _TEXT.replace(_ORDER_OLD, _ORDER_NEW)

_MODULE_NAME = "tools._finetune_formal_skill_openvla_resumable_shuffled"
_IMPLEMENTATION = types.ModuleType(_MODULE_NAME)
_IMPLEMENTATION.__file__ = str(_SOURCE)
_IMPLEMENTATION.__package__ = "tools"
sys.modules.setdefault(_MODULE_NAME, _IMPLEMENTATION)
exec(compile(_TEXT, str(_SOURCE), "exec"), _IMPLEMENTATION.__dict__)

for _name in dir(_IMPLEMENTATION):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_IMPLEMENTATION, _name)

main = _IMPLEMENTATION.main

