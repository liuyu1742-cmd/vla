"""Continue augmented formal-skill training from an audited PEFT adapter."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import tools.finetune_formal_skill_openvla_augmented as _augmented
import tools.finetune_formal_skill_openvla_resumable as _resumable
from tools.formal_skill_policy_prompt import build_policy_instruction


_SOURCE = Path(_resumable._SOURCE)
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
_RESUME_POSITION_OLD = """def resume_position(completed_updates: int, updates_per_epoch: int) -> tuple[int, int]:
    if completed_updates < 0 or updates_per_epoch < 1:
        raise ValueError("completed updates must be non-negative and epoch size positive")
    return divmod(completed_updates, updates_per_epoch)


"""
_RESUME_POSITION_NEW = _RESUME_POSITION_OLD + """def validate_initial_adapter(
    initial_adapter: Path | None,
    *,
    resume_checkpoint: Path | None,
) -> Path | None:
    if initial_adapter is None:
        return None
    if resume_checkpoint is not None:
        raise ValueError("initial adapter cannot be combined with resume checkpoint")
    path = Path(initial_adapter).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"initial adapter directory is missing: {path}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        candidate = path / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"initial adapter is missing {filename}: {path}")
    return path


"""
_CHECKPOINT_ARG_OLD = (
    '    parser.add_argument("--checkpoint-every", type=int, default=500)\n'
)
_CHECKPOINT_ARG_NEW = (
    _CHECKPOINT_ARG_OLD
    + '    parser.add_argument("--initial-adapter", type=Path)\n'
)
_CHECKPOINT_SUMMARY_OLD = (
    '    summary["resume_checkpoint"] = str(checkpoint.resolve()) if checkpoint else None\n'
)
_CHECKPOINT_SUMMARY_NEW = """    initial_adapter = validate_initial_adapter(
        args.initial_adapter, resume_checkpoint=checkpoint
    )
    summary["resume_checkpoint"] = str(checkpoint.resolve()) if checkpoint else None
    summary["initial_adapter"] = (
        str(initial_adapter) if initial_adapter is not None else None
    )
    summary["initial_adapter_sha256"] = (
        _sha256(initial_adapter / "adapter_model.safetensors")
        if initial_adapter is not None
        else None
    )
"""
_FRESH_OLD = "    if checkpoint is None:\n"
_FRESH_NEW = "    if checkpoint is None and initial_adapter is None:\n"
_CHECKPOINT_BRANCH_OLD = """    else:
        model = PeftModel.from_pretrained(
            base, str(checkpoint), is_trainable=True, local_files_only=True
        )
"""
_CHECKPOINT_BRANCH_NEW = """    elif checkpoint is not None:
        model = PeftModel.from_pretrained(
            base, str(checkpoint), is_trainable=True, local_files_only=True
        )
"""
_LOSS_TAIL_OLD = (
    '        loss_tail = [float(value) for value in state.get("loss_tail", [])][-100:]\n'
)
_LOSS_TAIL_NEW = _LOSS_TAIL_OLD + """    else:
        assert initial_adapter is not None
        model = PeftModel.from_pretrained(
            base,
            str(initial_adapter),
            is_trainable=True,
            local_files_only=True,
        )
        completed = 0
        loss_sum = 0.0
        loss_count = 0
        loss_tail = []
"""

for old in (
    _IMPORT_OLD,
    _SUMMARY_OLD,
    _ORDER_OLD,
    _RESUME_POSITION_OLD,
    _CHECKPOINT_ARG_OLD,
    _CHECKPOINT_SUMMARY_OLD,
    _FRESH_OLD,
    _CHECKPOINT_BRANCH_OLD,
    _LOSS_TAIL_OLD,
):
    if _TEXT.count(old) != 1:
        raise ImportError(
            f"formal DAgger trainer source substitution drifted: {old!r}"
        )

for old, new in (
    (_IMPORT_OLD, _IMPORT_NEW),
    (_SUMMARY_OLD, _SUMMARY_NEW),
    (_ORDER_OLD, _ORDER_NEW),
    (_RESUME_POSITION_OLD, _RESUME_POSITION_NEW),
    (_CHECKPOINT_ARG_OLD, _CHECKPOINT_ARG_NEW),
    (_CHECKPOINT_SUMMARY_OLD, _CHECKPOINT_SUMMARY_NEW),
    (_FRESH_OLD, _FRESH_NEW),
    (_CHECKPOINT_BRANCH_OLD, _CHECKPOINT_BRANCH_NEW),
    (_LOSS_TAIL_OLD, _LOSS_TAIL_NEW),
):
    _TEXT = _TEXT.replace(old, new)

_MODULE_NAME = "tools._finetune_formal_skill_openvla_dagger"
_IMPLEMENTATION = types.ModuleType(_MODULE_NAME)
_IMPLEMENTATION.__file__ = str(_SOURCE)
_IMPLEMENTATION.__package__ = "tools"
sys.modules.setdefault(_MODULE_NAME, _IMPLEMENTATION)
exec(compile(_TEXT, str(_SOURCE), "exec"), _IMPLEMENTATION.__dict__)

# Reuse the exact audited policy prompt, phase-balanced cached dataset, epoch
# ordering, appearance augmentation, and report metadata from augmented-r2.
_IMPLEMENTATION.build_instruction = build_policy_instruction
_IMPLEMENTATION.FormalSkillDataset = _augmented.AugmentedCachedPhaseDataset
_IMPLEMENTATION.epoch_sample_order = (
    _augmented._strict_augmented_epoch_sample_order
)
_IMPLEMENTATION.planned_training_summary = (
    _augmented._implementation.planned_training_summary
)

for _name in dir(_IMPLEMENTATION):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_IMPLEMENTATION, _name)

main = _IMPLEMENTATION.main


if __name__ == "__main__":
    raise SystemExit(main())
