"""Phase-balanced OpenVLA training with deterministic appearance augmentation."""

from __future__ import annotations

from typing import Any

import tools.finetune_formal_skill_openvla_phase_balanced as _phase_balanced
import tools.finetune_formal_skill_openvla_resumable as _trainer
from tools.formal_skill_visual_augmentation import augment_training_frame


_implementation = _trainer._IMPLEMENTATION
_base_dataset = _implementation.FormalSkillDataset
_original_epoch_order = _implementation.epoch_sample_order
_active_seed = 23
_active_epoch = 0


def set_augmentation_context(*, seed: int, epoch: int) -> None:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    global _active_seed, _active_epoch
    _active_seed = int(seed)
    _active_epoch = int(epoch)


class AugmentedCachedPhaseDataset(_base_dataset):
    """Apply one label-safe, deterministic appearance variant per epoch."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["frame"] = augment_training_frame(
            sample["frame"],
            seed=_active_seed,
            epoch=_active_epoch,
            sample_index=int(index),
        )
        return sample


def augmented_epoch_sample_order(
    sample_count: int,
    *,
    seed: int,
    epoch: int,
    start: int = 0,
) -> list[int]:
    """Set the augmentation epoch and return the audited phase-balanced order."""

    set_augmentation_context(seed=seed, epoch=epoch)
    if len(_phase_balanced._active_phases) != sample_count:
        # This branch keeps the context hook independently testable.  The
        # production trainer uses the strict wrapper below and never masks a
        # phase-cache mismatch.
        return list(range(start, sample_count))
    return _original_epoch_order(
        sample_count, seed=seed, epoch=epoch, start=start
    )


def _strict_augmented_epoch_sample_order(
    sample_count: int,
    *,
    seed: int,
    epoch: int,
    start: int = 0,
) -> list[int]:
    set_augmentation_context(seed=seed, epoch=epoch)
    return _original_epoch_order(
        sample_count, seed=seed, epoch=epoch, start=start
    )


_original_summary = _implementation.planned_training_summary


def _augmented_summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
    summary = _original_summary(*args, **kwargs)
    summary.update(
        {
            "visual_augmentation": (
                "epoch0=original; later epochs=deterministic "
                "brightness/contrast/saturation/sensor-noise"
            ),
            "geometric_augmentation": False,
            "augmentation_label_policy": "world-frame action unchanged",
        }
    )
    return summary


_implementation.FormalSkillDataset = AugmentedCachedPhaseDataset
_implementation.epoch_sample_order = _strict_augmented_epoch_sample_order
_implementation.planned_training_summary = _augmented_summary
main = _trainer.main


if __name__ == "__main__":
    raise SystemExit(main())
