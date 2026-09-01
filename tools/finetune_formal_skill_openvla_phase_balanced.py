"""Train with canonical policy text and equal locate/grasp/move/place quotas."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

import tools.finetune_formal_skill_openvla_balanced  # applies policy prompt
import tools.finetune_formal_skill_openvla_resumable as _trainer
from tools.formal_skill_phase_balancing import balanced_phase_epoch_order


_implementation = _trainer._IMPLEMENTATION
_base_dataset = _implementation.FormalSkillDataset
_active_phases: list[str] = []


class CachedPhaseDataset(_base_dataset):
    """Cache all 12 compact episodes so random balanced access does not re-decompress."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._episode_caches: dict[int, dict[str, np.ndarray]] = {}
        phases: list[str] = []
        for episode_index, entry in enumerate(self.episodes):
            with np.load(entry["episode"], allow_pickle=False) as episode:
                cache = {
                    "frames": np.asarray(episode["frames"], dtype=np.uint8),
                    "actions": np.asarray(episode["actions"], dtype=np.float32),
                    "phases": np.asarray(episode["phases"]),
                    "canonical_phases": np.asarray(episode["canonical_phases"]),
                }
            self._episode_caches[episode_index] = cache
        for episode_index, frame_index in self.index:
            phases.append(
                str(self._episode_caches[episode_index]["canonical_phases"][frame_index])
            )
        global _active_phases
        _active_phases = phases

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        return self._episode_caches[episode_index]


def _balanced_order(
    sample_count: int,
    *,
    seed: int,
    epoch: int,
    start: int = 0,
) -> list[int]:
    if len(_active_phases) != sample_count:
        raise RuntimeError("phase cache does not match the formal training split")
    return balanced_phase_epoch_order(
        _active_phases, seed=seed, epoch=epoch, start=start
    )


_original_summary = _implementation.planned_training_summary


def _phase_summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
    summary = _original_summary(*args, **kwargs)
    summary.update(
        {
            "phase_sampling": "equal quota with deterministic minority oversampling",
            "phase_sampling_target": {
                "locate": 0.25,
                "grasp": 0.25,
                "move": 0.25,
                "place": 0.25,
            },
            "episode_cache": "all training episodes in host RAM",
        }
    )
    return summary


_implementation.FormalSkillDataset = CachedPhaseDataset
_implementation.epoch_sample_order = _balanced_order
_implementation.planned_training_summary = _phase_summary
main = _trainer.main


if __name__ == "__main__":
    raise SystemExit(main())
