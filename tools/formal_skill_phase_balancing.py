"""Deterministic equal-quota phase sampling for formal skill training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


REQUIRED_PHASES = ("locate", "grasp", "move", "place")


def balanced_phase_epoch_order(
    phases: Sequence[str],
    *,
    seed: int,
    epoch: int,
    start: int = 0,
) -> list[int]:
    if not phases:
        raise ValueError("phases must be non-empty")
    if epoch < 0 or not 0 <= start <= len(phases):
        raise ValueError("invalid epoch or resume offset")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, phase in enumerate(phases):
        groups[str(phase)].append(index)
    missing = [phase for phase in REQUIRED_PHASES if not groups[phase]]
    if missing:
        raise ValueError(f"training split is missing phases: {missing}")

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), len(phases), 0xB41A])
    )
    base, remainder = divmod(len(phases), len(REQUIRED_PHASES))
    order: list[int] = []
    for phase_index, phase in enumerate(REQUIRED_PHASES):
        quota = base + (1 if phase_index < remainder else 0)
        source = np.asarray(groups[phase], dtype=np.int64)
        chosen = rng.choice(source, size=quota, replace=quota > source.size)
        order.extend(int(value) for value in chosen)
    rng.shuffle(order)
    return order[start:]


__all__ = ["REQUIRED_PHASES", "balanced_phase_epoch_order"]
