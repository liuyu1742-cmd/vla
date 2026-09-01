"""Deterministic cross-episode sample order for resumable formal training."""

from __future__ import annotations

import numpy as np


def epoch_sample_order(
    sample_count: int,
    *,
    seed: int,
    epoch: int,
    start: int = 0,
) -> list[int]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if not 0 <= start <= sample_count:
        raise ValueError("start must be within the epoch")
    sequence = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), int(sample_count)])
    ).permutation(sample_count)
    return [int(value) for value in sequence[start:]]


__all__ = ["epoch_sample_order"]
