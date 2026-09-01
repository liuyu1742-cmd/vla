"""Small, testable scheduling helpers for water-cup imitation training."""

from __future__ import annotations

import math


def training_updates(samples: int, batch_size: int, epochs: int) -> int:
    """Return optimizer updates after visiting every sample in each epoch."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    return math.ceil(samples / batch_size) * epochs
