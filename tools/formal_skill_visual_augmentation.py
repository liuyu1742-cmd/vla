"""Deterministic appearance augmentation for formal-skill camera frames."""

from __future__ import annotations

import numpy as np


def augment_training_frame(
    frame: np.ndarray,
    *,
    seed: int,
    epoch: int,
    sample_index: int,
) -> np.ndarray:
    """Keep epoch 0 exact and vary only appearance in later epochs.

    The augmentation intentionally avoids flips, rotations, and spatial crops:
    those operations can invalidate the world-frame action label.  Brightness,
    contrast, saturation, and sensor noise improve texture/lighting robustness
    while preserving the demonstrated robot action.
    """

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("training frame must have shape (H,W,3)")
    if image.dtype != np.uint8:
        raise ValueError("training frame must use uint8 pixels")
    if epoch < 0 or sample_index < 0:
        raise ValueError("epoch and sample_index must be non-negative")
    if epoch == 0:
        return image.copy()

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(epoch), int(sample_index), 0xA11CE]
        )
    )
    values = image.astype(np.float32) / 255.0
    brightness = float(rng.uniform(0.82, 1.18))
    contrast = float(rng.uniform(0.82, 1.18))
    saturation = float(rng.uniform(0.82, 1.18))

    values *= brightness
    channel_mean = values.mean(axis=(0, 1), keepdims=True)
    values = (values - channel_mean) * contrast + channel_mean
    gray = (
        0.299 * values[..., 0:1]
        + 0.587 * values[..., 1:2]
        + 0.114 * values[..., 2:3]
    )
    values = gray + saturation * (values - gray)
    values += rng.normal(0.0, 0.008, size=values.shape).astype(np.float32)
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


__all__ = ["augment_training_frame"]
