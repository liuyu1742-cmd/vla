"""Normalization and lightweight trainable components for RoboCasa OFT."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_NORM_KEY = "robocasa365_oft"


def quantile_stats(values: np.ndarray) -> dict[str, list[Any]]:
    """Calculate finite per-channel 1st/99th percentile statistics."""
    data = np.asarray(values, dtype=np.float32)
    if data.ndim < 2 or data.shape[0] < 1 or not np.isfinite(data).all():
        raise ValueError("values must be a non-empty finite array with channels")
    flattened = data.reshape(-1, data.shape[-1])
    low = np.quantile(flattened, 0.01, axis=0).astype(np.float32)
    high = np.quantile(flattened, 0.99, axis=0).astype(np.float32)
    mask = (high - low) > 1e-6
    return {
        "q01": low.tolist(),
        "q99": high.tolist(),
        "mask": mask.tolist(),
        "min": flattened.min(axis=0).astype(np.float32).tolist(),
        "max": flattened.max(axis=0).astype(np.float32).tolist(),
        "mean": flattened.mean(axis=0).astype(np.float32).tolist(),
        "std": flattened.std(axis=0).astype(np.float32).tolist(),
    }


def _stat_arrays(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    if low.ndim != 1 or high.shape != low.shape or mask.shape != low.shape:
        raise ValueError("normalization statistics must have aligned channel vectors")
    return low, high, mask


def normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Normalize active channels to [-1, 1], keeping constants at zero."""
    data = np.asarray(values, dtype=np.float32)
    low, high, mask = _stat_arrays(stats)
    if data.shape[-1] != len(low):
        raise ValueError("value channels do not match normalization statistics")
    result = np.zeros_like(data)
    result[..., mask] = np.clip(
        2.0 * (data[..., mask] - low[mask]) / (high[mask] - low[mask]) - 1.0,
        -1.0,
        1.0,
    )
    return result


def unnormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Map normalized values back to source units."""
    data = np.asarray(values, dtype=np.float32)
    low, high, mask = _stat_arrays(stats)
    if data.shape[-1] != len(low):
        raise ValueError("value channels do not match normalization statistics")
    result = np.zeros_like(data)
    result[..., mask] = (
        0.5 * (data[..., mask] + 1.0) * (high[mask] - low[mask]) + low[mask]
    )
    result[..., ~mask] = low[~mask]
    return result


def install_robocasa_stats(
    vla: Any,
    action_stats: dict[str, Any],
    proprio_stats: dict[str, Any],
    *,
    key: str = ACTION_NORM_KEY,
) -> str:
    """Install the RoboCasa action/proprio distribution on an OFT model."""
    if not key.strip():
        raise ValueError("normalization key must be non-empty")
    vla.norm_stats = {
        **getattr(vla, "norm_stats", {}),
        key: {"action": action_stats, "proprio": proprio_stats},
    }
    return key


def build_trainable_components(
    vla: Any,
    *,
    device: str = "cuda:0",
    action_head_state: dict[str, Any] | None = None,
    proprio_projector_state: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Freeze the VLA and create only the continuous RoboCasa components."""
    import torch
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    for parameter in vla.parameters():
        parameter.requires_grad_(False)
    vla.eval()
    dtype = torch.bfloat16
    action_head = L1RegressionActionHead(
        input_dim=int(vla.llm_dim), hidden_dim=int(vla.llm_dim), action_dim=7
    ).to(device=device, dtype=dtype)
    proprio_projector = ProprioProjector(
        llm_dim=int(vla.llm_dim), proprio_dim=8
    ).to(device=device, dtype=dtype)
    if action_head_state is not None:
        action_head.load_state_dict(action_head_state)
    if proprio_projector_state is not None:
        proprio_projector.load_state_dict(proprio_projector_state)
    action_head.train()
    proprio_projector.train()
    return action_head, proprio_projector
