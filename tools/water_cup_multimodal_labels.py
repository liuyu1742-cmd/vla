"""Label alignment helpers for Prismatic/OpenVLA multimodal sequences."""

from __future__ import annotations

import torch


def expand_labels_for_visual_tokens(labels: torch.Tensor, logits_length: int) -> torch.Tensor:
    """Insert ignored labels where OpenVLA inserts patch tokens after BOS."""
    visual_tokens = logits_length - labels.shape[1]
    if visual_tokens < 0:
        raise ValueError("logits sequence cannot be shorter than text labels")
    ignored = torch.full(
        (labels.shape[0], visual_tokens),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([labels[:, :1], ignored, labels[:, 1:]], dim=1)
