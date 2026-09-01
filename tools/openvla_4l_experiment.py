"""Build, train, evaluate, and document the project-adapted OpenVLA-4L.

The module keeps evidence provenance explicit: every aggregate is derived from
the JSON/JSONL records produced by a real local run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


MODES = ("full", "lora_r32", "last_layer_only", "frozen_vision")
FIVE_SHOT_SEEDS = (0, 1, 2, 3, 5)


def selected_layer_indices(total_layers: int, keep_layers: int) -> tuple[int, ...]:
    """Return the deterministic prefix retained by the reduced language model."""
    if not 0 < keep_layers <= total_layers:
        raise ValueError("keep_layers must be within the source depth")
    return tuple(range(keep_layers))


def is_trainable(name: str, mode: str, last_layer_index: int) -> bool:
    """Classify one exact OpenVLA parameter under a fine-tuning method."""
    lowered = name.lower()
    if mode == "full":
        return True
    if mode == "frozen_vision":
        return not lowered.startswith("vision_backbone.")
    if mode == "last_layer_only":
        return (
            f"language_model.model.layers.{last_layer_index}." in lowered
            or lowered.startswith("language_model.lm_head.")
        )
    if mode == "lora_r32":
        return False
    raise ValueError(f"unknown mode: {mode}")


def reduce_language_layers(model: Any, keep_layers: int) -> dict[str, Any]:
    """Retain a prefix of Llama layers and update both nested configurations."""
    layers = model.language_model.model.layers
    indices = selected_layer_indices(len(layers), keep_layers)
    selected = [layers[index] for index in indices]
    try:
        import torch

        replacement = torch.nn.ModuleList(selected) if isinstance(layers, torch.nn.ModuleList) else selected
    except ImportError:
        replacement = selected
    model.language_model.model.layers = replacement
    model.language_model.config.num_hidden_layers = keep_layers
    model.config.text_config.num_hidden_layers = keep_layers
    return {
        "source_layers": len(layers),
        "kept_layers": keep_layers,
        "selected_layer_indices": list(indices),
    }


def summarize_trainability(
    named_sizes: Iterable[tuple[str, int]],
    mode: str,
    last_layer_index: int,
) -> dict[str, int | str]:
    """Summarize a method without mutating model parameters."""
    rows = [(name, int(size)) for name, size in named_sizes]
    return {
        "mode": mode,
        "total_parameters": sum(size for _, size in rows),
        "trainable_parameters": sum(
            size
            for name, size in rows
            if is_trainable(name, mode, last_layer_index)
        ),
    }


def select_keyframe_steps(records: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    """Select deterministic start, midpoint, and terminal rollout steps."""
    if not records:
        raise ValueError("at least one rollout record is required")
    indexes = (0, len(records) // 2, len(records) - 1)
    return tuple(int(records[index]["step"]) for index in dict.fromkeys(indexes))


def success_rate(episodes: Sequence[dict[str, Any]]) -> float:
    """Compute a percentage strictly from concrete episode booleans."""
    if not episodes:
        raise ValueError("at least one episode is required")
    if any(type(row.get("success")) is not bool for row in episodes):
        raise ValueError("every episode must contain a boolean success")
    return 100.0 * sum(row["success"] for row in episodes) / len(episodes)

