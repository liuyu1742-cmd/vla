"""Deterministic stopping policy for OpenVLA-4L distillation rounds."""

from __future__ import annotations


ROUND_BETAS = (0.7, 0.3, 0.0)


def round_beta(round_index: int) -> float:
    if not 0 <= round_index < len(ROUND_BETAS):
        raise ValueError("round_index must be 0, 1, or 2")
    return ROUND_BETAS[round_index]


def next_action(
    *,
    pure_successes: int,
    round_index: int,
    max_rounds: int = 3,
) -> str:
    if pure_successes < 0:
        raise ValueError("pure_successes must be non-negative")
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if not 0 <= round_index < max_rounds:
        raise ValueError("round_index must be within max_rounds")
    if pure_successes > 0:
        return "accept_pure"
    if round_index + 1 < max_rounds:
        return "collect_next_round"
    return "run_hybrid"
