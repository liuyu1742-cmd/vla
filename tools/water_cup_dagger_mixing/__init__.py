"""DAgger mixing API with phase-specific contact-before-close gating.

This package form is the canonical import target. It keeps gripper safety
minimal and auditable: only a premature close during approach/descent is
overridden; forced oracle phases still replace the complete action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tools.water_cup_dagger_oracle import OracleDecision


class RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True)
class MixingResult:
    executed_action: np.ndarray
    oracle_label: np.ndarray
    source: str
    gripper_gated: bool


def _action(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (7,):
        raise ValueError(f"OpenVLA action must have shape (7,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("OpenVLA action contains a non-finite value")
    return np.clip(result, -1.0, 1.0)


def choose_executed_action(
    policy_action: np.ndarray,
    oracle_decision: OracleDecision,
    *,
    beta: float,
    rng: RandomSource,
) -> MixingResult:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between zero and one")
    policy = _action(policy_action)
    oracle = _action(oracle_decision.action)

    if oracle_decision.force_expert:
        return MixingResult(oracle.copy(), oracle.copy(), "oracle_forced", False)
    if rng.random() < beta:
        return MixingResult(oracle.copy(), oracle.copy(), "oracle_mixed", False)

    executed = policy.copy()
    premature_close = (
        oracle_decision.phase in {"approach_object", "descend_to_object"}
        and executed[6] > 0.5
    )
    if premature_close:
        executed[6] = 0.0
    return MixingResult(
        executed,
        oracle.copy(),
        "policy_gated" if premature_close else "policy",
        premature_close,
    )


__all__ = ["MixingResult", "choose_executed_action"]
