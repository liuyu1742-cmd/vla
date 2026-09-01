"""DAgger action mixing with an explicit gripper safety contract."""

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
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (7,):
        raise ValueError(f"OpenVLA action must have shape (7,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("OpenVLA action contains a non-finite value")
    return np.clip(array, -1.0, 1.0)


def choose_executed_action(
    policy_action: np.ndarray,
    oracle_decision: OracleDecision,
    *,
    beta: float,
    rng: RandomSource,
) -> MixingResult:
    """Choose the rollout action while always retaining the oracle label.

    ``beta`` is the probability of executing the full oracle action on movement
    phases. Gripper state is always gated to the oracle state even when policy
    translation/rotation is selected. This prevents premature close during
    approach and accidental release while transporting the cup.
    """

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between zero and one")
    policy = _action(policy_action)
    oracle = _action(oracle_decision.action)

    if oracle_decision.force_expert:
        return MixingResult(oracle.copy(), oracle.copy(), "oracle_forced", False)

    if rng.random() < beta:
        return MixingResult(oracle.copy(), oracle.copy(), "oracle_mixed", False)

    executed = policy.copy()
    gripper_gated = not np.isclose(executed[6], oracle[6], atol=1e-6)
    executed[6] = oracle[6]
    source = "policy_gated" if gripper_gated else "policy"
    return MixingResult(executed, oracle.copy(), source, gripper_gated)
