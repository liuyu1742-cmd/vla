"""Stateful safety supervisor for OpenVLA low-level actions.

OpenVLA remains the primary action source. The supervisor replaces an action
only when it violates the current contact/gripper state or departs materially
from the recoverable Cartesian command. Simulator state is used for this first
end-to-end milestone; a real deployment must supply the same state contract
from perception and robot proprioception.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.water_cup_dagger_oracle import OracleDecision


@dataclass(frozen=True)
class GuardedAction:
    action: np.ndarray
    source: str
    translation_error: float


def _action(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (7,):
        raise ValueError(f"expected a seven-dimensional action, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("action contains non-finite values")
    return np.clip(result, -1.0, 1.0)


def supervise_action(
    policy_action: np.ndarray,
    oracle: OracleDecision,
    *,
    max_translation_error: float = 0.15,
) -> GuardedAction:
    if max_translation_error < 0:
        raise ValueError("max_translation_error must be non-negative")
    policy = _action(policy_action)
    recovery = _action(oracle.action)
    translation_error = float(np.max(np.abs(policy[:3] - recovery[:3])))

    if oracle.force_expert:
        return GuardedAction(recovery, "supervisor_forced", translation_error)
    policy_closed = bool(policy[6] > 0.5)
    recovery_closed = bool(recovery[6] > 0.5)
    if policy_closed != recovery_closed:
        return GuardedAction(recovery, "supervisor_gripper", translation_error)
    if translation_error > max_translation_error:
        return GuardedAction(recovery, "supervisor_translation", translation_error)
    return GuardedAction(policy, "openvla", translation_error)
