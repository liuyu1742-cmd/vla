"""Stateful release guard that separates gripper opening from retreat motion."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tools.formal_skill_rollout import guarded_action


class ReleaseHoldGuard:
    def __init__(
        self,
        *,
        hold_decisions: int = 18,
        base_guard: Callable[[Sequence[float], str], np.ndarray] = guarded_action,
    ) -> None:
        if hold_decisions < 1:
            raise ValueError("hold_decisions must be positive")
        self.hold_decisions = int(hold_decisions)
        self.base_guard = base_guard
        self._place_steps = 0

    def apply(self, raw_action: Sequence[float], phase: str) -> np.ndarray:
        action = self.base_guard(raw_action, phase)
        if phase != "place":
            self._place_steps = 0
            return action
        if self._place_steps < self.hold_decisions:
            self._place_steps += 1
            held = np.zeros(7, dtype=np.float32)
            held[6] = -1.0
            return held
        return action


__all__ = ["ReleaseHoldGuard"]
