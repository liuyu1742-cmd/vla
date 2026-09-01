"""Stateful gripper timing constraints for robust pick-place execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tools.formal_skill_rollout import guarded_action


class PickPlaceExecutionGuard:
    def __init__(
        self,
        *,
        grasp_hold_decisions: int = 18,
        release_hold_decisions: int = 18,
        base_guard: Callable[[Sequence[float], str], np.ndarray] = guarded_action,
    ) -> None:
        if grasp_hold_decisions < 1 or release_hold_decisions < 1:
            raise ValueError("grasp and release hold decisions must be positive")
        self.grasp_hold_decisions = int(grasp_hold_decisions)
        self.release_hold_decisions = int(release_hold_decisions)
        self.base_guard = base_guard
        self._grasp_remaining = 0
        self._place_steps = 0
        self._previous_phase: str | None = None

    @staticmethod
    def _stationary(gripper: float) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[6] = float(gripper)
        return action

    def apply(self, raw_action: Sequence[float], phase: str) -> np.ndarray:
        action = self.base_guard(raw_action, phase)

        if phase == "place":
            self._grasp_remaining = 0
            if self._place_steps < self.release_hold_decisions:
                self._place_steps += 1
                self._previous_phase = phase
                return self._stationary(-1.0)
            self._previous_phase = phase
            return action
        self._place_steps = 0

        if phase == "locate":
            self._grasp_remaining = 0
        elif phase == "grasp" and self._previous_phase != "grasp":
            self._grasp_remaining = self.grasp_hold_decisions
        elif phase not in {"grasp", "move"}:
            self._grasp_remaining = 0

        if phase in {"grasp", "move"} and self._grasp_remaining > 0:
            self._grasp_remaining -= 1
            self._previous_phase = phase
            return self._stationary(1.0)

        self._previous_phase = phase
        return action


__all__ = ["PickPlaceExecutionGuard"]
