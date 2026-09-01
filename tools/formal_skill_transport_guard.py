"""Calibrated close-hold, vertical-lift, transport, and release sequence."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tools.formal_skill_rollout import guarded_action


class CalibratedPickPlaceGuard:
    def __init__(
        self,
        *,
        grasp_hold_decisions: int = 18,
        lift_decisions: int = 27,
        release_hold_decisions: int = 18,
        base_guard: Callable[[Sequence[float], str], np.ndarray] = guarded_action,
    ) -> None:
        if min(grasp_hold_decisions, lift_decisions, release_hold_decisions) < 1:
            raise ValueError("all calibrated decision counts must be positive")
        self.grasp_hold_decisions = int(grasp_hold_decisions)
        self.lift_decisions = int(lift_decisions)
        self.release_hold_decisions = int(release_hold_decisions)
        self.base_guard = base_guard
        self._grasp_remaining = 0
        self._lift_remaining = 0
        self._place_steps = 0
        self._previous_phase: str | None = None

    @staticmethod
    def _stationary(gripper: float) -> np.ndarray:
        result = np.zeros(7, dtype=np.float32)
        result[6] = float(gripper)
        return result

    @staticmethod
    def _vertical_lift() -> np.ndarray:
        result = np.zeros(7, dtype=np.float32)
        result[2] = 0.30
        result[6] = 1.0
        return result

    def _cancel_transport(self) -> None:
        self._grasp_remaining = 0
        self._lift_remaining = 0

    def apply(self, raw_action: Sequence[float], phase: str) -> np.ndarray:
        action = self.base_guard(raw_action, phase)
        if phase == "place":
            self._cancel_transport()
            if self._place_steps < self.release_hold_decisions:
                self._place_steps += 1
                self._previous_phase = phase
                return self._stationary(-1.0)
            self._previous_phase = phase
            return action
        self._place_steps = 0

        if phase == "locate":
            self._cancel_transport()
        elif phase == "grasp" and self._previous_phase != "grasp":
            self._grasp_remaining = self.grasp_hold_decisions
            self._lift_remaining = self.lift_decisions
        elif phase not in {"grasp", "move"}:
            self._cancel_transport()

        if phase in {"grasp", "move"} and self._grasp_remaining > 0:
            self._grasp_remaining -= 1
            self._previous_phase = phase
            return self._stationary(1.0)
        if phase == "move" and self._lift_remaining > 0:
            self._lift_remaining -= 1
            self._previous_phase = phase
            return self._vertical_lift()

        self._previous_phase = phase
        return action


__all__ = ["CalibratedPickPlaceGuard"]
