"""State-aware recovery for closed-loop locate stagnation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from tools.formal_skill_progress_guard import (
    GuardDecision,
    select_progress_guarded_action,
)


class LocateStagnationRecovery:
    """Latch safe recovery when a locate window makes insufficient progress."""

    def __init__(
        self,
        *,
        window_decisions: int = 20,
        minimum_progress: float = 0.01,
        model_weight: float = 0.15,
    ) -> None:
        if window_decisions < 1:
            raise ValueError("window_decisions must be positive")
        if minimum_progress <= 0:
            raise ValueError("minimum_progress must be positive")
        if not 0.0 <= model_weight <= 1.0:
            raise ValueError("model_weight must be in [0, 1]")
        self.window_decisions = int(window_decisions)
        self.minimum_progress = float(minimum_progress)
        self.model_weight = float(model_weight)
        self._distances: deque[float] = deque(
            maxlen=self.window_decisions + 1
        )
        self.active = False

    def reset(self) -> None:
        self._distances.clear()
        self.active = False

    def observe(self, phase: str, distance: float) -> bool:
        if phase != "locate":
            self.reset()
            return False
        self._distances.append(float(distance))
        if self.active:
            return True
        if len(self._distances) < self._distances.maxlen:
            return False
        progress = self._distances[0] - min(list(self._distances)[1:])
        if progress < self.minimum_progress:
            self.active = True
        return self.active

    def select_action(
        self,
        raw_action: Sequence[float],
        expert_action: Sequence[float],
        *,
        force_expert: bool,
    ) -> GuardDecision:
        if not self.active:
            raise RuntimeError("locate recovery is not active")
        return select_progress_guarded_action(
            raw_action,
            expert_action,
            force_expert=force_expert,
            model_weight=self.model_weight,
        )


__all__ = ["LocateStagnationRecovery"]
