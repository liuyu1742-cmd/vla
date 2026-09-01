"""Detect closed-loop locate stagnation before enabling state-aware recovery."""

from __future__ import annotations

from collections import deque


class LocateStagnationRecovery:
    """Latch recovery when a locate window fails to make enough progress."""

    def __init__(
        self, *, window_decisions: int = 20, minimum_progress: float = 0.01
    ) -> None:
        if window_decisions < 1:
            raise ValueError("window_decisions must be positive")
        if minimum_progress <= 0:
            raise ValueError("minimum_progress must be positive")
        self.window_decisions = int(window_decisions)
        self.minimum_progress = float(minimum_progress)
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


__all__ = ["LocateStagnationRecovery"]
