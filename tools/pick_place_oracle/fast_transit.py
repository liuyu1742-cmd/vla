"""Measured grasp geometry with bounded but practical closed-gripper transit."""

from __future__ import annotations

import numpy as np

from tools.pick_place_oracle import PickPlaceDecision, PickPlaceSnapshot
from tools.pick_place_oracle.open_gripper import OpenGripperPickPlaceOracle


class FastTransitPickPlaceOracle(OpenGripperPickPlaceOracle):
    """Raise closed transit from 0.2 to 0.5 while retaining full grasp closure."""

    GRASP_HEIGHT_OFFSET = 0.010
    CONTACT_TOLERANCE = 0.028
    CLOSED_TRANSLATION_LIMIT = 0.5

    def _motion(
        self,
        snapshot: PickPlaceSnapshot,
        target: np.ndarray,
        *,
        closed: bool,
    ) -> PickPlaceDecision:
        error = self.world_to_origin(target) - self.world_to_origin(
            self._vector(snapshot.eef_position)
        )
        limit = self.CLOSED_TRANSLATION_LIMIT if closed else 1.0
        translation = np.clip(error / 0.05, -limit, limit)
        action = np.r_[translation, np.zeros(3), 1.0 if closed else -1.0]
        return self._decision(action, force_expert=False)


__all__ = ["FastTransitPickPlaceOracle"]
