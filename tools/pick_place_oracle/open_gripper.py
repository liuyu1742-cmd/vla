"""Panda-calibrated expert that commands fully open (-1) and closed (+1)."""

from __future__ import annotations

import numpy as np

from tools.pick_place_oracle import PickPlaceDecision, PickPlaceOracle, PickPlaceSnapshot


class OpenGripperPickPlaceOracle(PickPlaceOracle):
    """Use the full normalized gripper range during collision-sensitive approach."""

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
        limit = 0.20 if closed else 1.0
        translation = np.clip(error / 0.05, -limit, limit)
        gripper_command = 1.0 if closed else -1.0
        action = np.r_[translation, np.zeros(3), gripper_command]
        return self._decision(action, force_expert=False)

    def _hold(self, *, closed: bool, force_expert: bool = True) -> PickPlaceDecision:
        gripper_command = 1.0 if closed else -1.0
        return self._decision(
            np.r_[np.zeros(6), gripper_command], force_expert=force_expert
        )


__all__ = ["OpenGripperPickPlaceOracle"]
