"""Container-specific pick-place expert verified for the storage-box proxy."""

from __future__ import annotations

import numpy as np

from tools.pick_place_oracle import PickPlaceDecision, PickPlaceSnapshot
from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


class StorageBoxPickPlaceOracle(SafeCabinetPickPlaceOracle):
    """Top-grasp a flat container and transit as soon as contact is confirmed."""

    GRASP_HEIGHT_OFFSET = 0.0
    LIFT_HEIGHT = 0.10
    CLOSED_TRANSLATION_LIMIT = 0.30

    def decide(self, snapshot: PickPlaceSnapshot) -> PickPlaceDecision:
        if self.phase == "close_gripper" and snapshot.grasped:
            eef = self._vector(snapshot.eef_position)
            self.lift_target = eef + np.array([0.0, 0.0, self.LIFT_HEIGHT])
            self._enter("lift_object")
        return super().decide(snapshot)


__all__ = ["StorageBoxPickPlaceOracle"]
