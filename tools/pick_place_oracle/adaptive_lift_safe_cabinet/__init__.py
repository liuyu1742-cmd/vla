"""Cabinet expert with a counter-clearance lift reachable across layouts."""

from __future__ import annotations

import numpy as np

from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


class AdaptiveLiftSafeCabinetPickPlaceOracle(SafeCabinetPickPlaceOracle):
    """Lift enough to clear the counter without demanding unreachable height."""

    LIFT_HEIGHT = 0.14
    CLOSED_TRANSLATION_LIMIT = np.float32(0.30)


__all__ = ["AdaptiveLiftSafeCabinetPickPlaceOracle"]
