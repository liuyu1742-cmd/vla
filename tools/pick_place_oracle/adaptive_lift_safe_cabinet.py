"""Cabinet expert with a counter-clearance lift reachable across layouts."""

from __future__ import annotations

from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


class AdaptiveLiftSafeCabinetPickPlaceOracle(SafeCabinetPickPlaceOracle):
    """Lift enough to clear the counter without demanding unreachable height."""

    LIFT_HEIGHT = 0.14


__all__ = ["AdaptiveLiftSafeCabinetPickPlaceOracle"]
