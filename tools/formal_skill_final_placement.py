"""Activate the verified center-release-retreat portion of a pick-place oracle."""

from __future__ import annotations

from tools.pick_place_oracle import PickPlaceOracle


def activate_final_placement(oracle: PickPlaceOracle) -> PickPlaceOracle:
    oracle.lift_target = None
    oracle._enter("move_inside_destination")
    return oracle


__all__ = ["activate_final_placement"]
