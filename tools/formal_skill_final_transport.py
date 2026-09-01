"""Activate the verified front-align, center, release, and retreat sequence."""

from __future__ import annotations

from tools.pick_place_oracle import PickPlaceOracle


def activate_final_transport(oracle: PickPlaceOracle) -> PickPlaceOracle:
    oracle.lift_target = None
    oracle._enter("approach_destination")
    return oracle


__all__ = ["activate_final_transport"]
