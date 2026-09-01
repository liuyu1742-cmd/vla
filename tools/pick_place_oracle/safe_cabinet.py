"""Expert parameters verified for a solid toy and shallow cabinet placement."""

from __future__ import annotations

from tools.pick_place_oracle.fast_transit import FastTransitPickPlaceOracle


class SafeCabinetPickPlaceOracle(FastTransitPickPlaceOracle):
    """Use the measured 0.3 transport limit that retained grasp at cabinet center."""

    CLOSED_TRANSLATION_LIMIT = 0.30


__all__ = ["SafeCabinetPickPlaceOracle"]
