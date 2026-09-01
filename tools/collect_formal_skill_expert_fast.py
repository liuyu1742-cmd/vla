"""Run the formal expert with measured grasp height and bounded fast transit."""

from __future__ import annotations

import tools.collect_formal_skill_expert as collector
from tools.pick_place_oracle.fast_transit import FastTransitPickPlaceOracle


def configure_expert() -> type[FastTransitPickPlaceOracle]:
    collector._BASE.PickPlaceOracle = FastTransitPickPlaceOracle
    return FastTransitPickPlaceOracle


def main() -> int:
    configure_expert()
    return collector.main()


if __name__ == "__main__":
    raise SystemExit(main())
