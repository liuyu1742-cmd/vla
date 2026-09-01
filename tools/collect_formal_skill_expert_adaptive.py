"""Collect formal demonstrations with reachable lift and safe cabinet transit."""

from __future__ import annotations

import tools.collect_formal_skill_expert as collector
from tools.collect_formal_skill_expert_safe import _safe_cabinet_waypoints
from tools.pick_place_oracle.adaptive_lift_safe_cabinet import (
    AdaptiveLiftSafeCabinetPickPlaceOracle,
)


def configure_expert() -> type[AdaptiveLiftSafeCabinetPickPlaceOracle]:
    collector._BASE.PickPlaceOracle = AdaptiveLiftSafeCabinetPickPlaceOracle
    collector._BASE.cabinet_waypoints = _safe_cabinet_waypoints
    return AdaptiveLiftSafeCabinetPickPlaceOracle


def main() -> int:
    configure_expert()
    return collector.main()


if __name__ == "__main__":
    raise SystemExit(main())
