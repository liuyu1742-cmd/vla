"""Run the formal collector with the fully verified toy-to-cabinet parameters."""

from __future__ import annotations

from typing import Any

import tools.collect_formal_skill_expert as collector
from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


SAFE_CABINET_DEPTH = 0.20
_ORIGINAL_CABINET_WAYPOINTS = collector._BASE.cabinet_waypoints


def _safe_cabinet_waypoints(raw: Any, depth_fraction: float = SAFE_CABINET_DEPTH):
    del depth_fraction
    return _ORIGINAL_CABINET_WAYPOINTS(raw, depth_fraction=SAFE_CABINET_DEPTH)


def configure_expert() -> type[SafeCabinetPickPlaceOracle]:
    collector._BASE.PickPlaceOracle = SafeCabinetPickPlaceOracle
    collector._BASE.cabinet_waypoints = _safe_cabinet_waypoints
    return SafeCabinetPickPlaceOracle


def main() -> int:
    configure_expert()
    return collector.main()


if __name__ == "__main__":
    raise SystemExit(main())
