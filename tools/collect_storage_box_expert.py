"""Collect a formal ``organizing::storage_box`` expert episode.

This specialization reuses the audited pick-place collector and calibrated
cabinet controller while replacing only the Task-2 relation contract.  Pass an
L2 storage-box Skill IR and an explicit output root on the command line.
"""

from __future__ import annotations

import tools.collect_formal_skill_expert as collector
from tools.collect_formal_skill_expert_safe import _safe_cabinet_waypoints
from tools.pick_place_oracle.storage_box import StorageBoxPickPlaceOracle


EXPECTED_RELATION = "organizing::storage_box"
EXPECTED_ACTIONS = [
    "locate(storage_box)",
    "grasp(storage_box)",
    "move(storage)",
    "place(storage_box)",
]


def configure_collector() -> type[StorageBoxPickPlaceOracle]:
    collector._BASE.EXPECTED_RELATION = EXPECTED_RELATION
    collector._BASE.EXPECTED_ACTIONS = list(EXPECTED_ACTIONS)
    collector._BASE.PickPlaceOracle = StorageBoxPickPlaceOracle
    collector._BASE.cabinet_waypoints = _safe_cabinet_waypoints
    return StorageBoxPickPlaceOracle


def main() -> int:
    configure_collector()
    return collector.main()


if __name__ == "__main__":
    raise SystemExit(main())
