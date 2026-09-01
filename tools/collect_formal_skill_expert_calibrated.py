"""Run the formal collector with the measured solid-toy grasp geometry."""

from __future__ import annotations

import tools.collect_formal_skill_expert as collector


def configure_toy_grasp() -> type:
    """Apply the offset verified by bilateral finger contact and a 13 cm lift."""
    oracle_class = collector._BASE.PickPlaceOracle
    oracle_class.GRASP_HEIGHT_OFFSET = 0.010
    oracle_class.CONTACT_TOLERANCE = 0.028
    return oracle_class


def main() -> int:
    configure_toy_grasp()
    return collector.main()


if __name__ == "__main__":
    raise SystemExit(main())
