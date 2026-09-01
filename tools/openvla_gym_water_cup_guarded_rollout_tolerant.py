"""Guarded rollout variant calibrated to the observed glass-cup contact shell."""

from __future__ import annotations

from tools import openvla_gym_water_cup_guarded_rollout as guarded_rollout
from tools.water_cup_dagger_oracle import WaterCupDaggerOracle


CONTACT_TOLERANCE_METERS = 0.016


def configure_contact_tolerance() -> None:
    """Accept stable simulator contact instead of waiting below the collision shell."""

    WaterCupDaggerOracle.CONTACT_TOLERANCE = CONTACT_TOLERANCE_METERS


def main() -> int:
    configure_contact_tolerance()
    return guarded_rollout.main()


if __name__ == "__main__":
    raise SystemExit(main())
