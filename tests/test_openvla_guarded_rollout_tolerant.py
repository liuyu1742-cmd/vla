from tools.openvla_gym_water_cup_guarded_rollout_tolerant import (
    configure_contact_tolerance,
)
from tools.water_cup_dagger_oracle import WaterCupDaggerOracle


def test_tolerant_rollout_accepts_observed_glass_contact_distance():
    original = WaterCupDaggerOracle.CONTACT_TOLERANCE
    try:
        configure_contact_tolerance()
        assert WaterCupDaggerOracle.CONTACT_TOLERANCE == 0.016
    finally:
        WaterCupDaggerOracle.CONTACT_TOLERANCE = original
