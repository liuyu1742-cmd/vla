import unittest

import tools.collect_formal_skill_expert as collector
from tools.collect_formal_skill_expert_adaptive import configure_expert
from tools.pick_place_oracle.adaptive_lift_safe_cabinet import (
    AdaptiveLiftSafeCabinetPickPlaceOracle,
)


class AdaptiveFormalCollectorTests(unittest.TestCase):
    def test_configure_expert_installs_adaptive_oracle(self) -> None:
        installed = configure_expert()

        self.assertIs(installed, AdaptiveLiftSafeCabinetPickPlaceOracle)
        self.assertIs(
            collector._BASE.PickPlaceOracle,
            AdaptiveLiftSafeCabinetPickPlaceOracle,
        )


if __name__ == "__main__":
    unittest.main()
