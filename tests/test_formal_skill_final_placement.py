import unittest

import numpy as np

from tools.formal_skill_final_placement import activate_final_placement
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


class FormalSkillFinalPlacementTests(unittest.TestCase):
    def make_oracle(self):
        return SafeCabinetPickPlaceOracle(
            destination_front=np.array([0.0, 0.0, 0.5]),
            destination_center=np.array([0.3, 0.0, 0.5]),
            destination_retreat=np.array([-0.2, 0.0, 0.5]),
        )

    def test_activation_skips_pickup_and_targets_verified_center(self):
        oracle = activate_final_placement(self.make_oracle())
        decision = oracle.decide(
            PickPlaceSnapshot(
                eef_position=np.array([0.0, 0.0, 0.5]),
                object_position=np.array([0.02, 0.0, 0.48]),
                grasped=True,
            )
        )

        self.assertEqual(decision.phase, "move_inside_destination")
        self.assertEqual(decision.canonical_action, "move")
        self.assertGreater(float(decision.action[0]), 0.0)
        self.assertEqual(float(decision.action[6]), 1.0)

    def test_center_reached_transitions_to_stationary_release(self):
        oracle = activate_final_placement(self.make_oracle())
        decision = oracle.decide(
            PickPlaceSnapshot(
                eef_position=np.array([0.3, 0.0, 0.5]),
                object_position=np.array([0.32, 0.0, 0.48]),
                grasped=True,
            )
        )

        self.assertEqual(decision.phase, "release_object")
        np.testing.assert_allclose(decision.action, [0, 0, 0, 0, 0, 0, -1])


if __name__ == "__main__":
    unittest.main()
