import unittest

import numpy as np

from tools.formal_skill_final_transport import activate_final_transport
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


class FormalSkillFinalTransportTests(unittest.TestCase):
    def make_oracle(self):
        return SafeCabinetPickPlaceOracle(
            destination_front=np.array([0.0, 0.0, 0.5]),
            destination_center=np.array([0.3, 0.0, 0.5]),
            destination_retreat=np.array([-0.2, 0.0, 0.5]),
        )

    def test_activation_aligns_at_front_before_moving_inside(self):
        oracle = activate_final_transport(self.make_oracle())
        decision = oracle.decide(
            PickPlaceSnapshot(
                eef_position=np.array([0.2, 0.2, 0.7]),
                object_position=np.array([0.22, 0.2, 0.68]),
                grasped=True,
            )
        )

        self.assertEqual(decision.phase, "approach_destination")
        self.assertEqual(decision.canonical_action, "move")
        self.assertEqual(float(decision.action[6]), 1.0)

    def test_front_alignment_transitions_to_inside_segment(self):
        oracle = activate_final_transport(self.make_oracle())
        decision = oracle.decide(
            PickPlaceSnapshot(
                eef_position=np.array([0.0, 0.0, 0.5]),
                object_position=np.array([0.02, 0.0, 0.48]),
                grasped=True,
            )
        )

        self.assertEqual(decision.phase, "move_inside_destination")
        self.assertGreater(float(decision.action[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
