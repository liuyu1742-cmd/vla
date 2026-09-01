import unittest

from tools.groundtruth_pick_place_controller import cabinet_drop_target, proportional_action


class GroundTruthPickPlaceControllerTests(unittest.TestCase):
    def test_proportional_action_is_bounded(self):
        action = proportional_action([0.0, 0.0, 0.0], [3.0, -3.0, 0.5], gripper=1.0)
        self.assertEqual(action, [1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 1.0])

    def test_cabinet_drop_target_uses_interior_floor_center(self):
        target = cabinet_drop_target([[0.0, 0.0, 1.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        self.assertEqual(target, [1.0, 1.0, 1.05])


if __name__ == "__main__":
    unittest.main()
