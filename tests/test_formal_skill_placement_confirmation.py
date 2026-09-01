import unittest

from tools.formal_skill_placement_confirmation import PlacementConfirmedPhaseController


class FormalSkillPlacementConfirmationTests(unittest.TestCase):
    def test_move_requires_consecutive_inside_observations_before_place(self):
        controller = PlacementConfirmedPhaseController(
            inside_confirm_decisions=3,
            max_grasp_decisions=4,
        )
        results = [
            controller(
                "move",
                object_eef_distance=0.02,
                grasped=True,
                inside=True,
                success=False,
            )
            for _ in range(3)
        ]

        self.assertEqual(results, ["move", "move", "place"])

    def test_leaving_storage_resets_confirmation(self):
        controller = PlacementConfirmedPhaseController(
            inside_confirm_decisions=2,
            max_grasp_decisions=4,
        )
        controller(
            "move",
            object_eef_distance=0.02,
            grasped=True,
            inside=True,
            success=False,
        )
        controller(
            "move",
            object_eef_distance=0.02,
            grasped=True,
            inside=False,
            success=False,
        )
        result = controller(
            "move",
            object_eef_distance=0.02,
            grasped=True,
            inside=True,
            success=False,
        )

        self.assertEqual(result, "move")

    def test_place_remains_latched_while_object_is_inside(self):
        controller = PlacementConfirmedPhaseController(
            inside_confirm_decisions=3,
            max_grasp_decisions=4,
        )
        self.assertEqual(
            controller(
                "place",
                object_eef_distance=0.06,
                grasped=False,
                inside=True,
                success=False,
            ),
            "place",
        )

    def test_success_is_still_immediate(self):
        controller = PlacementConfirmedPhaseController(
            inside_confirm_decisions=3,
            max_grasp_decisions=4,
        )
        self.assertEqual(
            controller(
                "place",
                object_eef_distance=0.20,
                grasped=False,
                inside=True,
                success=True,
            ),
            "done",
        )


if __name__ == "__main__":
    unittest.main()
