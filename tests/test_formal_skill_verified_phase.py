import unittest

from tools.formal_skill_verified_phase import VerifiedPhaseController


class FormalSkillVerifiedPhaseTests(unittest.TestCase):
    def test_entry_uses_successful_demonstration_contact_distance(self):
        controller = VerifiedPhaseController(max_grasp_decisions=3)
        self.assertEqual(
            controller(
                "locate",
                object_eef_distance=0.025,
                grasped=False,
                inside=False,
                success=False,
            ),
            "locate",
        )
        self.assertEqual(
            controller(
                "locate",
                object_eef_distance=0.024,
                grasped=False,
                inside=False,
                success=False,
            ),
            "grasp",
        )

    def test_grasp_hysteresis_has_bounded_timeout(self):
        controller = VerifiedPhaseController(max_grasp_decisions=3)
        results = [
            controller(
                "grasp",
                object_eef_distance=0.03,
                grasped=False,
                inside=False,
                success=False,
            )
            for _ in range(3)
        ]

        self.assertEqual(results, ["grasp", "grasp", "locate"])

    def test_lost_transport_returns_to_visual_reacquisition(self):
        controller = VerifiedPhaseController(max_grasp_decisions=3)
        self.assertEqual(
            controller(
                "move",
                object_eef_distance=0.03,
                grasped=False,
                inside=False,
                success=False,
            ),
            "locate",
        )

    def test_native_predicates_are_authoritative(self):
        controller = VerifiedPhaseController(max_grasp_decisions=3)
        self.assertEqual(
            controller(
                "grasp",
                object_eef_distance=0.04,
                grasped=True,
                inside=False,
                success=False,
            ),
            "move",
        )
        self.assertEqual(
            controller(
                "move",
                object_eef_distance=0.04,
                grasped=True,
                inside=True,
                success=False,
            ),
            "place",
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
