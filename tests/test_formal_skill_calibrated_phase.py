import unittest

from tools.formal_skill_calibrated_phase import next_calibrated_phase


class FormalSkillCalibratedPhaseTests(unittest.TestCase):
    def test_three_centimeters_is_not_verified_contact(self):
        self.assertEqual(
            next_calibrated_phase(
                "locate",
                object_eef_distance=0.030,
                grasped=False,
                inside=False,
                success=False,
            ),
            "locate",
        )

    def test_measured_2_8_centimeter_contact_enters_grasp(self):
        self.assertEqual(
            next_calibrated_phase(
                "locate",
                object_eef_distance=0.028,
                grasped=False,
                inside=False,
                success=False,
            ),
            "grasp",
        )

    def test_native_predicates_still_override_distance(self):
        self.assertEqual(
            next_calibrated_phase(
                "grasp",
                object_eef_distance=0.04,
                grasped=True,
                inside=False,
                success=False,
            ),
            "move",
        )
        self.assertEqual(
            next_calibrated_phase(
                "move",
                object_eef_distance=0.04,
                grasped=True,
                inside=True,
                success=False,
            ),
            "place",
        )
        self.assertEqual(
            next_calibrated_phase(
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
