import unittest

from tools.formal_skill_locate_recovery import LocateStagnationRecovery


class FormalSkillLocateRecoveryTests(unittest.TestCase):
    def test_activates_after_a_locate_window_without_required_progress(self):
        recovery = LocateStagnationRecovery(window_decisions=4, minimum_progress=0.01)

        active = [
            recovery.observe("locate", distance)
            for distance in (0.210, 0.209, 0.208, 0.207, 0.206)
        ]

        self.assertEqual(active, [False, False, False, False, True])
        self.assertTrue(recovery.active)

    def test_does_not_activate_while_locate_distance_is_improving(self):
        recovery = LocateStagnationRecovery(window_decisions=4, minimum_progress=0.01)

        active = [
            recovery.observe("locate", distance)
            for distance in (0.210, 0.205, 0.199, 0.193, 0.187, 0.181)
        ]

        self.assertEqual(active, [False] * 6)
        self.assertFalse(recovery.active)

    def test_leaving_locate_resets_latched_recovery(self):
        recovery = LocateStagnationRecovery(window_decisions=2, minimum_progress=0.01)
        for distance in (0.210, 0.209, 0.208):
            recovery.observe("locate", distance)
        self.assertTrue(recovery.active)

        self.assertFalse(recovery.observe("grasp", 0.020))
        self.assertFalse(recovery.active)
        self.assertFalse(recovery.observe("locate", 0.060))


if __name__ == "__main__":
    unittest.main()
