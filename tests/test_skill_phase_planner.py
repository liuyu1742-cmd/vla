import unittest


class SkillPhasePlannerTests(unittest.TestCase):
    def test_pick_remains_until_grasp_is_confirmed(self):
        from tools.skill_phase_planner import next_phase
        self.assertEqual(next_phase("pick", {"grasped": False}), "pick")

    def test_pick_advances_when_grasp_is_confirmed(self):
        from tools.skill_phase_planner import next_phase
        self.assertEqual(next_phase("pick", {"grasped": True}), "place")
