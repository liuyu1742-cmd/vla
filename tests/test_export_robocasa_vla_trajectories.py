import unittest


class RoboCasaVlaTrajectoryTests(unittest.TestCase):
    def test_valid_episode_has_synchronized_observations_actions_and_metadata(self):
        from tools.export_robocasa_vla_trajectories import validate_episode

        episode = {
            "instruction": "把杯子放到碗里",
            "task_id": "object_fetching",
            "object": "cup",
            "observation_paths": ["frames/000.png", "frames/001.png"],
            "actions": [[0.0] * 7, [0.1] * 7],
            "action_dim": 7,
            "control_hz": 5,
            "success": True,
            "failure_reason": None,
            "seed": 7,
        }

        self.assertEqual(validate_episode(episode), [])

    def test_mismatched_observations_and_actions_is_rejected(self):
        from tools.export_robocasa_vla_trajectories import validate_episode

        errors = validate_episode({"instruction": "x", "observation_paths": ["a"], "actions": [[0.0] * 7, [0.0] * 7], "action_dim": 7})

        self.assertIn("observation_action_count_mismatch", errors)

    def test_local_robocasa_source_is_recognized_without_editable_install(self):
        from tools.export_robocasa_vla_trajectories import inspect_simulator

        simulator = inspect_simulator()

        self.assertTrue(simulator["robocasa_source_available"])
        self.assertTrue(simulator["mujoco_importable"])
        self.assertEqual(simulator["status"], "ready_for_export")


if __name__ == "__main__":
    unittest.main()
