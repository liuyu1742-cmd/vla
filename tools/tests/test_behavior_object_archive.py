import unittest


class BehaviorObjectArchiveTests(unittest.TestCase):
    def test_validate_selection_rejects_duplicate_episode(self):
        from tools.behavior_object_archive import validate_selection

        rows = [
            {"task_dir": "task", "object_id": "book", "episode_index": 1, "instruction": "Move book", "operation": "place"},
            {"task_dir": "task", "object_id": "bottle", "episode_index": 1, "instruction": "Move bottle", "operation": "place"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate source episode"):
            validate_selection(rows)


if __name__ == "__main__":
    unittest.main()
