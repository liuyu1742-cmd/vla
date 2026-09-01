import unittest


class TestTask11Content(unittest.TestCase):
    def test_task11_definition_keeps_only_verified_objects(self):
        from tools.update_chapter5_task11_installation import TASK11_OBJECT_ROWS

        self.assertEqual(len(TASK11_OBJECT_ROWS), 4)
        self.assertEqual(
            [row[1] for row in TASK11_OBJECT_ROWS],
            ["海报", "墙钉", "数码相机", "相机三脚架"],
        )


if __name__ == "__main__":
    unittest.main()
