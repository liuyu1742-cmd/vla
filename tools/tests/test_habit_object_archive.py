import unittest


class TestHabitObjectArchive(unittest.TestCase):
    def test_episode_paths_use_six_digit_episode_index(self):
        from tools.habit_object_archive import episode_paths

        parquet, rgb = episode_paths("C:/habit", 19)
        self.assertTrue(str(parquet).endswith("data\\chunk-000\\episode_000019.parquet"))
        self.assertTrue(str(rgb).endswith("videos\\chunk-000\\observation.images.exo_view\\episode_000019.mp4"))


if __name__ == "__main__":
    unittest.main()
