import unittest

from tools.build_vla82_robocasa365_cache import sample_indices


class BuildVLA82RoboCasa365CacheTests(unittest.TestCase):
    def test_stride_sampling_never_exceeds_episode_length(self):
        self.assertEqual(sample_indices(23, 8), [0, 8, 16])


if __name__ == "__main__":
    unittest.main()
