from pathlib import Path
import tempfile
import unittest

from tools.export_vla82_episode_video import output_path_for_episode, require_strict_pass


class EpisodeVideoExportTest(unittest.TestCase):
    def test_output_path_uses_primary_camera_and_keeps_episode_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode-2000.npz"
            self.assertEqual(output_path_for_episode(episode, "primary"), Path(directory) / "episode-2000.primary.mp4")

    def test_only_strict_predicate_success_is_exportable(self) -> None:
        require_strict_pass({"status": "PASS", "predicate_success": True})
        with self.assertRaises(ValueError):
            require_strict_pass({"status": "PASS", "predicate_success": False})
