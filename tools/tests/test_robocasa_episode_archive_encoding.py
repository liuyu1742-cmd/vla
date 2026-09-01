import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.robocasa_episode_archive_v2 import _cut_video


class FfmpegEncodingTests(unittest.TestCase):
    def test_cut_video_decodes_ffmpeg_stderr_with_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"; source.write_bytes(b"x")
            dest = root / "dest.mp4"
            with patch("tools.robocasa_episode_archive_v2.subprocess.run") as run:
                _cut_video(source, dest, (0.0, 1.0))
            self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
            self.assertEqual(run.call_args.kwargs["errors"], "replace")
