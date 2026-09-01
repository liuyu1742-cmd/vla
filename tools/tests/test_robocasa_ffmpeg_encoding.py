from pathlib import Path


def test_cut_video_decodes_ffmpeg_stderr_as_utf8_with_replacement(monkeypatch):
    from tools.robocasa_episode_archive_v2 import _cut_video

    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("tools.robocasa_episode_archive_v2.subprocess.run", fake_run)
    _cut_video(Path("source.mp4"), Path("destination.mp4"), (0.0, 1.0))

    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"
