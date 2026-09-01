from pathlib import Path

from tools.verified_training_registry import has_trainable_triplet


def test_has_trainable_triplet_requires_instruction_rgb_and_action(tmp_path: Path):
    rgb = tmp_path / "episode.mp4"
    action = tmp_path / "episode.parquet"
    rgb.write_bytes(b"rgb")
    action.write_bytes(b"action")

    assert has_trainable_triplet("put the book into the box", rgb, action) is True
    assert has_trainable_triplet("", rgb, action) is False
    assert has_trainable_triplet("put the book into the box", rgb, tmp_path / "missing.parquet") is False
