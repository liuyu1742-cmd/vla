from pathlib import Path

from tools.robocasa_episode_archive import episode_assets, normalized_object_id


def test_episode_assets_uses_metadata_chunk_references(tmp_path: Path):
    episode = {
        "episode_index": 12,
        "dataset_from_index": 100,
        "dataset_to_index": 120,
        "data/chunk_index": 2,
        "data/file_index": 3,
        "videos/observation.images.robot0_agentview_left/chunk_index": 4,
        "videos/observation.images.robot0_agentview_left/file_index": 5,
        "videos/observation.images.robot0_agentview_left/from_timestamp": 1.5,
        "videos/observation.images.robot0_agentview_left/to_timestamp": 2.5,
    }
    assets = episode_assets(tmp_path, episode, "observation.images.robot0_agentview_left")
    assert assets["parquet"] == tmp_path / "data/chunk-002/file-003.parquet"
    assert assets["video"] == tmp_path / "videos/observation.images.robot0_agentview_left/chunk-004/file-005.mp4"
    assert assets["frame_slice"] == (100, 120)
    assert assets["time_slice"] == (1.5, 2.5)


def test_normalized_object_id_is_stable_and_path_safe():
    assert normalized_object_id("Coffee Cup / Mug") == "coffee_cup_mug"
    assert normalized_object_id("  鞋子  ") == "鞋子"
