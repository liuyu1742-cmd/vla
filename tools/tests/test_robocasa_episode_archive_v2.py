import pandas as pd

from tools.robocasa_episode_archive_v2 import chunk_relative_rows


def test_chunk_relative_rows_subtracts_the_selected_file_global_start():
    episodes = pd.DataFrame([
        {"episode_index": 1, "data/chunk_index": 0, "data/file_index": 1, "dataset_from_index": 100, "dataset_to_index": 120},
        {"episode_index": 2, "data/chunk_index": 0, "data/file_index": 1, "dataset_from_index": 120, "dataset_to_index": 150},
        {"episode_index": 3, "data/chunk_index": 0, "data/file_index": 2, "dataset_from_index": 150, "dataset_to_index": 170},
    ])
    assert chunk_relative_rows(episodes, episodes.iloc[1].to_dict()) == (20, 50)
