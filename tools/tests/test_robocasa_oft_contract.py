import numpy as np

from tools.robocasa_oft_contract import (
    episode_split,
    extract_arm_action,
    extract_proprio,
    future_action_chunk,
)


def test_extracts_oft_proprio_from_pandaomron_state():
    state = np.arange(16, dtype=np.float32)
    np.testing.assert_allclose(
        extract_proprio(state),
        np.array([7, 8, 9, 10, 11, 12, 13, 14.5], dtype=np.float32),
    )


def test_extracts_stationary_base_arm_action():
    action = np.arange(12, dtype=np.float32)
    np.testing.assert_array_equal(extract_arm_action(action), action[5:12])


def test_future_chunk_clamps_at_episode_end():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunk = future_action_chunk(actions, 1, chunk_size=4)
    np.testing.assert_array_equal(chunk, actions[[1, 2, 2, 2]])


def test_episode_split_has_no_leakage_and_is_deterministic():
    first = episode_split(list(range(20)), seed=82)
    second = episode_split(list(range(20)), seed=82)
    assert first == second
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["val"]) & set(first["test"]))
