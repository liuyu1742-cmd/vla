import numpy as np

from tools.robocasa_oft_tcp_server import (
    load_retrieval_arrays,
    retrieval_key,
    select_retrieval_chunk,
)


def test_retrieval_key_uses_action_token_mean_and_std():
    hidden = np.arange(2 * 56 * 4, dtype=np.float32).reshape(2, 56, 4)
    key = retrieval_key(hidden)
    assert key.shape == (2, 8)
    np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1.0, atol=1e-6)


def test_select_retrieval_chunk_returns_closest_cosine_action():
    keys = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    actions = np.asarray([np.zeros((8, 7)), np.ones((8, 7))], dtype=np.float32)
    chunk, index, similarity = select_retrieval_chunk(
        np.asarray([0.1, 0.9], dtype=np.float32), keys, actions
    )
    assert index == 1
    assert similarity > 0.9
    np.testing.assert_array_equal(chunk, np.ones((8, 7), dtype=np.float32))


def test_load_retrieval_arrays_accepts_base_split_cache(tmp_path):
    np.save(tmp_path / "train_features.npy", np.zeros((2, 8, 28), dtype=np.float16))
    np.save(tmp_path / "train_actions.npy", np.zeros((2, 8, 7), dtype=np.float16))
    np.save(tmp_path / "val_features.npy", np.ones((1, 8, 28), dtype=np.float16))
    np.save(tmp_path / "val_actions.npy", np.ones((1, 8, 7), dtype=np.float16))

    features, actions = load_retrieval_arrays(tmp_path)

    assert features.shape == (3, 8, 28)
    assert actions.shape == (3, 8, 7)
