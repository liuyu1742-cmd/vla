import numpy as np

from tools.train_robocasa_oft_cached_head import load_extra_feature_arrays


def test_load_extra_feature_arrays_concatenates_caches(tmp_path):
    roots = []
    for index, count in enumerate((2, 3)):
        root = tmp_path / str(index)
        root.mkdir()
        np.save(root / "features.npy", np.full((count, 8, 4), index, dtype=np.float16))
        np.save(root / "actions.npy", np.full((count, 8, 7), index, dtype=np.float16))
        roots.append(str(root))

    features, actions = load_extra_feature_arrays(roots, expected_feature_shape=(8, 4))

    assert features.shape == (5, 8, 4)
    assert actions.shape == (5, 8, 7)
    assert np.all(features[:2] == 0)
    assert np.all(features[2:] == 1)
