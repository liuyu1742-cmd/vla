import numpy as np

from tools.robocasa_oft_model import normalize, quantile_stats, unnormalize


def test_quantile_round_trip():
    values = np.linspace(-2, 2, 500, dtype=np.float32).reshape(100, 5)
    stats = quantile_stats(values)
    in_range = values[1:-1]
    restored = unnormalize(normalize(in_range, stats), stats)
    np.testing.assert_allclose(restored, in_range, atol=1e-5)


def test_normalization_is_finite_for_constant_channel():
    values = np.ones((10, 3), dtype=np.float32)
    assert np.isfinite(normalize(values, quantile_stats(values))).all()
