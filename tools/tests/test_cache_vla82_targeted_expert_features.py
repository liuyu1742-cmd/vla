from tools.cache_vla82_targeted_expert_features import selected_indices


def test_selected_indices_limits_source_before_stride():
    assert selected_indices(length=10, stride=2, max_samples=5) == [0, 2, 4]


def test_selected_indices_uses_full_source_without_limit():
    assert selected_indices(length=6, stride=2, max_samples=None) == [0, 2, 4]
