import numpy as np

from tools.libero_diffusion_contracts import (
    ActionNormalizer,
    build_window_indices,
    dataset_gripper_to_env,
    transform_dataset_gripper,
)


def test_libero_gripper_matches_official_transform_and_environment_inverse():
    raw = np.array([-1.0, 1.0], dtype=np.float32)
    transformed = transform_dataset_gripper(raw)
    np.testing.assert_array_equal(
        transformed,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dataset_gripper_to_env(transformed),
        np.array([-1.0, 1.0], dtype=np.float32),
    )


def test_action_normalizer_round_trip_uses_robust_quantiles():
    actions = np.arange(140, dtype=np.float32).reshape(20, 7)
    normalizer = ActionNormalizer.fit(actions)
    restored = normalizer.denormalize(normalizer.normalize(actions))
    np.testing.assert_allclose(restored, actions, atol=1e-4)


def test_windows_never_cross_episode_boundaries():
    rows = build_window_indices([3, 4], obs_horizon=2, action_horizon=3)
    assert all(row.episode_index in (0, 1) for row in rows)
    assert all(
        row.action_stop <= (3 if row.episode_index == 0 else 4)
        for row in rows
    )
