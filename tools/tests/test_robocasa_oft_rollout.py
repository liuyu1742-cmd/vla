import numpy as np

from tools.robocasa_oft_rollout import proprio_from_observation, validate_action_chunk


def test_validate_action_chunk_accepts_finite_8_by_7():
    result = validate_action_chunk(np.zeros((8, 7), dtype=np.float32))
    assert result.shape == (8, 7)


def test_validate_action_chunk_rejects_non_finite_values():
    chunk = np.zeros((8, 7), dtype=np.float32)
    chunk[0, 0] = np.nan
    try:
        validate_action_chunk(chunk)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite chunk was accepted")


def test_proprio_from_gym_observation_matches_training_order():
    observation = {
        "state.base_position": np.arange(3, dtype=np.float32),
        "state.base_rotation": np.arange(3, 7, dtype=np.float32),
        "state.end_effector_position_relative": np.arange(7, 10, dtype=np.float32),
        "state.end_effector_rotation_relative": np.arange(10, 14, dtype=np.float32),
        "state.gripper_qpos": np.array([14, 15], dtype=np.float32),
    }
    np.testing.assert_array_equal(
        proprio_from_observation(observation),
        np.array([7, 8, 9, 10, 11, 12, 13, 14.5], dtype=np.float32),
    )
