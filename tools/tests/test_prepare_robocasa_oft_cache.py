import numpy as np

from tools.prepare_robocasa_oft_cache import build_aligned_arrays, validate_lengths


def test_build_aligned_arrays_extracts_expected_shapes():
    states = np.arange(48, dtype=np.float32).reshape(3, 16)
    actions = np.arange(36, dtype=np.float32).reshape(3, 12)
    result = build_aligned_arrays(states, actions)
    assert result["proprio"].shape == (3, 8)
    assert result["actions"].shape == (3, 7)
    assert result["action_chunks"].shape == (3, 8, 7)


def test_validate_lengths_rejects_camera_mismatch():
    try:
        validate_lengths(10, 10, 9, 10)
    except ValueError as error:
        assert "alignment" in str(error)
    else:
        raise AssertionError("camera mismatch was accepted")
