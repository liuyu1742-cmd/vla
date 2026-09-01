import numpy as np
import pytest

from tools.verify_robocasa365_official_replay import validate_episode_contract


def test_episode_contract_accepts_aligned_successful_public_demo():
    states = np.zeros((3, 8), dtype=np.float32)
    actions = np.zeros((3, 12), dtype=np.float32)
    validate_episode_contract(states, actions, np.asarray([0.0, 1.0, 1.0]))


def test_episode_contract_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        validate_episode_contract(
            np.zeros((3, 8)), np.zeros((2, 12)), np.ones(2)
        )


def test_episode_contract_requires_recorded_success():
    with pytest.raises(ValueError, match="no successful frame"):
        validate_episode_contract(
            np.zeros((3, 8)), np.zeros((3, 12)), np.zeros(3)
        )
