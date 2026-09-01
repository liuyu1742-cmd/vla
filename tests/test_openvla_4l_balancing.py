import numpy as np

from tools.openvla_4l_runner import action_phase, balanced_indices


def test_action_phase_separates_open_motion_closed_motion_and_settle():
    assert action_phase(np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)) == "open_motion"
    assert action_phase(np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float32)) == "closed_motion"
    assert action_phase(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)) == "settle"


def test_balanced_indices_draws_each_phase_equally():
    actions = [
        np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
    ]
    result = balanced_indices(actions, samples=6, seed=7)
    phases = [action_phase(actions[index]) for index in result]
    assert phases.count("open_motion") == 2
    assert phases.count("closed_motion") == 2
    assert phases.count("settle") == 2

