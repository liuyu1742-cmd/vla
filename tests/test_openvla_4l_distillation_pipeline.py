import pytest

from tools.openvla_4l_distillation_pipeline import next_action, round_beta


def test_round_beta_uses_approved_curriculum():
    assert [round_beta(index) for index in range(3)] == [0.7, 0.3, 0.0]
    with pytest.raises(ValueError, match="round_index"):
        round_beta(3)


def test_pipeline_accepts_any_nonzero_pure_success():
    assert next_action(pure_successes=1, round_index=0) == "accept_pure"


def test_pipeline_collects_before_final_round():
    assert next_action(pure_successes=0, round_index=0) == "collect_next_round"
    assert next_action(pure_successes=0, round_index=1) == "collect_next_round"


def test_pipeline_runs_hybrid_after_final_round():
    assert next_action(pure_successes=0, round_index=2) == "run_hybrid"
    with pytest.raises(ValueError, match="pure_successes"):
        next_action(pure_successes=-1, round_index=0)
