from tools.train_robocasa_oft_head import EarlyStop, improvement


def test_improvement_requires_lower_validation_loss():
    assert improvement(0.19, 0.20, minimum=0.005)
    assert not improvement(0.198, 0.20, minimum=0.005)


def test_early_stop_counts_non_improving_epochs():
    stop = EarlyStop(patience=2, minimum=0.005)
    assert not stop.update(0.20)
    assert not stop.update(0.198)
    assert stop.update(0.197)
