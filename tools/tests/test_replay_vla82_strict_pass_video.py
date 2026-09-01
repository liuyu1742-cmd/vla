from tools.replay_vla82_strict_pass_video import replay_coverage_credit, replay_coverage_tracker


class _Spec:
    def __init__(self, phases):
        self.phases = phases


class _Contract:
    target_geometries = {"counter:tabletop": object()}


def test_replay_enables_contact_coverage_for_wipe_phases():
    tracker = replay_coverage_tracker(_Spec(("grasp", "wipe", "return")), _Contract())
    assert tracker is not None
    assert tracker.target_id == "counter:tabletop"


def test_replay_skips_coverage_for_non_cleaning_phases():
    assert replay_coverage_tracker(_Spec(("grasp", "place")), _Contract()) is None


def test_replay_credits_only_recorded_cleaning_phase():
    assert replay_coverage_credit(4) is True
    assert replay_coverage_credit(0) is False
    assert replay_coverage_credit(2) is False
