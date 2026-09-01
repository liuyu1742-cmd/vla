import numpy as np


def test_evidence_cases_cover_three_distinct_actions_and_two_conditions():
    from tools.record_experiment_4_2_5_evidence import EVIDENCE_CASES

    assert len(EVIDENCE_CASES) == 6
    assert {case.action_group for case in EVIDENCE_CASES} == {
        "drawer_open",
        "plate_push",
        "object_place",
    }
    assert {case.condition for case in EVIDENCE_CASES} == {
        "nominal",
        "generalization",
    }
    assert len({case.case_id for case in EVIDENCE_CASES}) == 6


def test_select_evidence_frame_indices_uses_first_middle_and_last():
    from tools.record_experiment_4_2_5_evidence import (
        select_evidence_frame_indices,
    )

    assert select_evidence_frame_indices(9) == (0, 4, 8)
    assert select_evidence_frame_indices(2) == (0, 1, 1)


def test_annotate_frame_preserves_shape_and_adds_overlay():
    from tools.record_experiment_4_2_5_evidence import annotate_frame

    frame = np.full((224, 224, 3), 255, dtype=np.uint8)
    annotated = annotate_frame(
        frame,
        title="Drawer opening | nominal",
        detail="step 12 | SUCCESS",
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated[:44], frame[:44])
