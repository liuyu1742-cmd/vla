import pytest


def test_parse_task_ids_preserves_requested_order():
    from tools.openvla_libero_goal_subset import parse_task_ids

    assert parse_task_ids("1,2,5,7") == [1, 2, 5, 7]


def test_parse_task_ids_rejects_duplicate_ids():
    from tools.openvla_libero_goal_subset import parse_task_ids

    with pytest.raises(ValueError, match="duplicate"):
        parse_task_ids("1,2,2")
