import pytest

from tools.archive_selected_robocasa_objects import validate_selection


def test_validate_selection_rejects_reused_episode():
    rows = [
        {"task_dir": "task", "object_id": "cup", "episode_index": 1, "instruction": "pick cup", "operation": "pick"},
        {"task_dir": "task", "object_id": "bowl", "episode_index": 1, "instruction": "pick bowl", "operation": "pick"},
    ]
    with pytest.raises(ValueError, match="allocated twice"):
        validate_selection(rows)


def test_validate_selection_allows_unique_rows():
    validate_selection([
        {"task_dir": "task", "object_id": "cup", "episode_index": 1, "instruction": "pick cup", "operation": "pick"},
        {"task_dir": "task", "object_id": "bowl", "episode_index": 2, "instruction": "pick bowl", "operation": "pick"},
    ])
