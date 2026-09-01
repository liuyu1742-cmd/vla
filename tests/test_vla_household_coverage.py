import pytest

from tools.vla_household_coverage import eligible_rows, validate_coverage_row


def test_catalog_only_row_is_excluded_from_trainable_coverage():
    row = {
        "dataset_id": "robocasa",
        "room_zone": "kitchen",
        "task_id": "washing_dishes::rinse_bowls",
        "object_id": "bowl",
        "trainability": "catalog_only",
        "evidence": "official_registry",
    }

    validate_coverage_row(row)

    assert eligible_rows([row]) == []


def test_eligible_row_requires_a_concrete_evidence_route():
    row = {
        "dataset_id": "robocasa",
        "room_zone": "kitchen",
        "task_id": "washing_dishes::rinse_bowls",
        "object_id": "bowl",
        "trainability": "generatable",
        "evidence": "local_task_code:expert_collector",
    }

    validate_coverage_row(row)


def test_eligible_row_without_evidence_is_rejected():
    row = {
        "dataset_id": "robocasa",
        "room_zone": "kitchen",
        "task_id": "washing_dishes::rinse_bowls",
        "object_id": "bowl",
        "trainability": "ready",
        "evidence": "",
    }

    with pytest.raises(ValueError, match="evidence"):
        validate_coverage_row(row)
