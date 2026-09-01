import json
from pathlib import Path

from tools.experiment_4_6_evidence import (
    EvidenceRecord,
    load_libero_report,
    mean_success,
)


def test_load_libero_report_counts_episode_booleans(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"episodes": [{"success": True}, {"success": False}]}),
        encoding="utf-8",
    )

    row = load_libero_report(path, "libero_goal")

    assert row.success_rate == 50.0
    assert row.trials == 2
    assert row.evidence_kind == "local_measured"


def test_load_libero_report_rejects_non_boolean_success(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"episodes": [{"success": 1}]}), encoding="utf-8")

    try:
        load_libero_report(path, "libero_goal")
    except ValueError as exc:
        assert "success booleans" in str(exc)
    else:
        raise AssertionError("non-boolean success values must be rejected")


def test_mean_success_is_trial_weighted():
    records = [
        EvidenceRecord("a", 100.0, 1, "local_measured", "a.json"),
        EvidenceRecord("b", 50.0, 3, "official_reported", "paper"),
    ]

    assert mean_success(records) == 62.5

