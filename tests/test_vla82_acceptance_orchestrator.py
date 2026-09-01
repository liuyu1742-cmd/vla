from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.vla82_acceptance.orchestrator import (
    ids_to_run,
    load_run_state,
    record_attempt,
    retry_decision,
)


def test_resume_skips_only_verified_pass():
    state = {
        "items": {
            "VLA82-001": {"status": "PASS", "evidence_valid": True},
            "VLA82-002": {"status": "PASS", "evidence_valid": False},
        }
    }

    assert ids_to_run(["VLA82-001", "VLA82-002", "VLA82-003"], state) == [
        "VLA82-002",
        "VLA82-003",
    ]


def test_third_same_root_cause_requests_full_diagnosis_and_continues():
    attempts = [
        {"root_cause": "asset_load"},
        {"root_cause": "asset_load"},
        {"root_cause": "asset_load"},
    ]

    decision = retry_decision(attempts)

    assert decision["action"] == "FULL_DIAGNOSIS_CONTINUE"
    assert decision["stop_entire_test"] is False


def test_different_root_causes_do_not_trigger_same_cause_threshold():
    attempts = [
        {"root_cause": "asset_load"},
        {"root_cause": "video_decode"},
        {"root_cause": "asset_load"},
    ]

    assert retry_decision(attempts)["action"] == "RETRY"


def test_record_attempt_is_atomic_and_preserves_history(tmp_path: Path):
    state_path = tmp_path / "run_state.json"
    record_attempt(
        state_path,
        selection_id="VLA82-001",
        status="FAIL",
        evidence_valid=False,
        root_cause="asset_load",
        details={"seed": 82001},
    )
    record_attempt(
        state_path,
        selection_id="VLA82-001",
        status="PASS",
        evidence_valid=True,
        root_cause=None,
        details={"seed": 82002},
    )

    state = load_run_state(state_path)

    assert len(state["items"]["VLA82-001"]["attempts"]) == 2
    assert state["items"]["VLA82-001"]["status"] == "PASS"
    assert json.loads(state_path.read_text("utf-8"))["schema_version"] == (
        "vla82_acceptance_run_state_v1"
    )


def test_acceptance_cli_exposes_resumable_subcommands():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "tools/run_vla82_acceptance_completion.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for command in ("audit-demos", "audit-assets", "run-recognition", "run-objects", "run-tasks", "verify"):
        assert command in completed.stdout
