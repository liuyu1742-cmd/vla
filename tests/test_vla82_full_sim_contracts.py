from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.vla82_full_sim.contracts import validate_no_shortcuts


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "run_vla82_full_simulation.py"


def _valid_report() -> dict[str, object]:
    return {
        "complete_robot_model": True,
        "execution_kind": "learned_closed_loop",
        "terminal_state_directly_written": False,
        "adapter_checkpoint_sha256": "a" * 64,
    }


def test_no_shortcuts_requires_complete_robot() -> None:
    report = _valid_report()
    report["complete_robot_model"] = False
    assert validate_no_shortcuts(report, Path("unused")) == ["complete_robot_missing"]


def test_no_shortcuts_rejects_each_forbidden_execution_kind() -> None:
    for kind in ("abstract_gantry", "direct_state_write", "scripted_full_trajectory"):
        report = _valid_report()
        report["execution_kind"] = kind
        assert validate_no_shortcuts(report, Path("unused")) == ["forbidden_execution_kind"]


def test_no_shortcuts_requires_terminal_state_disproof_and_adapter_fingerprint() -> None:
    report = _valid_report()
    report["terminal_state_directly_written"] = None
    report["adapter_checkpoint_sha256"] = None
    assert validate_no_shortcuts(report, Path("unused")) == [
        "terminal_state_write_not_disproved",
        "adapter_fingerprint_missing",
    ]


def test_no_shortcuts_requires_bundle_evidence_when_bundle_is_validated(tmp_path: Path) -> None:
    assert validate_no_shortcuts(_valid_report(), tmp_path) == ["bundle_evidence_missing"]
    (tmp_path / "bundle_evidence.json").write_text("{}", encoding="utf-8")
    assert validate_no_shortcuts(_valid_report(), tmp_path) == []


def test_cli_help_and_compile_specs() -> None:
    help_result = subprocess.run(
        [sys.executable, str(CLI), "--help"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert help_result.returncode == 0
    assert "compile-specs" in help_result.stdout

    compile_result = subprocess.run(
        [sys.executable, str(CLI), "compile-specs"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert compile_result.returncode == 0, compile_result.stderr
    assert compile_result.stdout.strip() == "compiled=60 errors=0"
