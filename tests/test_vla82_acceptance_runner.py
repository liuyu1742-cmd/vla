from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tools.vla82_acceptance.assets import DigitalTwinSpec, write_digital_twin
from tools.vla82_acceptance.runner import (
    build_trial_report,
    run_hybrid_trial,
    validate_visual_change,
)


def test_report_discloses_supervisor_and_action_sources():
    report = build_trial_report(
        raw_actions=[[0.1] * 7],
        executed_actions=[[0.0] * 7],
        interventions=1,
        success=True,
    )

    assert report["pure_autonomous_vla"] is False
    assert report["supervisor_intervention_count"] == 1
    assert report["trajectory"][0]["raw_openvla_action"] == [0.1] * 7
    assert report["trajectory"][0]["executed_action"] == [0.0] * 7


def test_identical_first_and_last_frames_fail_evidence_gate(tmp_path: Path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "first_frame.png"), image)
    cv2.imwrite(str(tmp_path / "last_frame.png"), image)

    assert "unchanged_first_last_frame" in validate_visual_change(tmp_path)


def test_digital_twin_hybrid_trial_writes_real_video_and_success_state(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "source_demo.json").write_text(
        json.dumps({"source_kind": "public_human_demonstration"}), encoding="utf-8"
    )
    cv2.imwrite(
        str(trial_dir / "source_demo_preview.png"),
        np.full((32, 32, 3), 127, dtype=np.uint8),
    )
    twin = write_digital_twin(
        DigitalTwinSpec(
            selection_id="VLA82-001",
            target_class="垃圾桶",
            shape="cylinder",
            size=(0.12, 0.12, 0.18),
            affordances=("graspable", "placeable"),
        ),
        tmp_path / "twins",
    )
    mapping = {
        "selection_id": "VLA82-001",
        "task": "室内卫生清洁",
        "object": "垃圾桶",
        "operation_label": "垃圾投放",
        "action_family": "composite",
    }
    skill_ir = {
        "schema": "vla82_hybrid_skill_ir_v1",
        "phases": [{"name": name} for name in (
            "locate", "approach", "contact_or_grasp", "operate", "complete"
        )],
    }
    recognition = {
        "expected": "垃圾桶",
        "predicted": "垃圾桶",
        "confidence": 1.0,
        "success": True,
        "method": "unit_fixture",
    }

    report = run_hybrid_trial(
        mapping=mapping,
        skill_ir=skill_ir,
        digital_twin=twin,
        raw_openvla_action=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        recognition=recognition,
        output_dir=trial_dir,
        frame_count=24,
        render_size=96,
    )

    assert report["operation"]["success"] is True
    assert report["acceptance_status"] == "PASS"
    assert (tmp_path / "trial/rollout.mp4").stat().st_size > 0
    assert validate_visual_change(tmp_path / "trial") == []
    saved = json.loads((tmp_path / "trial/report.json").read_text("utf-8"))
    assert saved["selection_id"] == "VLA82-001"
    capture = cv2.VideoCapture(str(tmp_path / "trial/rollout.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    capture.release()


def test_articulation_trial_template_is_loadable_and_reaches_target(tmp_path: Path):
    trial_dir = tmp_path / "articulation"
    trial_dir.mkdir()
    (trial_dir / "source_demo.json").write_text("{}", encoding="utf-8")
    cv2.imwrite(
        str(trial_dir / "source_demo_preview.png"),
        np.full((32, 32, 3), 127, dtype=np.uint8),
    )
    twin = write_digital_twin(
        DigitalTwinSpec(
            selection_id="VLA82-007",
            target_class="设备",
            shape="box",
            size=(0.1, 0.1, 0.1),
            affordances=("visual_target",),
        ),
        tmp_path / "twins",
    )
    report = run_hybrid_trial(
        mapping={
            "selection_id": "VLA82-007",
            "task": "设备控制",
            "object": "设备",
            "operation_label": "打开设备",
            "action_family": "articulation",
        },
        skill_ir={"schema": "vla82_hybrid_skill_ir_v1", "phases": []},
        digital_twin=twin,
        raw_openvla_action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        recognition={"success": True},
        output_dir=trial_dir,
        frame_count=12,
        render_size=64,
    )

    assert report["operation"]["family"] == "device_control"
    assert report["acceptance_status"] == "PASS"
