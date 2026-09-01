from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.vla82_acceptance.skill_learning import (
    PHASE_ORDER,
    classify_recognition,
    extract_motion_evidence,
    learn_skill_ir,
)


def _write_moving_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(30):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index, 16), (index + 12, 28), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_skill_ir_requires_visual_evidence_for_each_phase():
    task = {"task": "整理收纳", "operation_label": "物体归位"}

    with pytest.raises(ValueError, match="visual evidence"):
        learn_skill_ir(task, {"sampled_frames": [], "motion_detected": False})


def test_recognition_failure_is_not_promoted_to_pass():
    result = classify_recognition(expected="sponge", scores={"cup": 0.92})

    assert result == {
        "expected": "sponge",
        "predicted": "cup",
        "confidence": 0.92,
        "success": False,
    }


def test_motion_evidence_supports_ordered_skill_phases(tmp_path: Path):
    video = tmp_path / "moving.mp4"
    _write_moving_video(video)

    evidence = extract_motion_evidence(video, start_seconds=0.0, stop_seconds=2.9)
    skill_ir = learn_skill_ir(
        {"task": "整理收纳", "operation_label": "物体归位"}, evidence
    )

    assert evidence["motion_detected"] is True
    assert [phase["name"] for phase in skill_ir["phases"]] == list(PHASE_ORDER)
    assert all(phase["supporting_frames"] for phase in skill_ir["phases"])
    assert skill_ir["learning_source"] == "public_human_video_visual_motion"
