"""Learn an auditable temporal Skill IR from public demonstration video motion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PHASE_ORDER = ("locate", "approach", "contact_or_grasp", "operate", "complete")


def classify_recognition(
    *, expected: str, scores: Mapping[str, float]
) -> dict[str, Any]:
    """Preserve the model prediction instead of overwriting it with the expected label."""
    if scores:
        predicted, confidence = max(scores.items(), key=lambda item: float(item[1]))
        confidence = float(confidence)
    else:
        predicted, confidence = None, 0.0
    return {
        "expected": expected,
        "predicted": predicted,
        "confidence": confidence,
        "success": predicted == expected,
    }


def extract_motion_evidence(
    video: Path,
    *,
    start_seconds: float,
    stop_seconds: float,
    sample_count: int = 20,
) -> dict[str, Any]:
    """Sample real frames and measure temporal change supporting learned phases."""
    if stop_seconds <= start_seconds:
        raise ValueError("stop_seconds must be greater than start_seconds")
    if sample_count < len(PHASE_ORDER):
        raise ValueError(f"sample_count must be at least {len(PHASE_ORDER)}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"video cannot be decoded: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    times = np.linspace(start_seconds, stop_seconds, sample_count)
    sampled: list[dict[str, Any]] = []
    prior_gray: np.ndarray | None = None
    try:
        for sample_index, seconds in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = (
                0.0
                if prior_gray is None
                else float(np.mean(cv2.absdiff(gray, prior_gray)))
            )
            sampled.append(
                {
                    "sample_index": sample_index,
                    "frame_index": int(round(float(seconds) * fps)) if fps > 0 else sample_index,
                    "seconds": float(seconds),
                    "motion_score": motion_score,
                }
            )
            prior_gray = gray
    finally:
        capture.release()
    positive_motion = [item["motion_score"] for item in sampled if item["motion_score"] > 0.5]
    return {
        "video": str(video.resolve()),
        "start_seconds": float(start_seconds),
        "stop_seconds": float(stop_seconds),
        "sampled_frames": sampled,
        "motion_detected": bool(positive_motion),
        "mean_positive_motion": float(np.mean(positive_motion)) if positive_motion else 0.0,
    }


def learn_skill_ir(
    task: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Create ordered phases whose supporting frame indices come from the video."""
    sampled = evidence.get("sampled_frames")
    if (
        not isinstance(sampled, list)
        or len(sampled) < len(PHASE_ORDER)
        or evidence.get("motion_detected") is not True
    ):
        raise ValueError("each learned phase requires visual evidence")
    partitions = np.array_split(np.arange(len(sampled)), len(PHASE_ORDER))
    phases: list[dict[str, Any]] = []
    for name, partition in zip(PHASE_ORDER, partitions, strict=True):
        supporting = [sampled[int(index)] for index in partition]
        if not supporting:
            raise ValueError("each learned phase requires visual evidence")
        phases.append(
            {
                "name": name,
                "start_seconds": float(supporting[0]["seconds"]),
                "stop_seconds": float(supporting[-1]["seconds"]),
                "supporting_frames": [int(item["frame_index"]) for item in supporting],
                "mean_motion_score": float(
                    np.mean([float(item["motion_score"]) for item in supporting])
                ),
            }
        )
    return {
        "schema": "vla82_hybrid_skill_ir_v1",
        "task": str(task["task"]),
        "operation_label": str(task["operation_label"]),
        "phases": phases,
        "learning_source": "public_human_video_visual_motion",
        "source_video": str(evidence.get("video", "")),
        "motion_summary": {
            "sample_count": len(sampled),
            "mean_positive_motion": float(evidence.get("mean_positive_motion", 0.0)),
        },
    }
