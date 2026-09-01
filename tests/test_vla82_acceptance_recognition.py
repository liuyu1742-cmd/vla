from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools.vla82_acceptance.recognition import validate_policy_response


def test_finite_target_conditioned_response_counts_as_no_inference_failure(
    tmp_path: Path,
):
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))

    report = validate_policy_response(
        expected_object="垃圾桶",
        image_path=image,
        instruction="将垃圾投放至垃圾桶",
        action=[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
        latency_seconds=0.7,
        endpoint="127.0.0.1:8765",
    )

    assert report["success"] is True
    assert report["classification_accuracy_claimed"] is False
    assert report["predicted_object"] is None
    assert report["expected_object"] == "垃圾桶"


def test_nonfinite_action_is_a_recognition_pipeline_failure(tmp_path: Path):
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))

    report = validate_policy_response(
        expected_object="海绵",
        image_path=image,
        instruction="抓取海绵",
        action=[0.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 1.0],
        latency_seconds=0.7,
        endpoint="127.0.0.1:8765",
    )

    assert report["success"] is False
    assert "nonfinite_action" in report["errors"]


def test_missing_image_is_a_recognition_pipeline_failure(tmp_path: Path):
    report = validate_policy_response(
        expected_object="海绵",
        image_path=tmp_path / "missing.png",
        instruction="抓取海绵",
        action=[0.0] * 7,
        latency_seconds=0.7,
        endpoint="127.0.0.1:8765",
    )

    assert report["success"] is False
    assert "image_missing" in report["errors"]
