from __future__ import annotations

import unittest

import numpy as np

from tools.extract_task2_hand_object_evidence import (
    normalized_point_box_distance,
    summarize_hand_object_sequence,
)


class Task2HandObjectEvidenceTests(unittest.TestCase):
    def test_point_box_distance_is_zero_inside_and_metric_outside(self) -> None:
        box = [0.2, 0.2, 0.6, 0.6]
        self.assertEqual(0.0, normalized_point_box_distance([0.4, 0.4], box))
        self.assertAlmostEqual(
            np.sqrt(0.02),
            normalized_point_box_distance([0.1, 0.1], box),
        )

    def test_sequence_requires_object_coverage_and_visual_proximity(self) -> None:
        detections = [
            {"box": [0.2, 0.2, 0.6, 0.6], "confidence": 0.8},
            {"box": [0.21, 0.2, 0.61, 0.6], "confidence": 0.8},
            {"box": [0.22, 0.2, 0.62, 0.6], "confidence": 0.8},
            None,
        ]
        wrists = np.zeros((4, 2, 3), dtype=np.float32)
        wrists[:, 0, :] = [
            [0.1, 0.3, 0.9],
            [0.2, 0.3, 0.9],
            [0.3, 0.3, 0.9],
            [0.4, 0.3, 0.9],
        ]
        summary = summarize_hand_object_sequence(
            detections,
            wrists,
            wrist_confidence_threshold=0.5,
            near_distance=0.12,
            touching_distance=0.02,
            minimum_detection_ratio=0.5,
            minimum_near_frames=2,
        )
        self.assertTrue(summary["contact_evidence_ready"])
        self.assertEqual(3, summary["detected_object_frames"])
        self.assertGreaterEqual(summary["near_or_touching_frames"], 2)
        self.assertIn("touching", summary["contact_states"])

    def test_sequence_rejects_detector_only_evidence(self) -> None:
        detections = [
            {"box": [0.7, 0.7, 0.9, 0.9], "confidence": 0.8}
            for _ in range(4)
        ]
        wrists = np.zeros((4, 2, 3), dtype=np.float32)
        wrists[:, 0, :] = [0.1, 0.1, 0.9]
        summary = summarize_hand_object_sequence(
            detections,
            wrists,
            minimum_detection_ratio=0.5,
            minimum_near_frames=2,
        )
        self.assertFalse(summary["contact_evidence_ready"])
        self.assertEqual(0, summary["near_or_touching_frames"])


if __name__ == "__main__":
    unittest.main()
