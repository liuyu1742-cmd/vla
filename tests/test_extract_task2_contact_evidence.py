from __future__ import annotations

import unittest

from tools.extract_task2_contact_evidence import (
    select_interaction_detection,
    summarize_interaction_frames,
)


class Task2ContactEvidenceTests(unittest.TestCase):
    def test_selector_rejects_full_frame_box_and_prefers_hand_nearby_box(self) -> None:
        candidates = [
            {
                "box": [0.0, 0.0, 1.0, 1.0],
                "confidence": 0.9,
                "class_name": "storage box",
            },
            {
                "box": [0.1, 0.1, 0.3, 0.3],
                "confidence": 0.3,
                "class_name": "container",
            },
            {
                "box": [0.7, 0.7, 0.9, 0.9],
                "confidence": 0.35,
                "class_name": "container",
            },
        ]
        selected = select_interaction_detection(
            candidates, [[0.2, 0.2], [0.22, 0.21]], maximum_box_area=0.85
        )
        self.assertEqual([0.1, 0.1, 0.3, 0.3], selected["box"])

    def test_summary_accepts_sparse_contact_plus_object_track(self) -> None:
        frames = [
            {
                "hands": [[[0.2, 0.2, 0.0]]],
                "detection": {
                    "box": [0.15, 0.15, 0.4, 0.4],
                    "confidence": 0.4,
                },
            },
            {
                "hands": [],
                "detection": {
                    "box": [0.2, 0.2, 0.45, 0.45],
                    "confidence": 0.4,
                },
            },
            {"hands": [], "detection": None},
            {
                "hands": [],
                "detection": {
                    "box": [0.25, 0.2, 0.5, 0.45],
                    "confidence": 0.4,
                },
            },
        ]
        summary = summarize_interaction_frames(
            frames,
            minimum_detection_ratio=0.5,
            minimum_contact_frames=1,
        )
        self.assertTrue(summary["interaction_evidence_ready"])
        self.assertEqual(1, summary["touching_frames"])
        self.assertEqual(3, summary["detected_object_frames"])

    def test_summary_rejects_object_only_without_any_hand_contact(self) -> None:
        frames = [
            {
                "hands": [],
                "detection": {
                    "box": [0.2, 0.2, 0.4, 0.4],
                    "confidence": 0.5,
                },
            }
            for _ in range(4)
        ]
        summary = summarize_interaction_frames(frames)
        self.assertFalse(summary["interaction_evidence_ready"])
        self.assertEqual(0, summary["near_or_touching_frames"])


if __name__ == "__main__":
    unittest.main()
