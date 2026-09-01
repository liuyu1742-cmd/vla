from __future__ import annotations

import unittest

import numpy as np

from tools.extract_human_keypoints import (
    COCO_KEYPOINT_NAMES,
    select_person_keypoints,
    summarize_keypoint_track,
)


class HumanKeypointExtractionTests(unittest.TestCase):
    def test_selects_highest_scoring_person_and_normalizes_coordinates(self) -> None:
        keypoints = np.zeros((2, 17, 3), dtype=np.float32)
        keypoints[1, :, 0] = 100.0
        keypoints[1, :, 1] = 50.0
        prediction = {
            "scores": np.asarray([0.6, 0.95], dtype=np.float32),
            "keypoints": keypoints,
            "keypoints_scores": np.stack(
                [np.full(17, 0.4), np.full(17, 0.9)]
            ).astype(np.float32),
        }
        selected = select_person_keypoints(
            prediction, width=200, height=100, person_threshold=0.7
        )
        self.assertIsNotNone(selected)
        np.testing.assert_allclose(selected[:, 0], 0.5)
        np.testing.assert_allclose(selected[:, 1], 0.5)
        np.testing.assert_allclose(selected[:, 2], 0.9)

    def test_rejects_low_person_score(self) -> None:
        prediction = {
            "scores": np.asarray([0.2], dtype=np.float32),
            "keypoints": np.zeros((1, 17, 3), dtype=np.float32),
            "keypoints_scores": np.zeros((1, 17), dtype=np.float32),
        }
        self.assertIsNone(
            select_person_keypoints(
                prediction, width=100, height=100, person_threshold=0.7
            )
        )

    def test_summary_requires_two_nonzero_confident_frames(self) -> None:
        track = np.ones((2, len(COCO_KEYPOINT_NAMES), 3), dtype=np.float32)
        track[..., :2] *= 0.5
        track[..., 2] *= 0.9
        summary = summarize_keypoint_track(track, sampled_frames=3)
        self.assertEqual(summary["pose_status"], "available")
        self.assertEqual(summary["detected_person_frames"], 2)

        insufficient = summarize_keypoint_track(track[:1], sampled_frames=3)
        self.assertEqual(insufficient["pose_status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
