from __future__ import annotations

import unittest

import numpy as np

from tools.extract_human_keypoints import COCO_KEYPOINT_NAMES
from tools.extract_task2_human_video_keypoints import (
    build_segment_jobs,
    summarize_egocentric_track,
)


class Task2HumanVideoKeypointTests(unittest.TestCase):
    def test_egocentric_summary_accepts_confident_moving_wrists(self) -> None:
        track = np.zeros((4, len(COCO_KEYPOINT_NAMES), 3), dtype=np.float32)
        for frame in range(4):
            track[frame, 9, :] = [0.2 + 0.05 * frame, 0.4, 0.9]
            track[frame, 10, :] = [0.7 - 0.04 * frame, 0.4, 0.8]
        summary = summarize_egocentric_track(track, sampled_frames=4)
        self.assertEqual("available", summary["pose_status"])
        self.assertEqual(4, summary["frames_with_confident_wrist"])
        self.assertGreater(summary["wrist_trajectory_magnitude"], 0.0)
        self.assertLess(summary["mean_keypoint_confidence"], 0.2)

    def test_egocentric_summary_rejects_face_only_detection(self) -> None:
        track = np.zeros((4, len(COCO_KEYPOINT_NAMES), 3), dtype=np.float32)
        track[:, :5, :2] = 0.5
        track[:, :5, 2] = 0.95
        summary = summarize_egocentric_track(track, sampled_frames=4)
        self.assertEqual("insufficient", summary["pose_status"])
        self.assertEqual(0, summary["frames_with_confident_wrist"])

    def test_manifest_jobs_are_unique_and_preserve_episode_membership(self) -> None:
        manifest = {
            "schema_version": "task2_human_video_intake_v1",
            "relation_key": "organizing::storage_box",
            "episodes": [
                {
                    "episode_id": "episode_a",
                    "segments": [
                        {
                            "record_id": "pick_a",
                            "phase_role": "pick",
                            "video_path": "C:/clips/pick.mp4",
                            "video_sha256": "a" * 64,
                        },
                        {
                            "record_id": "place_a",
                            "phase_role": "place",
                            "video_path": "C:/clips/place.mp4",
                            "video_sha256": "b" * 64,
                        },
                    ],
                },
                {
                    "episode_id": "episode_b",
                    "segments": [
                        {
                            "record_id": "pick_a",
                            "phase_role": "pick",
                            "video_path": "C:/clips/pick.mp4",
                            "video_sha256": "a" * 64,
                        }
                    ],
                },
            ],
        }
        jobs = build_segment_jobs(manifest)
        self.assertEqual(["pick_a", "place_a"], [item["record_id"] for item in jobs])
        self.assertEqual(
            ["episode_a", "episode_b"], jobs[0]["episode_ids"]
        )

    def test_rejects_conflicting_duplicate_segment_fingerprint(self) -> None:
        manifest = {
            "schema_version": "task2_human_video_intake_v1",
            "relation_key": "organizing::storage_box",
            "episodes": [
                {
                    "episode_id": "one",
                    "segments": [
                        {
                            "record_id": "same",
                            "phase_role": "pick",
                            "video_path": "C:/clips/a.mp4",
                            "video_sha256": "a" * 64,
                        }
                    ],
                },
                {
                    "episode_id": "two",
                    "segments": [
                        {
                            "record_id": "same",
                            "phase_role": "pick",
                            "video_path": "C:/clips/b.mp4",
                            "video_sha256": "b" * 64,
                        }
                    ],
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            build_segment_jobs(manifest)


if __name__ == "__main__":
    unittest.main()
