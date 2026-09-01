from __future__ import annotations

import unittest

from tools.extract_task2_hand_keypoints import summarize_hand_frames


def _hand(x: float) -> dict:
    return {
        "handedness": "Left",
        "handedness_score": 0.9,
        "landmarks": [
            [x + index * 0.001, 0.4 + index * 0.001, 0.0]
            for index in range(21)
        ],
    }


class Task2HandKeypointTests(unittest.TestCase):
    def test_summary_accepts_detected_moving_hand_track(self) -> None:
        frames = [
            {"timestamp_seconds": 0.0, "hands": [_hand(0.2)]},
            {"timestamp_seconds": 0.1, "hands": [_hand(0.25)]},
            {"timestamp_seconds": 0.2, "hands": [_hand(0.3)]},
            {"timestamp_seconds": 0.3, "hands": []},
        ]
        summary = summarize_hand_frames(
            frames,
            minimum_hand_frame_ratio=0.5,
            minimum_wrist_trajectory=0.02,
        )
        self.assertEqual("available", summary["hand_status"])
        self.assertEqual(3, summary["detected_hand_frames"])
        self.assertGreater(summary["wrist_trajectory_magnitude"], 0.02)

    def test_summary_rejects_empty_or_static_hands(self) -> None:
        empty = [
            {"timestamp_seconds": 0.0, "hands": []},
            {"timestamp_seconds": 0.1, "hands": []},
        ]
        self.assertEqual(
            "insufficient", summarize_hand_frames(empty)["hand_status"]
        )
        static = [
            {"timestamp_seconds": 0.0, "hands": [_hand(0.2)]},
            {"timestamp_seconds": 0.1, "hands": [_hand(0.2)]},
        ]
        self.assertEqual(
            "insufficient",
            summarize_hand_frames(
                static, minimum_wrist_trajectory=0.01
            )["hand_status"],
        )

    def test_summary_rejects_non_monotonic_timestamps(self) -> None:
        frames = [
            {"timestamp_seconds": 0.1, "hands": [_hand(0.2)]},
            {"timestamp_seconds": 0.1, "hands": [_hand(0.3)]},
        ]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            summarize_hand_frames(frames)


if __name__ == "__main__":
    unittest.main()
