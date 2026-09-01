from __future__ import annotations

import unittest

from tools.build_video_conditioned_skill_ir import select_clip_record


class BuildVideoConditionedSkillIRTests(unittest.TestCase):
    def test_selects_unique_clip_from_payload(self) -> None:
        payload = {"records": [{"clip_id": "a"}, {"clip_id": "b"}]}
        self.assertEqual(select_clip_record(payload, "b"), {"clip_id": "b"})

    def test_rejects_missing_or_duplicate_clip(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly once"):
            select_clip_record({"records": []}, "a")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            select_clip_record(
                {"records": [{"clip_id": "a"}, {"clip_id": "a"}]}, "a"
            )

    def test_accepts_single_record_pose_file(self) -> None:
        self.assertEqual(
            select_clip_record({"clip_id": "a", "pose_status": "available"}, "a"),
            {"clip_id": "a", "pose_status": "available"},
        )


if __name__ == "__main__":
    unittest.main()
