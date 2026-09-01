from __future__ import annotations

import unittest

from tools.skill_transfer.human_video_bridge import (
    assess_robotproject_record,
    build_video_conditioned_skill_ir,
    resolve_task2_relation,
)


def _contract() -> dict:
    return {
        "organizing::toy": {
            "relation_key": "organizing::toy",
            "task_id": "organizing",
            "task_name": "整理任务",
            "object_id": "toy",
            "object_name": "玩具",
            "instruction": "整理并归位玩具",
            "target": "none",
            "canonical_actions": [
                "locate(toy)", "grasp(toy)", "move(storage)", "place(toy)"
            ],
        }
    }


def _mapping() -> dict:
    return {
        "clip_id": "demo_001",
        "mapping_status": "mapped",
        "task_id": "organizing",
        "object": "toy",
        "actions": [
            "locate(toy)", "grasp(toy)", "move(storage)", "place(toy)"
        ],
        "clip_path": "clips/demo_001.mp4",
        "clip_start_seconds": 1.0,
        "clip_stop_seconds": 3.0,
    }


def _pose() -> dict:
    return {
        "clip_id": "demo_001",
        "pose_status": "available",
        "sampled_frames": 2,
        "detected_person_frames": 2,
        "mean_keypoint_confidence": 0.9,
        "keypoint_names": ["left_wrist", "right_wrist"],
        "frame_times": [1.0, 2.0],
        "keypoints": [
            [[0.2, 0.3, 0.9], [0.6, 0.3, 0.8]],
            [[0.3, 0.4, 0.9], [0.7, 0.4, 0.8]],
        ],
    }


class HumanVideoSkillBridgeTests(unittest.TestCase):
    def test_resolves_exact_task2_relation_and_action_order(self) -> None:
        relation = resolve_task2_relation(_mapping(), _contract())
        self.assertEqual(relation["relation_key"], "organizing::toy")

    def test_rejects_mapping_with_different_action_sequence(self) -> None:
        mapping = _mapping()
        mapping["actions"] = list(reversed(mapping["actions"]))
        with self.assertRaisesRegex(ValueError, "actions"):
            resolve_task2_relation(mapping, _contract())

    def test_summary_only_pose_is_not_claimed_as_motion_capture(self) -> None:
        pose = _pose()
        pose.pop("keypoints")
        readiness = assess_robotproject_record(_mapping(), pose, _contract())
        self.assertFalse(readiness["motion_capture_ready"])
        self.assertIn("missing_raw_keypoint_track", readiness["reasons"])

    def test_builds_valid_skill_ir_from_real_nonzero_keypoint_track(self) -> None:
        skill_ir = build_video_conditioned_skill_ir(
            _mapping(), _pose(), _contract(), contract_version="task2_test"
        )
        self.assertEqual(skill_ir["relation_key"], "organizing::toy")
        self.assertEqual(skill_ir["source_video"]["clip_id"], "demo_001")
        self.assertEqual(
            skill_ir["perception_evidence"]["format"], "human_keypoint_track_v1"
        )
        self.assertEqual(
            skill_ir["execution_request"]["status"], "ready_for_robot_policy"
        )


if __name__ == "__main__":
    unittest.main()
