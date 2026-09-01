"""Tests for building Task-2 human-video L2 Skill IR records."""

from __future__ import annotations

import copy
import unittest


def contract_record() -> dict[str, object]:
    return {
        "relation_key": "organizing::storage_box",
        "task_id": "organizing",
        "task_name": "整理任务",
        "object_id": "storage_box",
        "object_name": "收纳箱",
        "instruction": "整理任务：整理并归位收纳箱（storage_box）。",
        "target": "storage",
        "canonical_actions": [
            "locate(storage_box)",
            "grasp(storage_box)",
            "move(storage)",
            "place(storage_box)",
        ],
    }


def episode() -> dict[str, object]:
    return {
        "episode_id": "organizing__storage_box__example_1",
        "relation_key": "organizing::storage_box",
        "task_id": "organizing",
        "object_id": "storage_box",
        "video_id": "example",
        "phase_roles": ["pick", "place"],
        "canonical_actions": list(contract_record()["canonical_actions"]),
        "status": "ready_for_pose_extraction",
        "motion_retargeting_role": "reference_only",
        "segments": [
            {
                "record_id": "pick_1",
                "phase_role": "pick",
                "actions": ["locate(storage_box)", "grasp(storage_box)"],
                "video_path": "C:\\source\\pick.mp4",
                "video_sha256": "a" * 64,
                "annotation_path": "C:\\source\\pick.json",
                "annotation_sha256": "b" * 64,
            },
            {
                "record_id": "place_1",
                "phase_role": "place",
                "actions": ["move(target)", "place(storage_box)"],
                "video_path": "C:\\source\\place.mp4",
                "video_sha256": "c" * 64,
                "annotation_path": "C:\\source\\place.json",
                "annotation_sha256": "d" * 64,
            },
        ],
    }


def evidence_records() -> tuple[dict[str, dict[str, object]], ...]:
    pose = {
        key: {
            "record_id": key,
            "pose_status": "available",
            "track_path": f"C:\\evidence\\pose\\{key}.json",
            "track_sha256": character * 64,
        }
        for key, character in (("pick_1", "1"), ("place_1", "2"))
    }
    hands = {
        "pick_1": {
            "record_id": "pick_1",
            "hand_status": "available",
            "track_path": "C:\\evidence\\hands\\pick_1.json",
            "track_sha256": "3" * 64,
        },
        "place_1": {
            "record_id": "place_1",
            "hand_status": "insufficient",
            "track_path": "C:\\evidence\\hands\\place_1.json",
            "track_sha256": "4" * 64,
        },
    }
    contacts = {
        "pick_1": {
            "record_id": "pick_1",
            "phase_role": "pick",
            "interaction_evidence_ready": True,
            "object_track_ready": True,
            "segment_evidence_ready": True,
            "contact_path": "C:\\evidence\\contact\\pick_1.json",
            "contact_sha256": "5" * 64,
        },
        "place_1": {
            "record_id": "place_1",
            "phase_role": "place",
            "interaction_evidence_ready": False,
            "object_track_ready": True,
            "segment_evidence_ready": True,
            "contact_path": "C:\\evidence\\contact\\place_1.json",
            "contact_sha256": "6" * 64,
        },
    }
    return pose, hands, contacts


class Task2VideoSkillIRTests(unittest.TestCase):
    def test_builds_l2_record_without_execution_claim(self) -> None:
        from tools.build_task2_video_skill_ir import build_episode_skill_ir

        pose, hands, contacts = evidence_records()
        contract = {"organizing::storage_box": contract_record()}

        record = build_episode_skill_ir(
            episode(),
            pose_records=pose,
            hand_records=hands,
            contact_records=contacts,
            contract=contract,
            contract_version="task2_snapshot_test",
            intake_path="C:\\evidence\\intake.json",
            intake_sha256="7" * 64,
        )

        self.assertEqual("L2", record["source_video"]["evidence_level"])
        self.assertTrue(
            record["perception_evidence"]["pick_contact_visually_verified"]
        )
        self.assertFalse(
            record["perception_evidence"]["full_terminal_hand_contact_verified"]
        )
        self.assertEqual(
            "object_track_plus_exact_task2_place_or_insert_annotation",
            record["perception_evidence"]["terminal_evidence_policy"],
        )
        self.assertEqual(
            "ready_for_simulator_mapping", record["execution_request"]["status"]
        )
        self.assertEqual("reference_only", record["execution_request"]["motion_retargeting_role"])
        self.assertIsNone(record["execution_evidence"])

    def test_rejects_pick_without_true_hand_contact(self) -> None:
        from tools.build_task2_video_skill_ir import build_episode_skill_ir

        pose, hands, contacts = evidence_records()
        contacts["pick_1"]["interaction_evidence_ready"] = False

        with self.assertRaisesRegex(ValueError, "pick interaction evidence"):
            build_episode_skill_ir(
                episode(),
                pose_records=pose,
                hand_records=hands,
                contact_records=contacts,
                contract={"organizing::storage_box": contract_record()},
                contract_version="task2_snapshot_test",
                intake_path="C:\\evidence\\intake.json",
                intake_sha256="7" * 64,
            )

    def test_rejects_terminal_without_object_track(self) -> None:
        from tools.build_task2_video_skill_ir import build_episode_skill_ir

        pose, hands, contacts = evidence_records()
        contacts["place_1"]["object_track_ready"] = False
        contacts["place_1"]["segment_evidence_ready"] = False

        with self.assertRaisesRegex(ValueError, "terminal object track"):
            build_episode_skill_ir(
                episode(),
                pose_records=pose,
                hand_records=hands,
                contact_records=contacts,
                contract={"organizing::storage_box": contract_record()},
                contract_version="task2_snapshot_test",
                intake_path="C:\\evidence\\intake.json",
                intake_sha256="7" * 64,
            )

    def test_rejects_canonical_action_drift(self) -> None:
        from tools.build_task2_video_skill_ir import build_episode_skill_ir

        pose, hands, contacts = evidence_records()
        changed = copy.deepcopy(episode())
        changed["canonical_actions"] = ["locate(storage_box)"]

        with self.assertRaisesRegex(ValueError, "canonical_actions"):
            build_episode_skill_ir(
                changed,
                pose_records=pose,
                hand_records=hands,
                contact_records=contacts,
                contract={"organizing::storage_box": contract_record()},
                contract_version="task2_snapshot_test",
                intake_path="C:\\evidence\\intake.json",
                intake_sha256="7" * 64,
            )


if __name__ == "__main__":
    unittest.main()
