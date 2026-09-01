"""Tests for the real-environment storage-box reset audit contract."""

from __future__ import annotations

import copy
import unittest


def valid_report() -> dict[str, object]:
    return {
        "schema_version": "formal_env_mapping_check_v1",
        "relation_key": "organizing::storage_box",
        "seed": 0,
        "reset_completed": True,
        "simulator_mapping_ready": True,
        "task_success_at_reset": False,
        "mapping": {
            "object_group": "tupperware",
            "simulator_object_proxy": "tupperware",
            "semantic_asset_proxy": True,
            "target_region": "cabinet",
        },
        "action_space": {"shape": [12]},
        "camera": {
            "key": "video.robot0_agentview_left",
            "shape": [256, 256, 3],
        },
        "initial_predicates": {
            "object_on_counter": True,
            "object_inside_cabinet": False,
            "gripper_far": True,
        },
        "object": {
            "category": "tupperware",
            "model_path": "C:\\assets\\tupperware\\Tupperware028\\model.xml",
            "size": [0.1, 0.08, 0.04],
            "position": [0.0, 0.0, 1.0],
        },
    }


class OrganizingStorageBoxEnvironmentAuditTests(unittest.TestCase):
    def test_reads_native_dimension_from_robosuite_action_spec(self) -> None:
        from tools.check_organizing_storage_box_env import native_action_shape

        shape = native_action_shape(([-1.0] * 12, [1.0] * 12))

        self.assertEqual([12], shape)

    def test_accepts_valid_reset_mapping_without_success_claim(self) -> None:
        from tools.check_organizing_storage_box_env import validate_reset_audit

        validated = validate_reset_audit(valid_report())

        self.assertTrue(validated["simulator_mapping_ready"])
        self.assertFalse(validated["task_success_at_reset"])

    def test_rejects_wrong_asset_category(self) -> None:
        from tools.check_organizing_storage_box_env import validate_reset_audit

        report = copy.deepcopy(valid_report())
        report["object"]["category"] = "toy"

        with self.assertRaisesRegex(ValueError, "tupperware"):
            validate_reset_audit(report)

    def test_rejects_non_native_action_shape(self) -> None:
        from tools.check_organizing_storage_box_env import validate_reset_audit

        report = copy.deepcopy(valid_report())
        report["action_space"]["shape"] = [7]

        with self.assertRaisesRegex(ValueError, "12-dimensional"):
            validate_reset_audit(report)

    def test_rejects_reset_that_is_already_successful(self) -> None:
        from tools.check_organizing_storage_box_env import validate_reset_audit

        report = copy.deepcopy(valid_report())
        report["task_success_at_reset"] = True

        with self.assertRaisesRegex(ValueError, "successful at reset"):
            validate_reset_audit(report)


if __name__ == "__main__":
    unittest.main()
