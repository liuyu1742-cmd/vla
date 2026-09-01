from __future__ import annotations

import copy
import unittest

from tools.formal_skill_evaluation_evidence import validate_model_evaluation


def _valid_report() -> dict:
    trajectory = []
    for phase, gripper, limit in (
        ("locate", -1.0, 1.0),
        ("grasp", 1.0, 0.25),
        ("move", 1.0, 0.30),
        ("place", -1.0, 0.50),
    ):
        trajectory.append(
            {
                "phase": phase,
                "guarded_action": [limit, -limit, 0.0, 0.0, 0.0, 0.0, gripper],
            }
        )
    return {
        "schema_version": "formal_skill_model_evaluation_v1",
        "evidence_scope": "formal_skill_model_evaluation",
        "relation_key": "organizing::toy",
        "counts_toward_task2_coverage": True,
        "seed": 101,
        "split": "held_out",
        "canonical_actions": [
            "locate(toy)", "grasp(toy)", "move(storage)", "place(toy)"
        ],
        "success": True,
        "success_predicates": {
            "simulator_success": True,
            "placed_in_storage": True,
            "gripper_released": True,
        },
        "max_decision_steps": 300,
        "executed_decision_steps": 210,
        "ever_grasped": True,
        "ever_inside": True,
        "manifest_sha256": "a" * 64,
        "skill_ir_sha256": "b" * 64,
        "adapter_sha256": "c" * 64,
        "trajectory": trajectory,
    }


class FormalSkillEvaluationEvidenceTests(unittest.TestCase):
    def test_accepts_complete_heldout_300_step_evidence(self) -> None:
        result = validate_model_evaluation(_valid_report())
        self.assertEqual(result["seed"], 101)
        self.assertEqual(result["covered_phases"], ["locate", "grasp", "move", "place"])

    def test_rejects_training_seed_evidence(self) -> None:
        report = _valid_report()
        report["split"] = "train"
        with self.assertRaisesRegex(ValueError, "held_out"):
            validate_model_evaluation(report)

    def test_rejects_missing_real_task_predicate(self) -> None:
        report = _valid_report()
        report["success_predicates"]["placed_in_storage"] = False
        with self.assertRaisesRegex(ValueError, "predicates"):
            validate_model_evaluation(report)

    def test_rejects_unsafe_or_untrained_action_channels(self) -> None:
        report = copy.deepcopy(_valid_report())
        report["trajectory"][2]["guarded_action"][3] = 0.1
        with self.assertRaisesRegex(ValueError, "rotation"):
            validate_model_evaluation(report)


if __name__ == "__main__":
    unittest.main()
