from __future__ import annotations

import unittest

from tools.formal_skill_posttrain_pipeline import build_pipeline_summary


def _report(seed: int, success: bool) -> dict:
    phases = (
        ("locate", -1.0, 1.0),
        ("grasp", 1.0, 0.25),
        ("move", 1.0, 0.30),
        ("place", -1.0, 0.50),
    )
    trajectory = [
        {
            "phase": phase,
            "guarded_action": [limit, 0, 0, 0, 0, 0, grip],
        }
        for phase, grip, limit in phases
    ]
    return {
        "schema_version": "formal_skill_model_evaluation_v1",
        "evidence_scope": "formal_skill_model_evaluation",
        "relation_key": "organizing::toy",
        "counts_toward_task2_coverage": success,
        "seed": seed,
        "split": "held_out",
        "canonical_actions": [
            "locate(toy)", "grasp(toy)", "move(storage)", "place(toy)"
        ],
        "success": success,
        "success_predicates": {
            "simulator_success": success,
            "placed_in_storage": success,
            "gripper_released": success,
        },
        "max_decision_steps": 300,
        "executed_decision_steps": len(trajectory),
        "ever_grasped": success,
        "ever_inside": success,
        "manifest_sha256": "a" * 64,
        "skill_ir_sha256": "b" * 64,
        "adapter_sha256": "c" * 64,
        "trajectory": trajectory,
    }


class FormalSkillPosttrainPipelineTests(unittest.TestCase):
    def test_two_of_three_valid_heldout_rollouts_meet_gate(self) -> None:
        summary = build_pipeline_summary(
            {101: _report(101, True), 102: _report(102, False), 103: _report(103, True)},
            required_successes=2,
        )
        self.assertTrue(summary["acceptance_passed"])
        self.assertEqual(summary["passed_seeds"], [101, 103])
        self.assertEqual(summary["failed_seeds"], [102])

    def test_one_success_does_not_meet_gate(self) -> None:
        summary = build_pipeline_summary(
            {101: _report(101, True), 102: _report(102, False), 103: _report(103, False)},
            required_successes=2,
        )
        self.assertFalse(summary["acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
