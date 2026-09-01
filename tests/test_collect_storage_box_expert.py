"""Tests for the storage-box specialization of the formal expert collector."""

from __future__ import annotations

import unittest

import numpy as np


class CollectStorageBoxExpertTests(unittest.TestCase):
    def test_storage_box_oracle_lifts_immediately_after_confirmed_grasp(self) -> None:
        from tools.pick_place_oracle import PickPlaceSnapshot
        from tools.pick_place_oracle.storage_box import StorageBoxPickPlaceOracle

        oracle = StorageBoxPickPlaceOracle(
            np.array([0.4, 0.0, 0.3]),
            np.array([0.5, 0.0, 0.3]),
            np.array([0.3, 0.0, 0.3]),
        )
        oracle._enter("close_gripper")
        snapshot = PickPlaceSnapshot(
            eef_position=np.array([0.0, 0.0, 0.1]),
            object_position=np.array([0.0, 0.0, 0.1]),
            grasped=True,
        )

        decision = oracle.decide(snapshot)

        self.assertEqual("lift_object", decision.phase)
        self.assertEqual(0, oracle.phase_steps)
        self.assertTrue(np.allclose([0.0, 0.0, 0.2], oracle.lift_target))

    def test_storage_box_oracle_uses_verified_container_parameters(self) -> None:
        from tools.pick_place_oracle.storage_box import StorageBoxPickPlaceOracle

        self.assertEqual(0.0, StorageBoxPickPlaceOracle.GRASP_HEIGHT_OFFSET)
        self.assertEqual(0.10, StorageBoxPickPlaceOracle.LIFT_HEIGHT)
        self.assertEqual(0.30, StorageBoxPickPlaceOracle.CLOSED_TRANSLATION_LIMIT)

    def test_configures_exact_relation_actions_and_adaptive_oracle(self) -> None:
        import tools.collect_formal_skill_expert as collector
        from tools.collect_storage_box_expert import (
            EXPECTED_ACTIONS,
            EXPECTED_RELATION,
            configure_collector,
        )
        from tools.pick_place_oracle.storage_box import (
            StorageBoxPickPlaceOracle,
        )

        original_relation = collector._BASE.EXPECTED_RELATION
        original_actions = collector._BASE.EXPECTED_ACTIONS
        original_oracle = collector._BASE.PickPlaceOracle
        original_waypoints = collector._BASE.cabinet_waypoints
        try:
            installed = configure_collector()

            self.assertIs(installed, StorageBoxPickPlaceOracle)
            self.assertEqual(EXPECTED_RELATION, collector._BASE.EXPECTED_RELATION)
            self.assertEqual(EXPECTED_ACTIONS, collector._BASE.EXPECTED_ACTIONS)
            skill_ir = {
                "relation_key": EXPECTED_RELATION,
                "instruction": "put away the storage box in the cabinet",
                "canonical_actions": list(EXPECTED_ACTIONS),
            }
            collector._BASE.validate_skill_ir_for_collection(skill_ir)
            report = collector._BASE.build_formal_report(
                skill_ir,
                seed=5,
                success=False,
                executed_steps=10,
                predicates={"simulator_success": False},
            )
            self.assertEqual(EXPECTED_RELATION, report["relation_key"])
            self.assertFalse(report["counts_toward_task2_coverage"])
        finally:
            collector._BASE.EXPECTED_RELATION = original_relation
            collector._BASE.EXPECTED_ACTIONS = original_actions
            collector._BASE.PickPlaceOracle = original_oracle
            collector._BASE.cabinet_waypoints = original_waypoints


if __name__ == "__main__":
    unittest.main()
