"""State-machine tests for the relation-agnostic pick/place expert."""

from __future__ import annotations

import unittest

import numpy as np

from tools.pick_place_oracle import PickPlaceOracle, PickPlaceSnapshot


class PickPlaceOracleTests(unittest.TestCase):
    def make_oracle(self) -> PickPlaceOracle:
        return PickPlaceOracle(
            destination_front=np.array([0.8, 0.0, 1.1]),
            destination_center=np.array([0.9, 0.0, 1.1]),
            destination_retreat=np.array([0.7, 0.0, 1.1]),
        )

    @staticmethod
    def snapshot(eef: object, obj: object, grasped: bool = False) -> PickPlaceSnapshot:
        return PickPlaceSnapshot(
            eef_position=np.asarray(eef, dtype=float),
            object_position=np.asarray(obj, dtype=float),
            grasped=grasped,
        )

    def test_approach_emits_locate_with_open_gripper(self) -> None:
        decision = self.make_oracle().decide(
            self.snapshot([0.0, 0.0, 0.0], [0.1, 0.2, 0.5])
        )
        self.assertEqual("approach_object", decision.phase)
        self.assertEqual("locate", decision.canonical_action)
        self.assertEqual((7,), decision.action.shape)
        self.assertEqual(0.0, decision.action[6])

    def test_contact_transitions_to_grasp(self) -> None:
        oracle = self.make_oracle()
        obj = np.array([0.1, 0.2, 0.5])
        oracle.decide(self.snapshot(obj + [0, 0, 0.18], obj))
        decision = oracle.decide(self.snapshot(obj + [0, 0, 0.002], obj))
        self.assertEqual("close_gripper", decision.phase)
        self.assertEqual("grasp", decision.canonical_action)
        self.assertEqual(1.0, decision.action[6])

    def test_failed_grasp_recovers_without_destination_hardcoding(self) -> None:
        oracle = self.make_oracle()
        obj = np.array([0.1, 0.2, 0.5])
        oracle.decide(self.snapshot(obj + [0, 0, 0.18], obj))
        contact = self.snapshot(obj + [0, 0, 0.002], obj)
        for _ in range(oracle.CLOSE_HOLD_STEPS):
            oracle.decide(contact)
        recovery = oracle.decide(contact)
        self.assertEqual("approach_object", recovery.phase)
        self.assertEqual("locate", recovery.canonical_action)
        self.assertEqual(1, oracle.grasp_attempts)


if __name__ == "__main__":
    unittest.main()
