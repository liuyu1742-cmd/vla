"""Locks the grasp height that produced bilateral contact and a stable lift."""

from __future__ import annotations

import unittest

from tools.collect_formal_skill_expert_calibrated import configure_toy_grasp


class ToyGraspCalibrationContractTests(unittest.TestCase):
    def test_formal_collector_uses_measured_one_centimeter_offset(self) -> None:
        oracle_class = configure_toy_grasp()
        self.assertEqual(0.010, oracle_class.GRASP_HEIGHT_OFFSET)
        self.assertEqual(0.028, oracle_class.CONTACT_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
