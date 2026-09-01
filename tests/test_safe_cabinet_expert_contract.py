"""Locks the cabinet placement parameters proven by the bounded diagnostic."""

from __future__ import annotations

import unittest

from tools.collect_formal_skill_expert_safe import (
    SAFE_CABINET_DEPTH,
    configure_expert,
)


class SafeCabinetExpertContractTests(unittest.TestCase):
    def test_expert_uses_verified_depth_and_transit_limit(self) -> None:
        oracle_class = configure_expert()
        self.assertEqual(0.20, SAFE_CABINET_DEPTH)
        self.assertEqual(0.30, oracle_class.CLOSED_TRANSLATION_LIMIT)
        self.assertEqual(0.010, oracle_class.GRASP_HEIGHT_OFFSET)


if __name__ == "__main__":
    unittest.main()
