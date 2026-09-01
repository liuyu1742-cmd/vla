"""Regression test for the evidence scope emitted by water-cup smoke runs."""

from __future__ import annotations

import unittest


class WaterCupEvidenceScopeTests(unittest.TestCase):
    def test_report_is_explicitly_infrastructure_only(self) -> None:
        from tools.run_openvla_robocasa_water_cup import build_report

        report = build_report(7, "pick up the cup")

        self.assertEqual("infrastructure_smoke", report["evidence_scope"])
        self.assertIsNone(report["relation_key"])
        self.assertFalse(report["counts_toward_task2_coverage"])


if __name__ == "__main__":
    unittest.main()
