"""Tests for the read-only OpenVLA environment verifier."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class OpenVlaEnvironmentTests(unittest.TestCase):
    def test_missing_model_directory_is_reported_without_cuda_requirement(self) -> None:
        from tools.check_openvla_environment import build_report

        with tempfile.TemporaryDirectory() as temporary_directory:
            report = build_report(Path(temporary_directory))

        self.assertFalse(report["model_files_present"])
        self.assertIn("python_version", report)
        self.assertIn("cuda_available", report)


if __name__ == "__main__":
    unittest.main()
