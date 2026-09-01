"""Regression test for warning-free execution of the Skill IR module CLI."""

from __future__ import annotations

import subprocess
import sys
import unittest


class SkillIRCliTests(unittest.TestCase):
    def test_module_help_has_no_runtime_warning(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-m",
                "tools.skill_transfer.skill_ir",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("RuntimeWarning", result.stderr)


if __name__ == "__main__":
    unittest.main()
