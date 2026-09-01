from __future__ import annotations

import subprocess
import sys
import unittest


class GenerateSkillIRCliTests(unittest.TestCase):
    def test_official_module_help_is_warning_free(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-m",
                "tools.skill_transfer.generate_skill_ir",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("RuntimeWarning", result.stderr)
        self.assertIn("--relation-key", result.stdout)


if __name__ == "__main__":
    unittest.main()
