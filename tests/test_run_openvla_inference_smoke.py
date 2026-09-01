"""Validation tests for the local OpenVLA inference smoke command."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class OpenVlaInferenceSmokeTests(unittest.TestCase):
    def test_model_directory_requires_config_file(self) -> None:
        from tools.run_openvla_inference_smoke import validate_model_dir

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                validate_model_dir(Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()
