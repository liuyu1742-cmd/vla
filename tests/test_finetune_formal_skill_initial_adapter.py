from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.finetune_formal_skill_openvla_resumable import (
    validate_initial_adapter,
)


class FormalSkillInitialAdapterTests(unittest.TestCase):
    def test_requires_adapter_config_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "adapter_model"):
                validate_initial_adapter(adapter, resume_checkpoint=None)

    def test_returns_resolved_complete_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

            selected = validate_initial_adapter(
                adapter, resume_checkpoint=None
            )

            self.assertEqual(adapter.resolve(), selected)

    def test_rejects_initial_adapter_with_resume_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            checkpoint = root / "checkpoint"
            adapter.mkdir()
            checkpoint.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

            with self.assertRaisesRegex(ValueError, "resume"):
                validate_initial_adapter(
                    adapter, resume_checkpoint=checkpoint
                )


if __name__ == "__main__":
    unittest.main()
