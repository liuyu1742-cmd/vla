from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.formal_skill_rollout import adapter_fingerprint


class FormalSkillAdapterFingerprintTests(unittest.TestCase):
    def test_training_checkpoints_do_not_change_deployed_adapter_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter / "formal_skill_action_stats.json").write_text(
                "{}", encoding="utf-8"
            )
            checkpoint = adapter / "checkpoints" / "checkpoint_step_000500"
            checkpoint.mkdir(parents=True)
            state = checkpoint / "optimizer.pt"
            state.write_bytes(b"first")
            first = adapter_fingerprint(adapter)
            state.write_bytes(b"changed optimizer state")
            second = adapter_fingerprint(adapter)
            self.assertEqual(first, second)

    def test_final_adapter_change_changes_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            model = adapter / "adapter_model.safetensors"
            model.write_bytes(b"first")
            first = adapter_fingerprint(adapter)
            model.write_bytes(b"second")
            self.assertNotEqual(first, adapter_fingerprint(adapter))


if __name__ == "__main__":
    unittest.main()
