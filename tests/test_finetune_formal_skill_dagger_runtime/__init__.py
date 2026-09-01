from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.finetune_formal_skill_openvla_dagger_r3 import (
    adapter_sha256,
    implementation_has_adapter_hash,
)


class FormalSkillDaggerRuntimeTests(unittest.TestCase):
    def test_dynamic_trainer_has_adapter_hash_helper(self) -> None:
        self.assertTrue(implementation_has_adapter_hash())

    def test_adapter_sha256_hashes_weights_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            path.write_bytes(b"adapter")

            self.assertEqual(
                "ae1eae1d76e5b7c865c4122ce366a08025842566d2d96c75cc13e6353a73db0d",
                adapter_sha256(path),
            )
