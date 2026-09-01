from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.collect_formal_skill_dagger import (
    load_heldout_seeds,
    validate_collection_args,
)


class CollectFormalSkillDaggerTests(unittest.TestCase):
    def test_reads_and_enforces_manifest_heldout_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "formal_skill_training_manifest_v1",
                        "relation_key": "organizing::toy",
                        "train": [{"seed": 0}],
                        "held_out": [
                            {"seed": 101},
                            {"seed": 102},
                            {"seed": 103},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            heldout = load_heldout_seeds(manifest_path)

            self.assertEqual({101, 102, 103}, heldout)
            with self.assertRaisesRegex(ValueError, "held-out"):
                validate_collection_args(
                    seed=102, beta=0.5, steps=300, heldout_seeds=heldout
                )

    def test_rejects_invalid_beta_and_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "beta"):
            validate_collection_args(
                seed=42, beta=1.1, steps=300, heldout_seeds={101, 102, 103}
            )
        with self.assertRaisesRegex(ValueError, "steps"):
            validate_collection_args(
                seed=42, beta=0.5, steps=0, heldout_seeds={101, 102, 103}
            )


if __name__ == "__main__":
    unittest.main()
