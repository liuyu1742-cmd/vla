import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_water_cup_dagger_manifest_v2 import discover_recovery


class PrepareWaterCupDaggerManifestV2Tests(unittest.TestCase):
    def test_report_sidecar_is_not_treated_as_episode_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_path = root / "episode_seed_001_round_00.npz"
            (root / "episode_seed_001_round_00.json").write_text(
                json.dumps({"seed": 1, "samples": 4, "episode": str(episode_path)}),
                encoding="utf-8",
            )
            (root / "episode_seed_001_round_00_report.json").write_text(
                json.dumps({"success": True, "steps": []}), encoding="utf-8"
            )

            episodes = discover_recovery(root, inspect_orphans=False)

            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0]["seed"], 1)


if __name__ == "__main__":
    unittest.main()
