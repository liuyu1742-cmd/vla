import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.collect_water_cup_dagger_v2 import json_safe, save_episode


class CollectWaterCupDaggerV2Tests(unittest.TestCase):
    def test_numpy_scalars_are_json_safe(self) -> None:
        payload = json_safe({"gated": np.bool_(True), "count": np.int64(3)})
        self.assertEqual(json.loads(json.dumps(payload)), {"gated": True, "count": 3})

    def test_runtime_report_with_numpy_bool_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
            actions = np.zeros((1, 7), dtype=np.float32)
            _, manifest = save_episode(
                Path(directory),
                seed=0,
                frames=frames,
                oracle_actions=actions,
                policy_actions=actions,
                executed_actions=actions,
                report={"success": np.bool_(True), "gated": np.bool_(False)},
            )
            self.assertTrue(json.loads(manifest.read_text(encoding="utf-8"))["success"])


if __name__ == "__main__":
    unittest.main()
