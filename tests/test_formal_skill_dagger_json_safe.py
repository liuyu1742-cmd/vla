from __future__ import annotations

import json
import unittest

import numpy as np

from tools.formal_skill_dagger import json_safe


class FormalSkillDaggerJsonSafeTests(unittest.TestCase):
    def test_converts_nested_numpy_scalars_for_json(self) -> None:
        payload = {
            "gripper_gated": np.bool_(True),
            "distance": np.float32(0.25),
            "counts": [np.int64(3)],
        }

        converted = json_safe(payload)

        self.assertEqual(
            {
                "gripper_gated": True,
                "distance": 0.25,
                "counts": [3],
            },
            converted,
        )
        self.assertIn('"gripper_gated": true', json.dumps(converted))


if __name__ == "__main__":
    unittest.main()
