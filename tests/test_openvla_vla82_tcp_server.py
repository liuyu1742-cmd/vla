import json
import tempfile
import unittest
from pathlib import Path

from tools.openvla_vla82_tcp_server import load_action_stats


class OpenVLAVLA82ServerTests(unittest.TestCase):
    def test_loads_seven_channel_action_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "vla82_action_stats.json").write_text(
                json.dumps(
                    {
                        "key": "vla82_robocasa365_19class",
                        "action": {"q01": [-1] * 7, "q99": [1] * 7, "mask": [True] * 7},
                    }
                ),
                encoding="utf-8",
            )
            key, action = load_action_stats(path)
        self.assertEqual(key, "vla82_robocasa365_19class")
        self.assertEqual(len(action["q01"]), 7)


if __name__ == "__main__":
    unittest.main()
