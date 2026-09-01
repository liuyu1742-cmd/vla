from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.formal_skill_action_codec import (
    ACTION_NORM_KEY,
    install_formal_skill_action_stats,
    load_formal_skill_action_stats,
)


class _Model:
    norm_stats = None


class FormalSkillActionCodecTests(unittest.TestCase):
    def _write(self, root: Path, *, key: str = ACTION_NORM_KEY) -> Path:
        path = root / "formal_skill_action_stats.json"
        path.write_text(
            json.dumps(
                {
                    "key": key,
                    "action": {
                        "q01": [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0],
                        "q99": [1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0],
                        "mask": [True, True, True, False, False, False, True],
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_loads_and_installs_exact_training_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            self._write(adapter)
            key, action = load_formal_skill_action_stats(adapter)
            model = _Model()
            installed = install_formal_skill_action_stats(model, adapter)

            self.assertEqual(key, ACTION_NORM_KEY)
            self.assertEqual(installed, ACTION_NORM_KEY)
            self.assertEqual(model.norm_stats[ACTION_NORM_KEY]["action"], action)

    def test_rejects_unexpected_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            self._write(adapter, key="bridge_orig")
            with self.assertRaisesRegex(ValueError, "normalization key"):
                load_formal_skill_action_stats(adapter)

    def test_rejects_non_seven_dimensional_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            path = self._write(adapter)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["action"]["mask"] = [True]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seven"):
                load_formal_skill_action_stats(adapter)


if __name__ == "__main__":
    unittest.main()
