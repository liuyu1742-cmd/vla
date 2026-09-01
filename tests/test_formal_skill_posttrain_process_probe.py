from __future__ import annotations

import os
import unittest

from tools.formal_skill_posttrain_pipeline import process_exists


class FormalSkillPosttrainProcessProbeTests(unittest.TestCase):
    def test_current_process_is_detected_without_signalling_it(self) -> None:
        self.assertTrue(process_exists(os.getpid()))

    def test_impossible_process_is_not_detected(self) -> None:
        self.assertFalse(process_exists(2_147_483_647))


if __name__ == "__main__":
    unittest.main()
