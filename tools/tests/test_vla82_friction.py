"""Material-aware contact friction must distinguish textured spray bottles."""

from __future__ import annotations

import unittest

from tools.vla82_full_sim.friction import profile_for


class MaterialFrictionTest(unittest.TestCase):
    def test_spray_bottle_uses_textured_plastic_grip_profile(self) -> None:
        profile = profile_for("VLA82-004", "spray")
        self.assertEqual(profile.material, "textured_plastic")
        self.assertEqual(profile.sliding, 0.70)


if __name__ == "__main__":
    unittest.main()
