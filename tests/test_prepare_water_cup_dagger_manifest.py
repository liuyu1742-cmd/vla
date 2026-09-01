import unittest

from tools.prepare_water_cup_dagger_manifest import build_manifest


class PrepareWaterCupDaggerManifestTests(unittest.TestCase):
    def test_keeps_nominal_and_recovery_but_never_trains_on_seed_two(self) -> None:
        nominal = [
            {"seed": 0, "episode": "nominal0.npz", "success": True},
            {"seed": 2, "episode": "heldout2.npz", "success": True},
        ]
        recovery = [
            {"seed": 0, "episode": "dagger0.npz", "source": "dagger_recovery"},
            {"seed": 2, "episode": "forbidden2.npz", "source": "dagger_recovery"},
        ]

        manifest = build_manifest(nominal, recovery, heldout_seed=2, recovery_repeat=3)

        self.assertTrue(all(item["seed"] != 2 for item in manifest["train"]))
        self.assertEqual([item["episode"] for item in manifest["held_out"]], ["heldout2.npz"])
        sources = [item["source"] for item in manifest["train"]]
        self.assertIn("nominal_expert", sources)
        self.assertEqual(sources.count("dagger_recovery"), 3)


if __name__ == "__main__":
    unittest.main()
