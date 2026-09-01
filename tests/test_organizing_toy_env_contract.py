"""Pure contract tests for the first formal Task-2 RoboCasa environment."""

from __future__ import annotations

import unittest

from tools.skill_transfer.robocasa_envs import formal_env_spec


class OrganizingToyEnvironmentContractTests(unittest.TestCase):
    def test_relation_maps_to_toy_counter_to_cabinet(self) -> None:
        spec = formal_env_spec("organizing::toy")
        self.assertEqual("PickPlaceCounterToCabinet", spec["robocasa_task"])
        self.assertEqual("toy", spec["object_group"])
        self.assertEqual("toy", spec["target_object"])
        self.assertEqual("cabinet", spec["target_region"])
        self.assertEqual("storage", spec["canonical_target"])
        self.assertEqual("organizing_toy_v1", spec["success_predicate_version"])

    def test_unimplemented_relation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not implemented"):
            formal_env_spec("organizing::book")


if __name__ == "__main__":
    unittest.main()
