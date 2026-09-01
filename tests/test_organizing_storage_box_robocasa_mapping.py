"""Contract tests for the formal storage-box RoboCasa mapping."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch


class OrganizingStorageBoxMappingTests(unittest.TestCase):
    def test_relation_uses_disclosed_native_tupperware_proxy(self) -> None:
        from tools.skill_transfer.robocasa_envs import formal_env_spec

        spec = formal_env_spec("organizing::storage_box")

        self.assertEqual("PickPlaceCounterToCabinet", spec["robocasa_task"])
        self.assertEqual("tupperware", spec["object_group"])
        self.assertEqual("storage_box", spec["target_object"])
        self.assertEqual("tupperware", spec["simulator_object_proxy"])
        self.assertTrue(spec["semantic_asset_proxy"])
        self.assertEqual("cabinet", spec["target_region"])
        self.assertEqual("storage", spec["canonical_target"])
        self.assertEqual(
            "organizing_storage_box_v1", spec["success_predicate_version"]
        )
        self.assertEqual(0.45, spec["object_scale"])
        self.assertEqual(math.pi / 2.0, spec["object_rotation"])
        self.assertEqual(("lightwheel",), tuple(spec["object_registries"]))

    def test_scale_patch_only_changes_target_object(self) -> None:
        from tools.skill_transfer.robocasa_envs import install_object_scale

        class FakeEnvironment:
            def _get_obj_cfgs(self) -> list[dict[str, object]]:
                return [{"name": "obj"}, {"name": "distr_counter"}]

        original = install_object_scale(FakeEnvironment, 0.45)
        try:
            configs = FakeEnvironment()._get_obj_cfgs()
        finally:
            FakeEnvironment._get_obj_cfgs = original

        self.assertEqual(0.45, configs[0]["object_scale"])
        self.assertNotIn("object_scale", configs[1])

    def test_instance_scale_patch_survives_until_reset_time(self) -> None:
        from tools.skill_transfer.robocasa_envs import install_instance_object_config

        class FakeEnvironment:
            def _get_obj_cfgs(self) -> list[dict[str, object]]:
                return [
                    {"name": "obj", "placement": {"size": (0.6, 0.3)}},
                    {"name": "distr_counter", "placement": {}},
                ]

        environment = FakeEnvironment()
        install_instance_object_config(
            environment,
            object_scale=0.45,
            object_rotation=math.pi / 2.0,
        )

        configs = environment._get_obj_cfgs()
        self.assertEqual(0.45, configs[0]["object_scale"])
        self.assertEqual(
            math.pi / 2.0,
            configs[0]["placement"]["rotation"],
        )
        self.assertNotIn("object_scale", configs[1])
        self.assertNotIn("rotation", configs[1]["placement"])

    def test_create_uses_standard_registry_without_registering_toy_assets(self) -> None:
        from tools.skill_transfer.robocasa_envs import create_formal_env

        calls: list[tuple[str, dict[str, object]]] = []
        sentinel = object()

        def fake_gym_make(gym_id: str, **kwargs: object) -> object:
            calls.append((gym_id, kwargs))
            return sentinel

        with patch(
            "tools.skill_transfer.robocasa_envs.register_local_toy_assets"
        ) as register_toy:
            env, metadata = create_formal_env(
                "organizing::storage_box",
                seed=17,
                camera_name="camera0",
                gym_make=fake_gym_make,
            )

        self.assertIs(sentinel, env)
        register_toy.assert_not_called()
        self.assertEqual(1, len(calls))
        gym_id, kwargs = calls[0]
        self.assertEqual("robocasa/PickPlaceCounterToCabinet", gym_id)
        self.assertEqual("tupperware", kwargs["obj_groups"])
        self.assertEqual(("lightwheel",), tuple(kwargs["obj_registries"]))
        self.assertEqual(["camera0"], kwargs["camera_names"])
        self.assertEqual(17, metadata["seed"])

    def test_existing_toy_mapping_still_registers_local_assets(self) -> None:
        from tools.skill_transfer.robocasa_envs import create_formal_env

        with patch(
            "tools.skill_transfer.robocasa_envs.register_local_toy_assets"
        ) as register_toy:
            _, metadata = create_formal_env(
                "organizing::toy",
                seed=0,
                gym_make=lambda *args, **kwargs: object(),
            )

        register_toy.assert_called_once_with()
        self.assertIn("task2_local", metadata["object_registries"])


if __name__ == "__main__":
    unittest.main()
