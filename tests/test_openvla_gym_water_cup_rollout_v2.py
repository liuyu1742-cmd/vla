"""Regression tests for matching the water-cup expert environment."""

from __future__ import annotations

import unittest

from tools.openvla_gym_water_cup_rollout_v2 import install_object_scale


class _Environment:
    def _get_obj_cfgs(self):
        return [{"name": "obj"}, {"name": "distractor", "object_scale": 0.4}]


class WaterCupRolloutEnvironmentTests(unittest.TestCase):
    def test_scales_only_the_task_object(self) -> None:
        original = install_object_scale(_Environment, 0.7)
        try:
            configs = _Environment()._get_obj_cfgs()
            self.assertEqual(configs[0]["object_scale"], 0.7)
            self.assertEqual(configs[1]["object_scale"], 0.4)
        finally:
            _Environment._get_obj_cfgs = original


if __name__ == "__main__":
    unittest.main()
