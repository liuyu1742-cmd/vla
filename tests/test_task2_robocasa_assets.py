"""Lightweight contracts for the project-owned RoboCasa toy asset."""

from __future__ import annotations

import unittest
from pathlib import Path

from tools.skill_transfer.robocasa_assets import (
    LOCAL_REGISTRY,
    TOY_CATEGORY,
    TOY_MODEL_XML,
    inspect_toy_asset,
    register_local_toy_assets,
)


class _FakeCategory:
    def __init__(self, path: Path) -> None:
        self.mjcf_paths = [str(path)]
        self.graspable = True


class Task2RoboCasaAssetTests(unittest.TestCase):
    def test_toy_mjcf_has_stable_visual_collision_and_bbox(self) -> None:
        report = inspect_toy_asset(TOY_MODEL_XML)
        self.assertEqual("task2_toy_block", report["model"])
        self.assertEqual([0.035, 0.03, 0.03], report["half_size"])
        self.assertTrue(report["has_visual_geom"])
        self.assertTrue(report["has_collision_geom"])
        self.assertEqual(0.12, report["mass"])
        self.assertEqual([0.9, 0.25, 0.08], report["friction"])

    def test_registration_is_idempotent_for_project_asset(self) -> None:
        categories: dict[str, object] = {}
        groups: dict[str, list[str]] = {}
        factory_calls: list[dict[str, object]] = []

        def factory(**kwargs: object) -> _FakeCategory:
            factory_calls.append(dict(kwargs))
            return _FakeCategory(TOY_MODEL_XML)

        first = register_local_toy_assets(categories, groups, factory=factory)
        second = register_local_toy_assets(categories, groups, factory=factory)

        self.assertIs(first, second)
        self.assertEqual(1, len(factory_calls))
        self.assertEqual([TOY_CATEGORY], groups[TOY_CATEGORY])
        self.assertIs(first, categories[TOY_CATEGORY][LOCAL_REGISTRY])

    def test_registration_refuses_to_overwrite_foreign_toy_category(self) -> None:
        categories = {TOY_CATEGORY: {"upstream": object()}}
        with self.assertRaisesRegex(ValueError, "refuse to overwrite"):
            register_local_toy_assets(categories, {}, factory=lambda **_: object())


if __name__ == "__main__":
    unittest.main()
