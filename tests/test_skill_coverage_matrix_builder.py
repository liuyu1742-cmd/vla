import json
import unittest
from pathlib import Path


def load_catalogs():
    root = Path(__file__).resolve().parents[1] / "data" / "skill_coverage"
    tasks = json.loads((root / "service_tasks.json").read_text(encoding="utf-8"))
    policy = json.loads(
        (root / "applicability_policy.json").read_text(encoding="utf-8")
    )
    return tasks, policy


def find_entry(matrix, service_task_id, object_id):
    return next(
        item
        for item in matrix
        if item["service_task_id"] == service_task_id
        and item["object_id"] == object_id
    )


class MatrixBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.policy = load_catalogs()
        self.candidates = [
            {
                "object_id": "water_cup",
                "display_name": "水杯",
                "dataset_task_groups": ["food_serving"],
                "source_rows": [98],
            },
            {
                "object_id": "fire_alarm",
                "display_name": "烟雾报警器",
                "dataset_task_groups": ["security_monitoring"],
                "source_rows": [52],
            },
            {
                "object_id": "air_conditioner",
                "display_name": "空调",
                "dataset_task_groups": [
                    "appliance_management",
                    "maintenance_management",
                ],
                "source_rows": [44, 125],
            },
        ]

    def test_builds_explicit_sparse_matrix(self):
        from tools.skill_coverage.matrix_builder import build_task_object_matrix

        matrix = build_task_object_matrix(
            self.tasks, self.candidates, self.policy
        )
        self.assertEqual(len(matrix), 15 * 3)
        self.assertEqual(
            find_entry(matrix, "item_delivery", "water_cup")["applicability"],
            "direct",
        )
        self.assertEqual(
            find_entry(matrix, "security_entry", "water_cup")["applicability"],
            "not_applicable",
        )
        self.assertEqual(
            find_entry(matrix, "security_environment", "fire_alarm")[
                "applicability"
            ],
            "direct",
        )
        self.assertEqual(
            find_entry(matrix, "facility_monitoring", "fire_alarm")[
                "applicability"
            ],
            "device_control",
        )
        self.assertEqual(
            find_entry(matrix, "security_entry", "fire_alarm")["applicability"],
            "not_applicable",
        )

    def test_appliance_secondary_routes_do_not_leak_into_cooking(self):
        from tools.skill_coverage.matrix_builder import build_task_object_matrix

        matrix = build_task_object_matrix(
            self.tasks, self.candidates, self.policy
        )
        expected = {
            "appliance_management": "direct",
            "environment_adjustment": "device_control",
            "energy_management": "device_control",
            "facility_monitoring": "device_control",
            "scene_linkage": "composite_resource",
            "cooking_assistance": "not_applicable",
        }
        for task_id, applicability in expected.items():
            self.assertEqual(
                find_entry(matrix, task_id, "air_conditioner")["applicability"],
                applicability,
            )

    def test_semantic_mapping_is_not_counted_as_execution_validation(self):
        from tools.skill_coverage.matrix_builder import (
            build_task_object_matrix,
            summarize_coverage,
        )

        matrix = build_task_object_matrix(
            self.tasks, self.candidates, self.policy
        )
        summary = summarize_coverage(
            self.tasks, self.candidates, matrix, acceptance_target=120
        )
        self.assertEqual(summary["semantically_mapped_object_count"], 3)
        self.assertEqual(summary["validated_object_count"], 0)
        self.assertEqual(summary["remaining_validation_gap"], 120)
        self.assertTrue(
            all(item["validation_state"] == "untested" for item in matrix)
        )


if __name__ == "__main__":
    unittest.main()
