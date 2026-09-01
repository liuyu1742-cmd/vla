import json
import unittest
from pathlib import Path


class SkillCoverageContractTests(unittest.TestCase):
    def test_candidate_requires_source_rows(self):
        from tools.skill_coverage.contracts import validate_object_candidate

        with self.assertRaisesRegex(ValueError, "source_rows"):
            validate_object_candidate(
                {
                    "object_id": "water_cup",
                    "display_name": "水杯",
                    "dataset_task_groups": ["food_serving"],
                    "source_rows": [],
                }
            )

    def test_candidate_requires_dataset_task_group(self):
        from tools.skill_coverage.contracts import validate_object_candidate

        with self.assertRaisesRegex(ValueError, "dataset_task_groups"):
            validate_object_candidate(
                {
                    "object_id": "water_cup",
                    "display_name": "水杯",
                    "dataset_task_groups": [],
                    "source_rows": [98],
                }
            )

    def test_not_applicable_pair_cannot_pass(self):
        from tools.skill_coverage.contracts import validate_matrix_entry

        with self.assertRaisesRegex(ValueError, "not_applicable"):
            validate_matrix_entry(
                {
                    "service_task_id": "security_entry",
                    "object_id": "water_cup",
                    "applicability": "not_applicable",
                    "validation_state": "passed",
                }
            )

    def test_needs_review_pair_cannot_count_as_passed(self):
        from tools.skill_coverage.contracts import validate_matrix_entry

        with self.assertRaisesRegex(ValueError, "needs_review"):
            validate_matrix_entry(
                {
                    "service_task_id": "item_delivery",
                    "object_id": "water_cup",
                    "applicability": "needs_review",
                    "validation_state": "passed",
                }
            )

    def test_authoritative_catalogs_have_unique_fifteen_item_taxonomies(self):
        data_dir = Path(__file__).resolve().parents[1] / "data" / "skill_coverage"
        service_tasks = json.loads(
            (data_dir / "service_tasks.json").read_text(encoding="utf-8")
        )
        primitives = json.loads(
            (data_dir / "action_primitives.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(service_tasks), 15)
        self.assertEqual(len({item["service_task_id"] for item in service_tasks}), 15)
        self.assertEqual(len(primitives), 15)
        self.assertEqual(len({item["primitive_id"] for item in primitives}), 15)

    def test_applicability_policy_references_known_tasks_and_all_dataset_groups(self):
        data_dir = Path(__file__).resolve().parents[1] / "data" / "skill_coverage"
        service_tasks = json.loads(
            (data_dir / "service_tasks.json").read_text(encoding="utf-8")
        )
        policy = json.loads(
            (data_dir / "applicability_policy.json").read_text(encoding="utf-8")
        )
        task_ids = {item["service_task_id"] for item in service_tasks}
        expected_dataset_groups = {
            "cleaning", "organizing", "smart_cooking", "appliance_management",
            "security_monitoring", "laundry", "waste_disposal", "clothing_care",
            "window_care", "bedroom_service", "food_serving", "object_fetching",
            "elderly_assistance", "maintenance_management", "entertainment_service",
        }
        self.assertEqual(set(policy["primary_by_dataset_task"]), expected_dataset_groups)
        referenced = set(policy["primary_by_dataset_task"].values())
        for rules in policy["secondary_by_object"].values():
            referenced.update(rule["service_task_id"] for rule in rules)
        self.assertTrue(referenced <= task_ids)
        self.assertEqual(referenced, task_ids)

        fire_rules = {
            (rule["service_task_id"], rule["applicability"])
            for rule in policy["secondary_by_object"]["fire_alarm"]
        }
        self.assertIn(("security_entry", "not_applicable"), fire_rules)
        self.assertIn(("security_environment", "direct"), fire_rules)


if __name__ == "__main__":
    unittest.main()
