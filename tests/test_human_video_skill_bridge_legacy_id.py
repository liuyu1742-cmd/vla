from __future__ import annotations

import unittest

from tools.skill_transfer.human_video_bridge import resolve_task2_relation


class HumanVideoLegacyRelationTests(unittest.TestCase):
    def test_unique_benchmark_id_migrates_legacy_task_id(self) -> None:
        contract = {
            "cleaning::plate": {
                "relation_key": "cleaning::plate",
                "benchmark_id": "acceptance_053_cleaning_plate",
                "task_id": "cleaning",
                "legacy_task_id": "dish_washing",
                "object_id": "plate",
                "canonical_actions": ["locate(plate)", "wash(plate)"],
            }
        }
        mapping = {
            "mapping_status": "mapped",
            "relation_id": "acceptance_053_cleaning_plate",
            "task_id": "dish_washing",
            "object": "plate",
            "actions": ["locate(plate)", "wash(plate)"],
        }
        relation = resolve_task2_relation(mapping, contract)
        self.assertEqual(relation["relation_key"], "cleaning::plate")

    def test_benchmark_id_cannot_override_object_mismatch(self) -> None:
        contract = {
            "cleaning::plate": {
                "relation_key": "cleaning::plate",
                "benchmark_id": "acceptance_053_cleaning_plate",
                "task_id": "cleaning",
                "legacy_task_id": "dish_washing",
                "object_id": "plate",
                "canonical_actions": ["locate(plate)", "wash(plate)"],
            }
        }
        mapping = {
            "mapping_status": "mapped",
            "relation_id": "acceptance_053_cleaning_plate",
            "task_id": "dish_washing",
            "object": "bowl",
            "actions": ["locate(plate)", "wash(plate)"],
        }
        with self.assertRaisesRegex(ValueError, "object"):
            resolve_task2_relation(mapping, contract)


if __name__ == "__main__":
    unittest.main()
