import unittest


def source_row(task_id: str, task_name: str, object_id: str, display_name: str):
    return {
        "task_id": task_id,
        "task_name": task_name,
        "object": object_id,
        "物体中文名称": display_name,
        "images": "10",
        "videos": "2",
        "annotations": "12",
        "metadata": "1",
        "object_dir": f"C:\\dataset\\{task_id}\\{object_id}",
    }


class RegistryBuilderTests(unittest.TestCase):
    def test_deduplicates_object_but_preserves_all_source_relations(self):
        from tools.skill_coverage.registry_builder import build_source_registries

        rows = [
            source_row(
                "appliance_management", "家电综合管理", "remote_control", "遥控器"
            ),
            source_row("object_fetching", "物品取送", "remote_control", "遥控器"),
            source_row(
                "entertainment_service", "娱乐服务", "remote_control", "遥控器"
            ),
        ]
        result = build_source_registries(rows)

        self.assertEqual(len(result["dataset_tasks"]), 3)
        self.assertEqual(len(result["object_candidates"]), 1)
        self.assertEqual(len(result["source_relations"]), 3)
        candidate = result["object_candidates"][0]
        self.assertEqual(
            candidate["dataset_task_groups"],
            ["appliance_management", "entertainment_service", "object_fetching"],
        )
        self.assertEqual(candidate["source_rows"], [2, 3, 4])

    def test_rejects_inconsistent_chinese_name_for_same_object_id(self):
        from tools.skill_coverage.registry_builder import build_source_registries

        rows = [
            source_row("food_serving", "送餐饮水", "water_cup", "水杯"),
            source_row("cleaning", "清洁任务", "water_cup", "玻璃杯"),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent object name"):
            build_source_registries(rows)

    def test_rejects_duplicate_task_object_relation(self):
        from tools.skill_coverage.registry_builder import build_source_registries

        rows = [
            source_row("food_serving", "送餐饮水", "water_cup", "水杯"),
            source_row("food_serving", "送餐饮水", "water_cup", "水杯"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate relation"):
            build_source_registries(rows)


if __name__ == "__main__":
    unittest.main()
