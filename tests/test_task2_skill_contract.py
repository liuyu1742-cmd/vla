"""Tests for the read-only Task-2 task/object/action contract."""

from __future__ import annotations

import unittest


def excel_row(
    task_id: str,
    task_name: str,
    object_id: str,
    display_name: str,
) -> dict[str, str]:
    return {
        "task_id": task_id,
        "task_name": task_name,
        "object": object_id,
        "物体中文名称": display_name,
        "images": "1",
        "videos": "2",
        "annotations": "3",
        "metadata": "4",
        "object_dir": f"C:\\Task2\\{task_id}\\{object_id}",
    }


def benchmark_record(
    task_id: str,
    object_id: str,
    actions: list[str],
) -> dict[str, object]:
    return {
        "id": f"acceptance_{task_id}_{object_id}",
        "relation_key": f"{task_id}::{object_id}",
        "instruction": f"execute {task_id} for {object_id}",
        "acceptance_task": task_id,
        "legacy_task_id": None,
        "object": object_id,
        "target": "none",
        "actions": actions,
    }


TOY_ACTIONS = [
    "locate(toy)",
    "grasp(toy)",
    "move(storage)",
    "place(toy)",
]


class Task2SkillContractTests(unittest.TestCase):
    def test_builds_relation_with_source_and_canonical_actions(self) -> None:
        from tools.skill_coverage.task2_contract import build_task2_contract

        records = build_task2_contract(
            [excel_row("organizing", "整理任务", "toy", "玩具")],
            [benchmark_record("organizing", "toy", TOY_ACTIONS)],
        )

        self.assertEqual(1, len(records))
        self.assertEqual("organizing::toy", records[0]["relation_key"])
        self.assertEqual("整理任务", records[0]["task_name"])
        self.assertEqual("玩具", records[0]["object_name"])
        self.assertEqual(TOY_ACTIONS, records[0]["canonical_actions"])
        self.assertEqual(2, records[0]["source_row"])
        self.assertEqual(1, records[0]["media_counts"]["images"])

    def test_rejects_benchmark_relation_missing_from_excel(self) -> None:
        from tools.skill_coverage.task2_contract import build_task2_contract

        with self.assertRaisesRegex(ValueError, "benchmark-only"):
            build_task2_contract(
                [excel_row("organizing", "整理任务", "toy", "玩具")],
                [
                    benchmark_record("organizing", "toy", TOY_ACTIONS),
                    benchmark_record(
                        "object_fetching",
                        "book",
                        ["locate(book)", "grasp(book)", "move(person)", "release(book)"],
                    ),
                ],
            )

    def test_rejects_excel_relation_missing_from_benchmark(self) -> None:
        from tools.skill_coverage.task2_contract import build_task2_contract

        with self.assertRaisesRegex(ValueError, "excel-only"):
            build_task2_contract(
                [
                    excel_row("organizing", "整理任务", "toy", "玩具"),
                    excel_row("object_fetching", "物品取送", "book", "书"),
                ],
                [benchmark_record("organizing", "toy", TOY_ACTIONS)],
            )

    def test_rejects_malformed_relation_key(self) -> None:
        from tools.skill_coverage.task2_contract import build_task2_contract

        record = benchmark_record("organizing", "toy", TOY_ACTIONS)
        record["relation_key"] = "organizing__toy"
        with self.assertRaisesRegex(ValueError, "relation_key"):
            build_task2_contract(
                [excel_row("organizing", "整理任务", "toy", "玩具")],
                [record],
            )

    def test_rejects_duplicate_benchmark_relation(self) -> None:
        from tools.skill_coverage.task2_contract import build_task2_contract

        record = benchmark_record("organizing", "toy", TOY_ACTIONS)
        with self.assertRaisesRegex(ValueError, "duplicate benchmark relation"):
            build_task2_contract(
                [excel_row("organizing", "整理任务", "toy", "玩具")],
                [record, dict(record)],
            )

    def test_rejects_empty_actions(self) -> None:
        from tools.skill_coverage.task2_contract import build_task2_contract

        with self.assertRaisesRegex(ValueError, "actions"):
            build_task2_contract(
                [excel_row("organizing", "整理任务", "toy", "玩具")],
                [benchmark_record("organizing", "toy", [])],
            )

    def test_rejects_malformed_action_call(self) -> None:
        from tools.skill_coverage.task2_contract import parse_action_call

        with self.assertRaisesRegex(ValueError, "action call"):
            parse_action_call("move storage")

    def test_preserves_same_object_in_multiple_tasks(self) -> None:
        from tools.skill_coverage.task2_contract import build_task2_contract

        rows = [
            excel_row("appliance_management", "家电综合管理", "fan", "风扇"),
            excel_row("maintenance_management", "维护管理", "fan", "风扇"),
        ]
        benchmark = [
            benchmark_record(
                "appliance_management",
                "fan",
                ["locate(fan)", "turn_on(fan)", "inspect(fan)"],
            ),
            benchmark_record(
                "maintenance_management",
                "fan",
                [
                    "locate(fan)",
                    "inspect(fan)",
                    "clean_blades(fan)",
                    "maintain(fan)",
                    "confirm_state(fan)",
                ],
            ),
        ]

        records = build_task2_contract(rows, benchmark)

        self.assertEqual(
            ["appliance_management::fan", "maintenance_management::fan"],
            [record["relation_key"] for record in records],
        )
        self.assertNotEqual(
            records[0]["canonical_actions"], records[1]["canonical_actions"]
        )


if __name__ == "__main__":
    unittest.main()
