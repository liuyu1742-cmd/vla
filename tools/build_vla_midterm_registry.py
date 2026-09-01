"""Build the midterm 8-task / 60-object registry from local VLA sources only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from robocasa.models.objects import kitchen_objects
from robocasa.models.objects.kitchen_objects import OBJ_GROUPS


TASK_METADATA = Path(
    "datasets/robocasa365_pretrain_atomic/meta/tasks.parquet"
)

SELECTED_TASK_OBJECTS: dict[str, list[str]] = {
    "PickPlaceCounterToCabinet": [
        "aluminum_foil",
        "bar_soap",
        "bottled_drink",
        "boxed_drink",
        "cereal",
        "cinnamon",
        "condiment_bottle",
        "jam",
        "teapot",
        "tupperware",
    ],
    "PickPlaceCounterToDrawer": [
        "dish_brush",
        "ladle",
        "measuring_cup",
        "peeler",
        "pizza_cutter",
        "rolling_pin",
        "tongs",
        "whisk",
        "wooden_spoon",
    ],
    "PickPlaceCounterToSink": [
        "avocado",
        "beer",
        "colander",
        "cucumber",
        "cup",
        "juice",
        "sponge",
        "water_bottle",
    ],
    "PickPlaceCounterToStove": [
        "bell_pepper",
        "cream_cheese_stick",
        "hot_dog",
        "lemon_wedge",
        "lime",
        "onion",
        "squash",
        "tomato",
    ],
    "PickPlaceCounterToMicrowave": [
        "carrot",
        "cheese",
        "chicken_drumstick",
        "egg",
        "garlic",
        "marshmallow",
        "mushroom",
    ],
    "PickPlaceCounterToOven": [
        "broccoli",
        "corn",
        "eggplant",
        "fish",
        "potato",
        "steak",
        "sweet_potato",
    ],
    "PickPlaceCounterToBlender": [
        "apple",
        "banana",
        "kiwi",
        "mango",
        "orange",
        "peach",
        "pear",
        "tangerine",
    ],
    "PickPlaceCounterToToasterOven": [
        "baguette",
        "bread",
        "hotdog_bun",
    ],
}

TASK_LABELS_ZH = {
    "PickPlaceCounterToCabinet": "台面物体放入橱柜",
    "PickPlaceCounterToDrawer": "台面物体放入抽屉",
    "PickPlaceCounterToSink": "台面物体放入水槽",
    "PickPlaceCounterToStove": "台面物体放到炉灶",
    "PickPlaceCounterToMicrowave": "台面物体放入微波炉",
    "PickPlaceCounterToOven": "台面物体放入烤箱",
    "PickPlaceCounterToBlender": "台面物体放入搅拌机",
    "PickPlaceCounterToToasterOven": "台面物体放入小烤箱",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_descriptions(task_metadata: Path) -> dict[str, list[str]]:
    rows = [str(value) for value in pd.read_parquet(task_metadata).index]
    result: dict[str, list[str]] = {}
    pending: list[str] = []
    for value in rows:
        if re.fullmatch(r"[A-Za-z0-9_]+", value):
            result[value] = pending
            pending = []
        else:
            pending.append(value)
    return result


def _manipulated_object_text(instruction: str) -> str | None:
    for pattern in (
        r"^Pick(?: up)? the (.+?) from\b",
        r"^Place the (.+?) (?:on|in)\b",
    ):
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    return None


def _find_instruction(descriptions: list[str], object_group: str) -> str | None:
    expected = object_group.replace("_", " ").lower()
    return next(
        (
            text
            for text in descriptions
            if _manipulated_object_text(text) == expected
        ),
        None,
    )


def build_registry(root: Path | None = None) -> dict[str, Any]:
    repo_root = (root or Path.cwd()).resolve()
    task_metadata = repo_root / TASK_METADATA
    if not task_metadata.is_file():
        raise FileNotFoundError(task_metadata)

    descriptions_by_task = _task_descriptions(task_metadata)
    relations: list[dict[str, Any]] = []
    for task_class, object_groups in SELECTED_TASK_OBJECTS.items():
        descriptions = descriptions_by_task.get(task_class, [])
        for object_group in object_groups:
            instruction = _find_instruction(descriptions, object_group)
            relations.append(
                {
                    "task_class": task_class,
                    "task_label_zh": TASK_LABELS_ZH[task_class],
                    "object_group": object_group,
                    "example_instruction": instruction,
                    "manipulated_object_text": (
                        _manipulated_object_text(instruction)
                        if instruction is not None
                        else None
                    ),
                    "task_class_found": task_class in descriptions_by_task,
                    "object_group_registered": object_group in OBJ_GROUPS,
                    "attested_by_task_language": instruction is not None,
                    "planned_episodes": 10,
                    "acceptance_successes_required": 8,
                }
            )

    objects = [row["object_group"] for row in relations]
    task_registry_path = Path(kitchen_objects.__file__).resolve()
    return {
        "schema_version": "1.1",
        "contract": {
            "midterm_task_count": 8,
            "midterm_object_count": 60,
            "episodes_per_object": 10,
            "object_pass_threshold": "successes >= 8 of 10",
            "scope_note": (
                "The selected tasks and manipulated objects are derived only "
                "from local RoboCasa365 VLA metadata and the RoboCasa "
                "simulator registry."
            ),
        },
        "source": {
            "benchmark": "RoboCasa365 pretrain atomic",
            "task_metadata": TASK_METADATA.as_posix(),
            "task_metadata_sha256": _sha256(task_metadata),
            "simulator": "RoboCasa",
            "object_registry_module": "robocasa.models.objects.kitchen_objects",
            "object_registry_file": str(task_registry_path),
            "object_registry_sha256": _sha256(task_registry_path),
        },
        "summary": {
            "task_count": len(SELECTED_TASK_OBJECTS),
            "object_count": len(objects),
            "unique_object_count": len(set(objects)),
            "relation_count": len(relations),
            "planned_episode_count": len(relations) * 10,
            "all_task_classes_found": all(
                row["task_class_found"] for row in relations
            ),
            "all_object_groups_registered": all(
                row["object_group_registered"] for row in relations
            ),
            "all_objects_attested_by_task_language": all(
                row["attested_by_task_language"] for row in relations
            ),
            "all_objects_are_manipulated_subjects": all(
                row["manipulated_object_text"]
                == row["object_group"].replace("_", " ")
                for row in relations
            ),
        },
        "tasks": [
            {
                "task_class": task_class,
                "task_label_zh": TASK_LABELS_ZH[task_class],
                "object_count": len(object_groups),
                "object_groups": object_groups,
            }
            for task_class, object_groups in SELECTED_TASK_OBJECTS.items()
        ],
        "relations": relations,
    }


def write_registry(registry: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vla_midterm_registry.json"
    csv_path = output_dir / "vla_midterm_relations.csv"
    runs_csv_path = output_dir / "vla_midterm_600_runs.csv"

    json_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    relation_fields = [
        "task_class",
        "task_label_zh",
        "object_group",
        "example_instruction",
        "manipulated_object_text",
        "planned_episodes",
        "acceptance_successes_required",
        "task_class_found",
        "object_group_registered",
        "attested_by_task_language",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=relation_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in relation_fields}
            for row in registry["relations"]
        )

    run_fields = [
        "run_id",
        "task_class",
        "task_label_zh",
        "object_group",
        "episode",
        "seed",
        "instruction",
        "status",
        "success",
        "steps",
        "result_json",
        "video",
        "failure_reason",
    ]
    with runs_csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_fields)
        writer.writeheader()
        run_number = 0
        for relation in registry["relations"]:
            for episode in range(1, 11):
                run_number += 1
                writer.writerow(
                    {
                        "run_id": f"VLA-MID-{run_number:04d}",
                        "task_class": relation["task_class"],
                        "task_label_zh": relation["task_label_zh"],
                        "object_group": relation["object_group"],
                        "episode": episode,
                        "seed": 202607300000 + run_number,
                        "instruction": relation["example_instruction"],
                        "status": "PENDING",
                        "success": "",
                        "steps": "",
                        "result_json": "",
                        "video": "",
                        "failure_reason": "",
                    }
                )

    return {
        "json": json_path,
        "csv": csv_path,
        "runs_csv": runs_csv_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/midterm_testing_vla"),
    )
    args = parser.parse_args()

    registry = build_registry()
    outputs = write_registry(registry, args.output)
    print(json.dumps(registry["summary"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {name: str(path.resolve()) for name, path in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(
        [
            registry["summary"]["task_count"] == 8,
            registry["summary"]["unique_object_count"] == 60,
            registry["summary"]["all_task_classes_found"],
            registry["summary"]["all_object_groups_registered"],
            registry["summary"]["all_objects_attested_by_task_language"],
            registry["summary"]["all_objects_are_manipulated_subjects"],
        ]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
