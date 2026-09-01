from pathlib import Path

import pytest

from tools.vla_metadata_survey.core import (
    EvidenceLevel,
    extract_bridge_summary,
    extract_robocasa_object_registry,
    extract_robocasa_tasks,
    rank_objects,
    select_task_families,
    validate_task_record,
)


def test_task_record_requires_provenance():
    with pytest.raises(ValueError, match="source"):
        validate_task_record(
            {
                "dataset_id": "fixture",
                "task_id": "wipe_table",
                "task_name": "Wipe Table",
            }
        )


def test_extracts_task_language_objects_and_activity(tmp_path: Path):
    task_file = tmp_path / "composite" / "sanitizing_surface" / "wipe_table.py"
    task_file.parent.mkdir(parents=True)
    task_file.write_text(
        """
class WipeTable(Kitchen):
    '''Wipe Table.

    Wipe a counter with a sponge.
    '''
    def get_ep_meta(self):
        ep_meta = {}
        ep_meta["lang"] = "Pick up the sponge and wipe the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(name="sponge", obj_groups="sponge", graspable=True))
        return cfgs
""",
        encoding="utf-8",
    )

    records = extract_robocasa_tasks(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record["task_family_native"] == "sanitizing_surface"
    assert record["instructions"] == [
        "Pick up the sponge and wipe the counter."
    ]
    assert record["manipulated_objects"] == ["sponge"]
    assert record["evidence_level"] == EvidenceLevel.LOCAL_TASK_CODE.value


def test_extracts_registered_objects_and_properties(tmp_path: Path):
    registry = tmp_path / "kitchen_objects.py"
    registry.write_text(
        """
OBJ_CATEGORIES = dict(
    apple=dict(types=("fruit"), graspable=True, washable=True),
    coffee_mug=dict(types=("receptacle",), graspable=True, washable=True),
)
""",
        encoding="utf-8",
    )

    objects = extract_robocasa_object_registry(registry)

    assert [item["canonical_name"] for item in objects] == ["apple", "coffee_mug"]
    assert objects[0]["properties"]["graspable"] is True
    assert objects[0]["properties"]["types"] == ["fruit"]


def test_top_families_are_native_and_ranked_by_observed_task_count():
    tasks = [
        {"task_family_native": "washing", "task_id": "a"},
        {"task_family_native": "washing", "task_id": "b"},
        {"task_family_native": "organizing", "task_id": "c"},
    ]

    families = select_task_families(tasks, limit=2)

    assert families == [
        {"rank": 1, "task_family": "washing", "task_count": 2, "task_ids": ["a", "b"]},
        {
            "rank": 2,
            "task_family": "organizing",
            "task_count": 1,
            "task_ids": ["c"],
        },
    ]


def test_objects_used_by_more_tasks_rank_before_registry_only_objects():
    registry = [
        {"canonical_name": "apple", "properties": {}},
        {"canonical_name": "banana", "properties": {}},
    ]
    tasks = [
        {"task_id": "t1", "manipulated_objects": ["banana"]},
        {"task_id": "t2", "manipulated_objects": ["banana"]},
    ]

    ranked = rank_objects(registry, tasks, limit=2)

    assert [row["canonical_name"] for row in ranked] == ["banana", "apple"]
    assert ranked[0]["task_count"] == 2
    assert ranked[1]["task_count"] == 0


def test_bridge_summary_counts_examples_without_loading_trajectories(tmp_path: Path):
    (tmp_path / "dataset_info.json").write_text(
        """
{
  "name": "bridge_dataset",
  "version": "1.0.0",
  "splits": [
    {"name": "train", "shardLengths": ["2", "3"], "numBytes": "10"},
    {"name": "val", "shardLengths": ["1"], "numBytes": "4"}
  ]
}
""",
        encoding="utf-8",
    )
    (tmp_path / "features.json").write_text(
        '{"pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict"}',
        encoding="utf-8",
    )

    summary = extract_bridge_summary(tmp_path)

    assert summary["dataset_id"] == "bridge_dataset"
    assert summary["total_episodes"] == 6
    assert summary["splits"]["train"]["episodes"] == 5
