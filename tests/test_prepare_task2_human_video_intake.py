from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_task2_human_video_intake import (
    build_intake_manifest,
    classify_action_fragment,
)


def _contract() -> dict[str, dict]:
    return {
        "organizing::storage_box": {
            "relation_key": "organizing::storage_box",
            "task_id": "organizing",
            "task_name": "整理任务",
            "object_id": "storage_box",
            "object_name": "收纳箱",
            "instruction": "整理并归位收纳箱",
            "target": "none",
            "canonical_actions": [
                "locate(storage_box)",
                "grasp(storage_box)",
                "move(storage)",
                "place(storage_box)",
            ],
        }
    }


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_record(
    root: Path,
    *,
    record_id: str,
    task: str,
    obj: str,
    actions: list[str],
) -> None:
    source = "epic_kitchens_100"
    clip_name = record_id.removeprefix("epic_") + ".mp4"
    video = root / "videos" / source / f"{source}__{clip_name}"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(f"video:{record_id}".encode("utf-8"))
    annotation = root / "annotations" / "action_records" / source / f"{record_id}.json"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_type": "unified_action_dataset",
        "record": {
            "id": record_id,
            "split": "train",
            "task": task,
            "scene": "_".join(record_id.split("_")[1:3]),
            "object": obj,
            "target": "none",
            "actions": actions,
            "verb": actions[-1].split("(", 1)[0],
            "instruction": "fixture",
            "source": source,
            "modality": "video_segment",
            "media": {
                "video_id": "_".join(record_id.split("_")[1:3]),
                "clip_path": f"datasets/clips/{clip_name}",
                "segment_start_in_clip": 0.0,
                "segment_stop_in_clip": 1.0,
            },
        },
    }
    annotation.write_text(json.dumps(payload), encoding="utf-8")


class Task2HumanVideoIntakeTests(unittest.TestCase):
    def test_classifies_only_pick_and_place_family_fragments(self) -> None:
        self.assertEqual(
            "pick",
            classify_action_fragment(
                ["locate(storage_box)", "grasp(storage_box)"], "storage_box"
            ),
        )
        self.assertEqual(
            "place",
            classify_action_fragment(
                ["move(target)", "place(storage_box)"], "storage_box"
            ),
        )
        self.assertEqual(
            "insert",
            classify_action_fragment(
                ["grasp(storage_box)", "insert(storage_box,target)"], "storage_box"
            ),
        )
        self.assertIsNone(
            classify_action_fragment(
                ["locate(storage_box)", "wash(storage_box)"], "storage_box"
            )
        )

    def test_builds_read_only_manifest_and_blocks_interrupted_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "storage_box"
            _write_record(
                root,
                record_id="epic_P01_01_10",
                task="organizing",
                obj="storage_box",
                actions=["locate(storage_box)", "grasp(storage_box)"],
            )
            _write_record(
                root,
                record_id="epic_P01_01_12",
                task="organizing",
                obj="storage_box",
                actions=["move(target)", "place(storage_box)"],
            )
            _write_record(
                root,
                record_id="epic_P01_01_20",
                task="organizing",
                obj="storage_box",
                actions=["locate(storage_box)", "grasp(storage_box)"],
            )
            _write_record(
                root,
                record_id="epic_P01_01_21",
                task="organizing",
                obj="storage_box",
                actions=["locate(storage_box)", "inspect(storage_box)"],
            )
            _write_record(
                root,
                record_id="epic_P01_01_22",
                task="organizing",
                obj="storage_box",
                actions=["move(target)", "place(storage_box)"],
            )
            _write_record(
                root,
                record_id="epic_P01_01_30",
                task="object_fetching",
                obj="container",
                actions=["locate(container)", "grasp(container)"],
            )
            before = _tree_hash(root)

            manifest = build_intake_manifest(
                root,
                _contract(),
                relation_key="organizing::storage_box",
                max_action_index_gap=5,
            )

            self.assertEqual(before, _tree_hash(root))
            self.assertTrue(manifest["source_read_only"])
            self.assertEqual(6, manifest["audit"]["annotations_scanned"])
            self.assertEqual(5, manifest["audit"]["exact_relation_annotations"])
            self.assertEqual(1, manifest["audit"]["ready_episode_count"])
            self.assertEqual(1, manifest["audit"]["blocked_episode_count"])
            episode = manifest["episodes"][0]
            self.assertEqual(
                ["epic_P01_01_10", "epic_P01_01_12"],
                [item["record_id"] for item in episode["segments"]],
            )
            self.assertEqual(["pick", "place"], episode["phase_roles"])
            self.assertEqual(
                _contract()["organizing::storage_box"]["canonical_actions"],
                episode["canonical_actions"],
            )
            for segment in episode["segments"]:
                self.assertTrue(Path(segment["annotation_path"]).is_absolute())
                self.assertTrue(Path(segment["video_path"]).is_absolute())
                self.assertEqual(64, len(segment["annotation_sha256"]))
                self.assertEqual(64, len(segment["video_sha256"]))
            self.assertEqual(
                "intervening_non_transfer_fragment",
                manifest["blocked_episodes"][0]["reason"],
            )

    def test_rejects_relation_that_does_not_match_contract_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown Task-2 relation"):
                build_intake_manifest(
                    Path(directory),
                    _contract(),
                    relation_key="organizing::toy",
                )


if __name__ == "__main__":
    unittest.main()
