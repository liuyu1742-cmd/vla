"""Audit Task-2 human-video fragments and build a read-only Skill-IR intake manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.skill_transfer.skill_ir import DEFAULT_CONTRACT, load_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELATION_KEY = "organizing::storage_box"
DEFAULT_OBJECT_ROOT = Path(
    r"C:\RobotProject\RobotProject\datasets\midterm_15task_delivery"
    r"\organizing__整理任务\objects\storage_box"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "task2_human_video_intake"
    / "organizing_storage_box_manifest.json"
)
_INDEX_PATTERN = re.compile(r"_(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_action_fragment(
    actions: Sequence[object], object_id: str
) -> str | None:
    normalized = [str(action) for action in actions]
    if normalized == [f"locate({object_id})", f"grasp({object_id})"]:
        return "pick"
    if normalized == ["move(target)", f"place({object_id})"]:
        return "place"
    if normalized == [
        f"grasp({object_id})",
        f"insert({object_id},target)",
    ]:
        return "insert"
    return None


def _resolve_video_path(object_root: Path, record: Mapping[str, Any]) -> Path:
    source = str(record.get("source", "")).strip()
    media = record.get("media")
    if not source or not isinstance(media, Mapping):
        raise ValueError("record is missing source or media")
    clip_path = str(media.get("clip_path", "")).strip()
    if not clip_path:
        raise ValueError("record media is missing clip_path")
    name = Path(clip_path).name
    return (object_root / "videos" / source / f"{source}__{name}").resolve()


def _record_index(record_id: str) -> int:
    match = _INDEX_PATTERN.search(record_id)
    if match is None:
        raise ValueError(f"record id has no terminal action index: {record_id}")
    return int(match.group(1))


def _segment(
    *,
    annotation_path: Path,
    record: Mapping[str, Any],
    phase_role: str,
    video_path: Path,
) -> dict[str, Any]:
    media = record["media"]
    return {
        "record_id": str(record["id"]),
        "phase_role": phase_role,
        "actions": [str(item) for item in record["actions"]],
        "source_split": str(record.get("split", "")),
        "video_id": str(media.get("video_id", "")),
        "action_index": _record_index(str(record["id"])),
        "segment_start_seconds": float(media.get("segment_start_in_clip", 0.0)),
        "segment_stop_seconds": float(media.get("segment_stop_in_clip", 0.0)),
        "annotation_path": str(annotation_path.resolve()),
        "annotation_sha256": _sha256(annotation_path),
        "video_path": str(video_path),
        "video_sha256": _sha256(video_path),
    }


def build_intake_manifest(
    object_root: Path,
    contract: Mapping[str, Mapping[str, Any]],
    *,
    relation_key: str = DEFAULT_RELATION_KEY,
    max_action_index_gap: int = 5,
) -> dict[str, Any]:
    root = Path(object_root).resolve()
    if relation_key not in contract:
        raise ValueError(f"unknown Task-2 relation: {relation_key}")
    if max_action_index_gap < 1:
        raise ValueError("max_action_index_gap must be positive")
    expected = dict(contract[relation_key])
    task_id = str(expected.get("task_id", "")).strip()
    object_id = str(expected.get("object_id", "")).strip()
    if relation_key != f"{task_id}::{object_id}":
        raise ValueError("Task-2 relation key differs from task/object fields")
    annotations_root = root / "annotations"
    videos_root = root / "videos"
    annotation_paths = sorted(annotations_root.rglob("*.json"))
    exact: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    foreign_relations: Counter[str] = Counter()
    missing_media: list[str] = []
    phase_counts: Counter[str] = Counter()

    for annotation_path in annotation_paths:
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            record = payload.get("record")
            if not isinstance(record, Mapping):
                raise ValueError("annotation is missing record object")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parse_errors.append(
                {"path": str(annotation_path.resolve()), "error": str(error)}
            )
            continue
        current_task = str(record.get("task", "")).strip()
        current_object = str(record.get("object", "")).strip()
        if current_task != task_id or current_object != object_id:
            foreign_relations[f"{current_task}::{current_object}"] += 1
            continue
        record_id = str(record.get("id", "")).strip()
        media = record.get("media")
        actions = record.get("actions")
        if (
            not record_id
            or not isinstance(media, Mapping)
            or not isinstance(actions, list)
        ):
            parse_errors.append(
                {
                    "path": str(annotation_path.resolve()),
                    "error": "exact relation record is missing id, media, or actions",
                }
            )
            continue
        try:
            video_path = _resolve_video_path(root, record)
            index = _record_index(record_id)
        except ValueError as error:
            parse_errors.append(
                {"path": str(annotation_path.resolve()), "error": str(error)}
            )
            continue
        if not video_path.is_file():
            missing_media.append(str(video_path))
            continue
        phase = classify_action_fragment(actions, object_id)
        phase_counts[phase or "non_transfer"] += 1
        exact.append(
            {
                "annotation_path": annotation_path,
                "record": record,
                "record_id": record_id,
                "video_id": str(media.get("video_id", "")).strip(),
                "index": index,
                "phase": phase,
                "video_path": video_path,
            }
        )

    exact.sort(key=lambda item: (item["video_id"], item["index"], item["record_id"]))
    episodes: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    picks = [item for item in exact if item["phase"] == "pick"]
    for pick in picks:
        following = [
            item
            for item in exact
            if item["video_id"] == pick["video_id"]
            and pick["index"] < item["index"]
            and item["index"] - pick["index"] <= max_action_index_gap
        ]
        terminal = next(
            (
                item
                for item in following
                if item["phase"] in {"place", "insert"}
            ),
            None,
        )
        if terminal is None:
            blocked.append(
                {
                    "pick_record_id": pick["record_id"],
                    "reason": "no_place_or_insert_within_gap",
                }
            )
            continue
        intervening = [
            item
            for item in following
            if item["index"] < terminal["index"]
        ]
        if intervening:
            blocked.append(
                {
                    "pick_record_id": pick["record_id"],
                    "terminal_record_id": terminal["record_id"],
                    "reason": "intervening_non_transfer_fragment",
                    "intervening_record_ids": [
                        item["record_id"] for item in intervening
                    ],
                }
            )
            continue
        segments = [
            _segment(
                annotation_path=item["annotation_path"],
                record=item["record"],
                phase_role=str(item["phase"]),
                video_path=item["video_path"],
            )
            for item in (pick, terminal)
        ]
        episodes.append(
            {
                "episode_id": f"{relation_key.replace('::', '__')}__{pick['record_id']}",
                "relation_key": relation_key,
                "task_id": task_id,
                "object_id": object_id,
                "video_id": pick["video_id"],
                "phase_roles": [item["phase_role"] for item in segments],
                "canonical_actions": list(expected["canonical_actions"]),
                "segments": segments,
                "status": "ready_for_pose_extraction",
                "motion_retargeting_role": "reference_only",
            }
        )

    return {
        "schema_version": "task2_human_video_intake_v1",
        "relation_key": relation_key,
        "task_id": task_id,
        "object_id": object_id,
        "instruction": expected.get("instruction"),
        "canonical_actions": list(expected["canonical_actions"]),
        "source_root": str(root),
        "source_read_only": True,
        "selection_policy": {
            "exact_task_and_object_required": True,
            "pick_fragment": [
                f"locate({object_id})",
                f"grasp({object_id})",
            ],
            "terminal_fragments": [
                ["move(target)", f"place({object_id})"],
                [f"grasp({object_id})", f"insert({object_id},target)"],
            ],
            "max_action_index_gap": max_action_index_gap,
            "intervening_exact_relation_fragments_allowed": False,
        },
        "audit": {
            "annotations_scanned": len(annotation_paths),
            "videos_scanned": len(list(videos_root.rglob("*.mp4"))),
            "exact_relation_annotations": len(exact),
            "foreign_relation_annotations": sum(foreign_relations.values()),
            "foreign_relation_counts": dict(sorted(foreign_relations.items())),
            "phase_counts": dict(sorted(phase_counts.items())),
            "parse_error_count": len(parse_errors),
            "missing_media_count": len(missing_media),
            "ready_episode_count": len(episodes),
            "blocked_episode_count": len(blocked),
        },
        "episodes": episodes,
        "blocked_episodes": blocked,
        "parse_errors": parse_errors,
        "missing_media": sorted(set(missing_media)),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--relation-key", default=DEFAULT_RELATION_KEY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--max-action-index-gap", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_intake_manifest(
        args.object_root,
        load_contract(args.contract),
        relation_key=args.relation_key,
        max_action_index_gap=args.max_action_index_gap,
    )
    manifest["contract_path"] = str(args.contract.resolve())
    manifest["contract_sha256"] = _sha256(args.contract)
    _write_json_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "relation_key": manifest["relation_key"],
                **manifest["audit"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if manifest["audit"]["parse_error_count"]:
        return 2
    if manifest["audit"]["missing_media_count"]:
        return 3
    if not manifest["episodes"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
