"""Build auditable Task-2 human-video L2 Skill IR records.

L2 means that a human demonstration was parsed into the canonical skill
sequence with visual evidence.  It does not mean that a robot executed the
skill, so this builder always leaves ``execution_evidence`` null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.skill_transfer.skill_ir import (
    DEFAULT_CONTRACT,
    DEFAULT_MANIFEST,
    build_skill_ir,
    contract_version_from_manifest,
    load_contract,
    validate_skill_ir,
    write_skill_ir,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE = (
    ROOT
    / "outputs"
    / "task2_human_video_intake"
    / "organizing_storage_box_manifest.json"
)
DEFAULT_POSE = (
    ROOT / "outputs" / "task2_human_video_intake" / "keypoints" / "pose_manifest.json"
)
DEFAULT_HANDS = (
    ROOT / "outputs" / "task2_human_video_intake" / "hands" / "hand_manifest.json"
)
DEFAULT_CONTACT = (
    ROOT / "outputs" / "task2_human_video_intake" / "contact" / "contact_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "task2_human_video_intake" / "skill_ir"
)
TERMINAL_POLICY = "object_track_plus_exact_task2_place_or_insert_annotation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return value


def _verify_reference(path_value: object, sha_value: object, *, label: str) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is missing")
    expected = _require_sha256(sha_value, label=f"{label} sha256")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected}, found {actual}"
        )


def _index_records(
    records: Sequence[Mapping[str, object]], *, label: str
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} record_id is missing")
        if record_id in indexed:
            raise ValueError(f"duplicate {label} record_id: {record_id}")
        indexed[record_id] = dict(record)
    return indexed


def _evidence_link(
    record: Mapping[str, object], *, path_field: str, sha_field: str
) -> dict[str, object]:
    return {
        "status": record.get("pose_status", record.get("hand_status")),
        "track_path": record[path_field],
        "track_sha256": record[sha_field],
    }


def build_episode_skill_ir(
    episode: Mapping[str, object],
    *,
    pose_records: Mapping[str, Mapping[str, object]],
    hand_records: Mapping[str, Mapping[str, object]],
    contact_records: Mapping[str, Mapping[str, object]],
    contract: Mapping[str, Mapping[str, object]],
    contract_version: str,
    intake_path: str,
    intake_sha256: str,
) -> dict[str, Any]:
    """Build one L2 record after enforcing the human-video evidence gates."""

    relation_key = episode.get("relation_key")
    if not isinstance(relation_key, str) or relation_key not in contract:
        raise ValueError(f"unknown relation: {relation_key}")
    expected_actions = list(contract[relation_key]["canonical_actions"])
    if episode.get("canonical_actions") != expected_actions:
        raise ValueError("episode canonical_actions differ from the Task-2 contract")

    raw_segments = episode.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) < 2:
        raise ValueError("episode requires pick and terminal segments")
    segments = [dict(segment) for segment in raw_segments if isinstance(segment, Mapping)]
    if len(segments) != len(raw_segments):
        raise ValueError("episode segments must be objects")

    pick_segments = [segment for segment in segments if segment.get("phase_role") == "pick"]
    terminal_segments = [
        segment
        for segment in segments
        if segment.get("phase_role") in {"place", "insert"}
    ]
    if len(pick_segments) != 1 or len(terminal_segments) != 1:
        raise ValueError("episode requires exactly one pick and one place/insert segment")

    pick_id = str(pick_segments[0].get("record_id"))
    terminal_id = str(terminal_segments[0].get("record_id"))
    for record_id in (pick_id, terminal_id):
        if record_id not in pose_records:
            raise ValueError(f"missing pose evidence for {record_id}")
        if record_id not in hand_records:
            raise ValueError(f"missing hand evidence for {record_id}")
        if record_id not in contact_records:
            raise ValueError(f"missing contact evidence for {record_id}")
        if pose_records[record_id].get("pose_status") != "available":
            raise ValueError(f"pose evidence is not available for {record_id}")

    pick_hand = hand_records[pick_id]
    pick_contact = contact_records[pick_id]
    if pick_hand.get("hand_status") != "available":
        raise ValueError(f"pick hand evidence is not available for {pick_id}")
    if (
        pick_contact.get("interaction_evidence_ready") is not True
        or pick_contact.get("segment_evidence_ready") is not True
    ):
        raise ValueError(f"pick interaction evidence is not ready for {pick_id}")

    terminal_contact = contact_records[terminal_id]
    if (
        terminal_contact.get("object_track_ready") is not True
        or terminal_contact.get("segment_evidence_ready") is not True
    ):
        raise ValueError(f"terminal object track is not ready for {terminal_id}")

    terminal_hand_verified = (
        hand_records[terminal_id].get("hand_status") == "available"
        and terminal_contact.get("interaction_evidence_ready") is True
    )
    skill_ir = build_skill_ir(
        relation_key,
        contract,
        contract_version=contract_version,
    )
    skill_ir["source_video"] = {
        "evidence_level": "L2",
        "episode_id": episode.get("episode_id"),
        "video_id": episode.get("video_id"),
        "task2_source_read_only": True,
        "intake_manifest_path": intake_path,
        "intake_manifest_sha256": _require_sha256(
            intake_sha256, label="intake manifest sha256"
        ),
        "segments": [
            {
                "record_id": segment["record_id"],
                "phase_role": segment["phase_role"],
                "actions": list(segment["actions"]),
                "video_path": segment["video_path"],
                "video_sha256": segment["video_sha256"],
                "annotation_path": segment["annotation_path"],
                "annotation_sha256": segment["annotation_sha256"],
            }
            for segment in segments
        ],
    }
    skill_ir["perception_evidence"] = {
        "method": "first_person_pose_true_hand_keypoints_and_open_vocab_object_tracking",
        "pick_contact_visually_verified": True,
        "full_terminal_hand_contact_verified": terminal_hand_verified,
        "terminal_evidence_policy": (
            "true_hand_object_contact"
            if terminal_hand_verified
            else TERMINAL_POLICY
        ),
        "pose_tracks": {
            record_id: _evidence_link(
                pose_records[record_id],
                path_field="track_path",
                sha_field="track_sha256",
            )
            for record_id in (pick_id, terminal_id)
        },
        "hand_tracks": {
            record_id: _evidence_link(
                hand_records[record_id],
                path_field="track_path",
                sha_field="track_sha256",
            )
            for record_id in (pick_id, terminal_id)
        },
        "contact_tracks": {
            record_id: {
                "interaction_evidence_ready": contact_records[record_id].get(
                    "interaction_evidence_ready"
                ),
                "object_track_ready": contact_records[record_id].get(
                    "object_track_ready"
                ),
                "track_path": contact_records[record_id]["contact_path"],
                "track_sha256": contact_records[record_id]["contact_sha256"],
            }
            for record_id in (pick_id, terminal_id)
        },
    }
    skill_ir["parser_evidence"] = {
        "canonical_action_sequence_verified": True,
        "phase_roles": list(episode.get("phase_roles", [])),
        "annotation_source": "Task2 exact task-object action records",
        "annotation_hashes": {
            str(segment["record_id"]): segment["annotation_sha256"]
            for segment in segments
        },
    }
    skill_ir["execution_request"] = {
        "status": "ready_for_simulator_mapping",
        "target_platform": "RoboCasa",
        "motion_retargeting_role": "reference_only",
        "policy_learning_role": "human_demonstration_reference",
        "required_next_evidence_level": "L3",
    }
    skill_ir["execution_evidence"] = None
    return validate_skill_ir(skill_ir, contract)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return data


def _verify_source_and_tracks(
    episodes: Sequence[Mapping[str, object]],
    pose_records: Mapping[str, Mapping[str, object]],
    hand_records: Mapping[str, Mapping[str, object]],
    contact_records: Mapping[str, Mapping[str, object]],
) -> None:
    for episode in episodes:
        for raw_segment in episode.get("segments", []):
            if not isinstance(raw_segment, Mapping):
                raise ValueError("episode segment must be an object")
            record_id = str(raw_segment.get("record_id"))
            _verify_reference(
                raw_segment.get("video_path"),
                raw_segment.get("video_sha256"),
                label=f"{record_id} source video",
            )
            _verify_reference(
                raw_segment.get("annotation_path"),
                raw_segment.get("annotation_sha256"),
                label=f"{record_id} annotation",
            )
            _verify_reference(
                pose_records[record_id].get("track_path"),
                pose_records[record_id].get("track_sha256"),
                label=f"{record_id} pose track",
            )
            _verify_reference(
                hand_records[record_id].get("track_path"),
                hand_records[record_id].get("track_sha256"),
                label=f"{record_id} hand track",
            )
            _verify_reference(
                contact_records[record_id].get("contact_path"),
                contact_records[record_id].get("contact_sha256"),
                label=f"{record_id} contact track",
            )


def build_batch(
    *,
    intake_path: Path,
    pose_path: Path,
    hands_path: Path,
    contact_path: Path,
    contract_path: Path,
    contract_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    intake = _load_json(intake_path)
    pose = _load_json(pose_path)
    hands = _load_json(hands_path)
    contact = _load_json(contact_path)
    contract = load_contract(contract_path)
    contract_version = contract_version_from_manifest(contract_manifest_path)

    relation_keys = {
        intake.get("relation_key"),
        pose.get("relation_key"),
        hands.get("relation_key"),
        contact.get("relation_key"),
    }
    if relation_keys != {"organizing::storage_box"}:
        raise ValueError(f"manifest relation mismatch: {sorted(map(str, relation_keys))}")

    intake_sha = sha256_file(intake_path)
    if pose.get("intake_sha256") != intake_sha or hands.get("intake_sha256") != intake_sha:
        raise ValueError("pose/hand intake manifest hash mismatch")
    hands_sha = sha256_file(hands_path)
    if contact.get("hand_manifest_sha256") != hands_sha:
        raise ValueError("contact hand manifest hash mismatch")
    if contact.get("all_segments_ready") is not True:
        raise ValueError("contact batch is not fully ready")

    episodes = intake.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("intake manifest has no ready episodes")
    pose_records = _index_records(pose.get("records", []), label="pose")
    hand_records = _index_records(hands.get("records", []), label="hand")
    contact_records = _index_records(contact.get("records", []), label="contact")
    _verify_source_and_tracks(
        episodes, pose_records, hand_records, contact_records
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, object]] = []
    for episode in episodes:
        skill_ir = build_episode_skill_ir(
            episode,
            pose_records=pose_records,
            hand_records=hand_records,
            contact_records=contact_records,
            contract=contract,
            contract_version=contract_version,
            intake_path=str(Path(intake_path).resolve()),
            intake_sha256=intake_sha,
        )
        episode_id = str(episode["episode_id"])
        output_path = output_dir / f"{episode_id}.json"
        write_skill_ir(output_path, skill_ir, contract)
        output_records.append(
            {
                "episode_id": episode_id,
                "path": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
                "evidence_level": "L2",
                "ready_for_simulator_mapping": True,
                "pure_autonomous_vla": False,
            }
        )

    manifest = {
        "schema_version": "task2_video_skill_ir_batch_v1",
        "relation_key": "organizing::storage_box",
        "evidence_level": "L2",
        "record_count": len(output_records),
        "all_records_ready_for_simulator_mapping": True,
        "execution_success_claimed": False,
        "source_manifests": {
            "intake": {
                "path": str(Path(intake_path).resolve()),
                "sha256": intake_sha,
            },
            "pose": {
                "path": str(Path(pose_path).resolve()),
                "sha256": sha256_file(pose_path),
            },
            "hands": {
                "path": str(Path(hands_path).resolve()),
                "sha256": hands_sha,
            },
            "contact": {
                "path": str(Path(contact_path).resolve()),
                "sha256": sha256_file(contact_path),
            },
        },
        "records": output_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, default=DEFAULT_INTAKE)
    parser.add_argument("--pose", type=Path, default=DEFAULT_POSE)
    parser.add_argument("--hands", type=Path, default=DEFAULT_HANDS)
    parser.add_argument("--contact", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-manifest", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_batch(
        intake_path=args.intake,
        pose_path=args.pose,
        hands_path=args.hands,
        contact_path=args.contact,
        contract_path=args.contract,
        contract_manifest_path=args.contract_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "relation_key": manifest["relation_key"],
                "evidence_level": manifest["evidence_level"],
                "record_count": manifest["record_count"],
                "all_records_ready_for_simulator_mapping": manifest[
                    "all_records_ready_for_simulator_mapping"
                ],
                "manifest": str((args.output_dir / "manifest.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
