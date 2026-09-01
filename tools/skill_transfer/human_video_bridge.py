"""Bridge Task-2 human-video evidence into an auditable robot Skill IR."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .skill_ir import (
    DEFAULT_CONTRACT,
    DEFAULT_MANIFEST,
    build_skill_ir,
    contract_version_from_manifest,
    load_contract,
    validate_skill_ir,
)


ROOT = Path(__file__).resolve().parents[2]
ROBOTPROJECT_PROCESSED = Path(
    r"C:\RobotProject\RobotProject\datasets\epic_kitchens\processed"
)
DEFAULT_MAPPING = ROBOTPROJECT_PROCESSED / "epic_video_skill_mapping.json"
DEFAULT_POSE = ROBOTPROJECT_PROCESSED / "epic_pose_evidence.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "human_video_skill_bridge_readiness.json"


def resolve_task2_relation(
    mapping: Mapping[str, Any],
    contract: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if mapping.get("mapping_status") != "mapped":
        raise ValueError("video record is not mapped")
    task = str(mapping.get("task_id", "")).strip()
    obj = str(mapping.get("object", "")).strip()
    if not task or not obj:
        raise ValueError("mapped video record is missing task_id or object")
    key = f"{task}::{obj}"
    if key not in contract:
        raise ValueError(f"mapped video relation is not in Task-2 contract: {key}")
    expected = dict(contract[key])
    actions = mapping.get("actions")
    if not isinstance(actions, list) or actions != expected["canonical_actions"]:
        raise ValueError(f"mapped video actions differ from Task-2 relation {key}")
    return expected


def _pose_reasons(mapping: Mapping[str, Any], pose: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if pose.get("clip_id") != mapping.get("clip_id"):
        reasons.append("clip_id_mismatch")
    if pose.get("pose_status") != "available":
        reasons.append("pose_not_available")
    keypoints_raw = pose.get("keypoints")
    if not isinstance(keypoints_raw, list) or not keypoints_raw:
        reasons.append("missing_raw_keypoint_track")
        return reasons
    try:
        keypoints = np.asarray(keypoints_raw, dtype=np.float64)
    except (TypeError, ValueError):
        reasons.append("invalid_keypoint_track")
        return reasons
    if keypoints.ndim != 3 or keypoints.shape[0] < 2 or keypoints.shape[2] != 3:
        reasons.append("invalid_keypoint_track")
        return reasons
    if not np.isfinite(keypoints).all():
        reasons.append("non_finite_keypoint_track")
    if np.allclose(keypoints[..., :2], 0.0):
        reasons.append("all_zero_keypoint_track")
    names = pose.get("keypoint_names")
    if not isinstance(names, list) or len(names) != keypoints.shape[1]:
        reasons.append("missing_keypoint_names")
    frame_times = pose.get("frame_times")
    if not isinstance(frame_times, list) or len(frame_times) != keypoints.shape[0]:
        reasons.append("missing_frame_times")
    else:
        try:
            times = np.asarray(frame_times, dtype=np.float64)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                reasons.append("non_monotonic_frame_times")
        except (TypeError, ValueError):
            reasons.append("invalid_frame_times")
    return reasons


def assess_robotproject_record(
    mapping: Mapping[str, Any],
    pose: Mapping[str, Any],
    contract: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    relation_key: str | None = None
    try:
        relation = resolve_task2_relation(mapping, contract)
        relation_key = str(relation["relation_key"])
    except ValueError as error:
        reasons.append(f"relation_error:{error}")
    reasons.extend(_pose_reasons(mapping, pose))
    return {
        "clip_id": mapping.get("clip_id"),
        "relation_key": relation_key,
        "relation_ready": relation_key is not None,
        "pose_summary_available": pose.get("pose_status") == "available",
        "motion_capture_ready": not reasons,
        "reasons": reasons,
    }


def build_video_conditioned_skill_ir(
    mapping: Mapping[str, Any],
    pose: Mapping[str, Any],
    contract: Mapping[str, Mapping[str, Any]],
    *,
    contract_version: str,
) -> dict[str, Any]:
    readiness = assess_robotproject_record(mapping, pose, contract)
    if not readiness["motion_capture_ready"]:
        raise ValueError(
            "human video record is not ready for Skill IR transfer: "
            + ", ".join(readiness["reasons"])
        )
    relation = resolve_task2_relation(mapping, contract)
    key = str(relation["relation_key"])
    skill_ir = build_skill_ir(key, contract, contract_version=contract_version)
    skill_ir["source_video"] = {
        "format": "robotproject_epic_clip_v1",
        "clip_id": mapping["clip_id"],
        "clip_path": mapping.get("clip_path"),
        "clip_start_seconds": mapping.get("clip_start_seconds"),
        "clip_stop_seconds": mapping.get("clip_stop_seconds"),
        "epic_narration": mapping.get("epic_narration"),
    }
    skill_ir["perception_evidence"] = {
        "format": "human_keypoint_track_v1",
        "keypoint_names": list(pose["keypoint_names"]),
        "frame_times": list(pose["frame_times"]),
        "keypoints": pose["keypoints"],
        "mean_keypoint_confidence": float(pose["mean_keypoint_confidence"]),
        "detected_person_frames": int(pose["detected_person_frames"]),
    }
    skill_ir["parser_evidence"] = {
        "format": "robotproject_video_skill_mapping_v1",
        "mapping_status": mapping["mapping_status"],
        "relation_key": key,
        "predicted_actions": list(mapping["actions"]),
        "actions_match_task2_contract": True,
    }
    skill_ir["execution_request"] = {
        "format": "openvla_formal_skill_request_v1",
        "relation_key": key,
        "canonical_actions": list(relation["canonical_actions"]),
        "status": "ready_for_robot_policy",
        "motion_retargeting_role": "reference_only",
    }
    return validate_skill_ir(skill_ir, contract)


def audit_robotproject_video_bridge(
    mapping_payload: Mapping[str, Any],
    pose_payload: Mapping[str, Any],
    contract: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    mappings = mapping_payload.get("records")
    poses = pose_payload.get("records")
    if not isinstance(mappings, list) or not isinstance(poses, list):
        raise ValueError("RobotProject mapping and pose payloads must contain record lists")
    pose_by_clip = {str(item.get("clip_id")): item for item in poses}
    assessments: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()
    organizing_toy_candidates = 0
    for mapping in mappings:
        if mapping.get("mapping_status") != "mapped":
            continue
        clip_id = str(mapping.get("clip_id"))
        pose = pose_by_clip.get(clip_id, {"clip_id": clip_id, "pose_status": "missing"})
        assessment = assess_robotproject_record(mapping, pose, contract)
        assessments.append(assessment)
        if assessment["relation_key"] == "organizing::toy":
            organizing_toy_candidates += 1
        failure_reasons.update(assessment["reasons"])
    return {
        "schema_version": "human_video_skill_bridge_readiness_v1",
        "source_mapping_format": mapping_payload.get("format"),
        "source_pose_format": pose_payload.get("format"),
        "mapped_records_checked": len(assessments),
        "task2_relation_exact": sum(item["relation_ready"] for item in assessments),
        "pose_summary_available": sum(
            item["pose_summary_available"] for item in assessments
        ),
        "raw_motion_capture_ready": sum(
            item["motion_capture_ready"] for item in assessments
        ),
        "organizing_toy_candidates": organizing_toy_candidates,
        "failure_reasons": dict(failure_reasons),
        "ready_records": [
            item for item in assessments if item["motion_capture_ready"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--pose", type=Path, default=DEFAULT_POSE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = audit_robotproject_video_bridge(
        json.loads(args.mapping.read_text(encoding="utf-8")),
        json.loads(args.pose.read_text(encoding="utf-8")),
        contract,
    )
    report.update(
        {
            "mapping_path": str(args.mapping.resolve()),
            "pose_path": str(args.pose.resolve()),
            "contract_path": str(args.contract.resolve()),
            "contract_version": contract_version_from_manifest(args.manifest),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in (
        "mapped_records_checked", "task2_relation_exact", "pose_summary_available",
        "raw_motion_capture_ready", "organizing_toy_candidates", "failure_reasons"
    )}, ensure_ascii=False, indent=2))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
