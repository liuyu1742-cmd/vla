"""Build one executable Skill IR from a mapped clip and raw keypoint track."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.skill_transfer.human_video_bridge import build_video_conditioned_skill_ir
from tools.skill_transfer.skill_ir import (
    DEFAULT_CONTRACT,
    DEFAULT_MANIFEST,
    contract_version_from_manifest,
    load_contract,
    write_skill_ir,
)


ROOT = Path(__file__).resolve().parents[1]
ROBOTPROJECT_PROCESSED = Path(
    r"C:\RobotProject\RobotProject\datasets\epic_kitchens\processed"
)


def select_clip_record(payload: Mapping[str, Any], clip_id: str) -> dict[str, Any]:
    records = payload.get("records")
    if isinstance(records, list):
        matches = [dict(record) for record in records if record.get("clip_id") == clip_id]
    elif payload.get("clip_id") == clip_id:
        matches = [dict(payload)]
    else:
        matches = []
    if len(matches) != 1:
        raise ValueError(f"clip_id must appear exactly once: {clip_id!r}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROBOTPROJECT_PROCESSED / "epic_video_skill_mapping.json",
    )
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping_payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    pose_payload = json.loads(args.pose.read_text(encoding="utf-8"))
    mapping = select_clip_record(mapping_payload, args.clip_id)
    pose = select_clip_record(pose_payload, args.clip_id)
    contract = load_contract(args.contract)
    skill_ir = build_video_conditioned_skill_ir(
        mapping,
        pose,
        contract,
        contract_version=contract_version_from_manifest(args.manifest),
    )
    write_skill_ir(args.output, skill_ir, contract)
    print(
        json.dumps(
            {
                "clip_id": args.clip_id,
                "relation_key": skill_ir["relation_key"],
                "canonical_actions": skill_ir["canonical_actions"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
