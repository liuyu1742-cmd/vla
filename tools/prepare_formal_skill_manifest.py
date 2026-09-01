"""Build an auditable, seed-disjoint manifest for formal Skill IR training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.skill_transfer.evidence import validate_formal_skill_evidence
from tools.skill_transfer.skill_ir import DEFAULT_CONTRACT, load_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "datasets" / "formal_skills" / "organizing_toy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_manifest(
    records: Sequence[Mapping[str, object]],
    *,
    heldout_seeds: Sequence[int],
    min_training: int,
) -> dict[str, object]:
    if min_training < 1:
        raise ValueError("min_training must be positive")
    heldout = [int(seed) for seed in heldout_seeds]
    if not heldout or len(set(heldout)) != len(heldout):
        raise ValueError("held-out seeds must be a unique non-empty list")

    successful: list[dict[str, Any]] = []
    known_seeds: set[int] = set()
    for raw in records:
        if raw.get("success") is not True:
            continue
        if raw.get("relation_key") != "organizing::toy":
            raise ValueError("formal manifest only accepts organizing::toy")
        if raw.get("counts_toward_task2_coverage") is not True:
            raise ValueError("successful episode must count toward Task-2 coverage")
        seed = int(raw["seed"])
        if seed in known_seeds:
            raise ValueError(f"duplicate seed in formal episodes: {seed}")
        known_seeds.add(seed)
        successful.append(dict(raw))

    successful.sort(key=lambda item: int(item["seed"]))
    heldout_set = set(heldout)
    train = [item for item in successful if int(item["seed"]) not in heldout_set]
    held_out = [item for item in successful if int(item["seed"]) in heldout_set]
    found_heldout = {int(item["seed"]) for item in held_out}
    missing = sorted(heldout_set - found_heldout)
    if missing:
        raise RuntimeError(f"held-out successful seeds are missing: {missing}")
    if len(train) < min_training:
        raise RuntimeError(
            f"need at least {min_training} successful training episodes, found {len(train)}"
        )
    if {int(item["seed"]) for item in train} & found_heldout:
        raise RuntimeError("training and held-out seed sets overlap")

    skill_hashes = {str(item.get("skill_ir_sha256")) for item in successful}
    if len(skill_hashes) != 1 or "None" in skill_hashes:
        raise ValueError("all formal episodes must use one explicit Skill IR hash")
    return {
        "schema_version": "formal_skill_training_manifest_v1",
        "relation_key": "organizing::toy",
        "skill_ir_sha256": next(iter(skill_hashes)),
        "heldout_seeds": heldout,
        "minimum_training_episodes": min_training,
        "training_episode_count": len(train),
        "heldout_episode_count": len(held_out),
        "training_sample_count": sum(int(item["samples"]) for item in train),
        "heldout_sample_count": sum(int(item["samples"]) for item in held_out),
        "train": train,
        "held_out": held_out,
    }


def discover_formal_episodes(
    root: Path,
    contract: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for episode_dir in sorted(Path(root).glob("seed_[0-9][0-9][0-9]")):
        if not episode_dir.is_dir():
            continue
        report_path = episode_dir / "report.json"
        episode_path = episode_dir / "episode.npz"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("success") is not True:
            continue
        validated = validate_formal_skill_evidence(report, contract)
        if not episode_path.is_file():
            raise FileNotFoundError(f"successful episode data is missing: {episode_path}")
        with np.load(episode_path, allow_pickle=False) as episode:
            frames_shape = episode["frames"].shape
            actions_shape = episode["actions"].shape
            relation = str(episode["relation_key"].item())
            phase_count = int(episode["phases"].shape[0])
            canonical_count = int(episode["canonical_phases"].shape[0])
        if len(frames_shape) != 4 or frames_shape[-1] != 3:
            raise ValueError(f"invalid frame shape in {episode_path}: {frames_shape}")
        samples = int(frames_shape[0])
        if actions_shape != (samples, 7):
            raise ValueError(f"invalid action shape in {episode_path}: {actions_shape}")
        if phase_count != samples or canonical_count != samples:
            raise ValueError(f"phase arrays do not align in {episode_path}")
        if relation != validated["relation_key"]:
            raise ValueError(f"episode/report relation mismatch in {episode_path}")
        if int(validated.get("samples", samples)) != samples:
            raise ValueError(f"episode/report sample count mismatch in {episode_path}")
        records.append(
            {
                "seed": int(validated["seed"]),
                "relation_key": relation,
                "success": True,
                "counts_toward_task2_coverage": True,
                "samples": samples,
                "episode": str(episode_path.resolve()),
                "report": str(report_path.resolve()),
                "episode_sha256": _sha256(episode_path),
                "report_sha256": _sha256(report_path),
                "skill_ir_sha256": validated["skill_ir_sha256"],
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--min-training", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    records = discover_formal_episodes(args.root, contract)
    manifest = build_training_manifest(
        records,
        heldout_seeds=args.heldout_seeds,
        min_training=args.min_training,
    )
    output = args.output or args.root / "training_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "relation_key": manifest["relation_key"],
                "training_episodes": manifest["training_episode_count"],
                "heldout_episodes": manifest["heldout_episode_count"],
                "training_samples": manifest["training_sample_count"],
                "heldout_samples": manifest["heldout_sample_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
