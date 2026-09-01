"""Aggregate nominal demonstrations and DAgger corrections for fine-tuning."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EPISODE_PATTERN = re.compile(r"episode_seed_(\d+)_round_(\d+)\.npz$")


def build_manifest(
    nominal: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    *,
    heldout_seed: int,
    recovery_repeat: int,
) -> dict[str, Any]:
    if recovery_repeat < 1:
        raise ValueError("recovery_repeat must be positive")
    nominal_train = []
    held_out = []
    for episode in nominal:
        if not episode.get("success", True):
            continue
        tagged = {**episode, "source": "nominal_expert"}
        if int(episode["seed"]) == heldout_seed:
            held_out.append(tagged)
        else:
            nominal_train.append(tagged)

    recovery_train = []
    for episode in recovery:
        if int(episode["seed"]) == heldout_seed:
            continue
        for repeat_index in range(recovery_repeat):
            recovery_train.append(
                {
                    **episode,
                    "source": "dagger_recovery",
                    "repeat_index": repeat_index,
                }
            )
    if not nominal_train:
        raise RuntimeError("no nominal training episodes found")
    if not recovery_train:
        raise RuntimeError("no DAgger recovery episodes found")
    if not held_out:
        raise RuntimeError(f"held-out seed {heldout_seed} is missing")
    return {
        "held_out_seed": heldout_seed,
        "frame_stride": 1,
        "recovery_repeat": recovery_repeat,
        "nominal_episode_count": len(nominal_train),
        "recovery_episode_count": len(recovery),
        "train": nominal_train + recovery_train,
        "held_out": held_out,
    }


def discover_recovery(root: Path) -> list[dict[str, Any]]:
    episodes = []
    known_paths: set[str] = set()
    for manifest_path in sorted(root.glob("episode_seed_*_round_*.json")):
        episode = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode["episode"] = str(Path(episode["episode"]).resolve())
        known_paths.add(str(Path(episode["episode"]).resolve()).lower())
        episodes.append(episode)

    # A process can finish np.savez before a report-write failure. Preserve such
    # fully-shaped recovery data instead of silently discarding it.
    for episode_path in sorted(root.glob("episode_seed_*_round_*.npz")):
        resolved = str(episode_path.resolve())
        if resolved.lower() in known_paths:
            continue
        match = EPISODE_PATTERN.match(episode_path.name)
        if match is None:
            continue
        with np.load(episode_path) as data:
            frames_shape = data["frames"].shape
            actions_shape = data["actions"].shape
        if len(frames_shape) != 4 or actions_shape != (frames_shape[0], 7):
            raise ValueError(f"invalid orphan DAgger episode: {episode_path}")
        episodes.append(
            {
                "seed": int(match.group(1)),
                "round": int(match.group(2)),
                "instruction": "pick up the glass cup and place it in the cabinet",
                "source": "dagger_recovery",
                "success": None,
                "samples": int(frames_shape[0]),
                "beta": None,
                "episode": resolved,
                "report": None,
                "recovered_orphan": True,
            }
        )
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-seed", type=int, default=2)
    parser.add_argument("--recovery-repeat", type=int, default=2)
    parser.add_argument(
        "--nominal-manifest",
        type=Path,
        default=ROOT / "datasets" / "water_cup_expert_stride1" / "training_manifest.json",
    )
    parser.add_argument(
        "--recovery-dir",
        type=Path,
        default=ROOT / "datasets" / "water_cup_dagger",
    )
    parser.add_argument(
        "--active-manifest",
        type=Path,
        default=ROOT / "datasets" / "water_cup_expert" / "training_manifest.json",
    )
    args = parser.parse_args()
    nominal_manifest = json.loads(args.nominal_manifest.read_text(encoding="utf-8"))
    nominal = list(nominal_manifest["train"]) + list(nominal_manifest["held_out"])
    recovery = discover_recovery(args.recovery_dir)
    manifest = build_manifest(
        nominal,
        recovery,
        heldout_seed=args.held_out_seed,
        recovery_repeat=args.recovery_repeat,
    )
    backup = args.active_manifest.with_name("training_manifest_stride1_nominal.json")
    if args.active_manifest.exists() and not backup.exists():
        backup.write_text(args.active_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    args.active_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.active_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    unique_recovery_samples = sum(int(item["samples"]) for item in recovery)
    nominal_samples = sum(int(item["samples"]) for item in manifest["train"] if item["source"] == "nominal_expert")
    summary = {
        "active_manifest": str(args.active_manifest.resolve()),
        "nominal_samples": nominal_samples,
        "unique_recovery_episodes": len(recovery),
        "unique_recovery_samples": unique_recovery_samples,
        "weighted_recovery_samples": unique_recovery_samples * args.recovery_repeat,
        "held_out_seed": args.held_out_seed,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
