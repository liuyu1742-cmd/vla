"""Sidecar-safe DAgger manifest aggregation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.prepare_water_cup_dagger_manifest import EPISODE_PATTERN, ROOT, build_manifest


def discover_recovery(root: Path, *, inspect_orphans: bool = True) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    known_paths: set[str] = set()
    for manifest_path in sorted(root.glob("episode_seed_*_round_*.json")):
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "episode" not in candidate:
            continue
        candidate["episode"] = str(Path(candidate["episode"]).resolve())
        known_paths.add(candidate["episode"].lower())
        episodes.append(candidate)

    if not inspect_orphans:
        return episodes
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
        "--recovery-dir", type=Path, default=ROOT / "datasets" / "water_cup_dagger"
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
    args.active_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.active_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    nominal_samples = sum(
        int(item["samples"])
        for item in manifest["train"]
        if item["source"] == "nominal_expert"
    )
    recovery_samples = sum(int(item["samples"]) for item in recovery)
    print(
        json.dumps(
            {
                "active_manifest": str(args.active_manifest.resolve()),
                "nominal_samples": nominal_samples,
                "unique_recovery_episodes": len(recovery),
                "unique_recovery_samples": recovery_samples,
                "weighted_recovery_samples": recovery_samples * args.recovery_repeat,
                "total_weighted_train_samples": nominal_samples
                + recovery_samples * args.recovery_repeat,
                "held_out_seed": args.held_out_seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
