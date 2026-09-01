"""Build the stride-1 water-cup manifest and activate it for local training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def split_successful(
    episodes: list[dict[str, Any]], heldout_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful = [episode for episode in episodes if episode.get("success")]
    train = [episode for episode in successful if episode["seed"] != heldout_seed]
    held_out = [episode for episode in successful if episode["seed"] == heldout_seed]
    return train, held_out


def main() -> int:
    source = ROOT / "datasets" / "water_cup_expert_stride1"
    active_root = ROOT / "datasets" / "water_cup_expert"
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(source.glob("episode_seed_*.json"))
    ]
    train, held_out = split_successful(episodes, heldout_seed=2)
    if not train:
        raise RuntimeError("no successful stride-1 training episodes found")
    if not held_out:
        raise RuntimeError("held-out seed 2 stride-1 episode is missing")
    manifest = {
        "held_out_seed": 2,
        "frame_stride": 1,
        "train": train,
        "held_out": held_out,
    }
    source_manifest = source / "training_manifest.json"
    active_manifest = active_root / "training_manifest.json"
    backup_manifest = active_root / "training_manifest_stride4_backup.json"
    if active_manifest.exists() and not backup_manifest.exists():
        backup_manifest.write_text(
            active_manifest.read_text(encoding="utf-8"), encoding="utf-8"
        )
    payload = json.dumps(manifest, indent=2)
    source_manifest.write_text(payload, encoding="utf-8")
    active_manifest.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "train_episodes": len(train),
                "train_samples": sum(int(item["samples"]) for item in train),
                "held_out_episodes": len(held_out),
                "active_manifest": str(active_manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
