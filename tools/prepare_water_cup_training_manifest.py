"""Create an auditable manifest of all successful water-cup expert episodes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    root = ROOT / "datasets" / "water_cup_expert"
    episodes = []
    for manifest_path in sorted(root.glob("episode_seed_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("success"):
            episodes.append(manifest)
    output = root / "training_manifest.json"
    output.write_text(json.dumps({"held_out_seed": 2, "train": [item for item in episodes if item["seed"] != 2], "held_out": [item for item in episodes if item["seed"] == 2]}, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
