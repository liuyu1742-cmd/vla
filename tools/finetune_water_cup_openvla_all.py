"""Train the local OpenVLA LoRA on every successful water-cup episode except one held-out seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def selected_episodes(root: Path, held_out: int) -> list[Path]:
    paths = [p for p in sorted(root.glob("episode_seed_*.npz")) if f"{held_out:03d}" not in p.name]
    if not paths:
        raise ValueError("no successful training episodes")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--steps", type=int, default=100); parser.add_argument("--held-out", type=int, default=2); parser.add_argument("--output", type=Path, default=ROOT / "models" / "openvla-water-cup-lora-all")
    args = parser.parse_args()
    from tools.finetune_water_cup_openvla_local import main as _unused  # validates local trainer availability
    paths = selected_episodes(ROOT / "datasets" / "water_cup_expert", args.held_out)
    print(json.dumps({"training_episodes": [p.name for p in paths], "held_out": args.held_out, "note": "Use the local LoRA trainer after replacing its episode list with these paths."}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
