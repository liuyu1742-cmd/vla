"""Probe the formal OpenVLA service on labelled held-out demonstration frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.openvla_tcp_client import predict


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "minimal_pure_vla_eval" / "offline_probe.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "datasets" / "formal_skills" / "organizing_toy" / "training_manifest_dagger_r2.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["held_out"] if int(item["seed"]) == args.seed)
    with np.load(entry["episode"], allow_pickle=False) as episode:
        frames = np.asarray(episode["frames"], dtype=np.uint8)
        actions = np.asarray(episode["actions"], dtype=np.float32)
        phases = np.asarray(episode["canonical_phases"]).astype(str)
        instruction = str(episode["instruction"].item())
    from PIL import Image
    probes = []
    for phase in ("locate", "grasp", "move", "place"):
        indices = np.flatnonzero(phases == phase)
        if not len(indices):
            continue
        index = int(indices[len(indices) // 2])
        image = args.output.parent / f"probe_seed{args.seed}_{phase}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[index]).save(image)
        prediction = predict(args.host, args.port, {"image_path": str(image.resolve()), "instruction": instruction, "canonical_phase": phase, "relation_key": "organizing::toy"})
        probes.append({"phase": phase, "frame_index": index, "label_action": actions[index].tolist(), "predicted_action": prediction})
    args.output.write_text(json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
