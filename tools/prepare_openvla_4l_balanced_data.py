"""Create a phase-balanced view of the same five water-cup demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.openvla_4l_experiment import FIVE_SHOT_SEEDS


def action_phase(action: np.ndarray) -> str:
    values = np.asarray(action, dtype=np.float32)
    if float(np.linalg.norm(values[:3])) <= 0.2:
        return "settle"
    return "closed_motion" if values[-1] >= 0.5 else "open_motion"


def balanced_indices(
    actions: list[np.ndarray], samples: int, seed: int
) -> list[int]:
    if samples < 1:
        raise ValueError("samples must be positive")
    phases = ("open_motion", "closed_motion", "settle")
    groups = {
        phase: [
            index
            for index, action in enumerate(actions)
            if action_phase(action) == phase
        ]
        for phase in phases
    }
    if any(not indexes for indexes in groups.values()):
        raise ValueError("every action phase must contain at least one sample")
    generator = np.random.default_rng(seed)
    return [
        int(generator.choice(groups[phases[index % len(phases)]]))
        for index in range(samples)
    ]


def prepare(source: Path, output: Path, samples_per_demo: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in FIVE_SHOT_SEEDS:
        source_path = source / f"episode_seed_{seed:03d}.npz"
        with np.load(source_path, allow_pickle=False) as episode:
            frames = episode["frames"]
            actions = episode["actions"]
            indices = balanced_indices(
                [action for action in actions],
                samples=samples_per_demo,
                seed=seed + 461,
            )
            selected_frames = frames[indices]
            selected_actions = actions[indices]
        target_path = output / source_path.name
        np.savez_compressed(
            target_path,
            frames=selected_frames,
            actions=selected_actions,
        )
        counts = {
            phase: sum(
                action_phase(action) == phase for action in selected_actions
            )
            for phase in ("open_motion", "closed_motion", "settle")
        }
        rows.append(
            {
                "seed": seed,
                "source": str(source_path.resolve()),
                "output": str(target_path.resolve()),
                "source_samples": len(actions),
                "balanced_samples": len(selected_actions),
                "phase_counts": counts,
            }
        )
    report = {
        "schema_version": "openvla_4l_balanced_five_shot_v1",
        "source_demonstrations": list(FIVE_SHOT_SEEDS),
        "sampling": "equal open_motion, closed_motion, settle",
        "rows": rows,
    }
    (output / "balance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-demo", type=int, default=300)
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.samples_per_demo)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

