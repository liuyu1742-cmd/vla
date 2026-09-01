"""Build a five-seed OpenVLA dataset from nominal and DAgger teacher labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


PHASES = ("open_motion", "closed_motion", "settle")


def _action(value: np.ndarray) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"action must have shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("action must contain only finite values")
    return action


def action_phase(value: np.ndarray) -> str:
    action = _action(value)
    if float(np.linalg.norm(action[:3])) <= 0.05:
        return "settle"
    return "closed_motion" if float(action[6]) >= 0.5 else "open_motion"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw(
    generator: np.random.Generator,
    indexes: list[int],
    count: int,
) -> list[int]:
    if count == 0:
        return []
    if not indexes:
        raise ValueError("required action phase contains no samples")
    return [
        int(index)
        for index in generator.choice(
            indexes,
            size=count,
            replace=len(indexes) < count,
        )
    ]


def select_indices(
    actions: np.ndarray,
    *,
    max_settle_fraction: float,
    limit: int,
    seed: int,
) -> list[int]:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1:] != (7,):
        raise ValueError(f"actions must have shape (N,7), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("actions must contain only finite values")
    if not 0.0 <= max_settle_fraction < 1.0:
        raise ValueError("max_settle_fraction must be in [0,1)")
    if limit < 1:
        raise ValueError("limit must be positive")

    groups = {
        phase: [
            index
            for index, action in enumerate(values)
            if action_phase(action) == phase
        ]
        for phase in PHASES
    }
    if not groups["open_motion"] or not groups["closed_motion"]:
        raise ValueError("open_motion and closed_motion samples are required")

    settle_count = min(
        int(np.floor(limit * max_settle_fraction)),
        limit - 2,
    )
    if not groups["settle"]:
        settle_count = 0
    motion_count = limit - settle_count
    open_count = (motion_count + 1) // 2
    closed_count = motion_count - open_count
    generator = np.random.default_rng(seed)
    selected = (
        _draw(generator, groups["open_motion"], open_count)
        + _draw(generator, groups["closed_motion"], closed_count)
        + _draw(generator, groups["settle"], settle_count)
    )
    generator.shuffle(selected)
    return selected


def _load_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as episode:
        frames = np.asarray(episode["frames"], dtype=np.uint8)
        actions = np.asarray(episode["actions"], dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{path} frames must have shape (N,H,W,3)")
    if actions.ndim != 2 or actions.shape[1:] != (7,):
        raise ValueError(f"{path} actions must have shape (N,7)")
    if len(frames) != len(actions):
        raise ValueError(f"{path} frames/actions are not aligned")
    if not np.isfinite(actions).all():
        raise ValueError(f"{path} actions contain non-finite values")
    return frames, actions


def build_distillation_dataset(
    nominal_root: Path,
    recovery_root: Path,
    output_root: Path,
    *,
    seeds: Sequence[int],
    max_settle_fraction: float,
    samples_per_seed: int,
) -> dict:
    nominal_root = Path(nominal_root)
    recovery_root = Path(recovery_root)
    output_root = Path(output_root)
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")

    recovery_by_seed: dict[int, list[tuple[Path, str]]] = {}
    hashes: dict[str, str] = {}
    for seed in seed_values:
        rows = []
        for path in sorted(
            recovery_root.glob(f"episode_seed_{seed:03d}_round_*.npz")
        ):
            digest = file_sha256(path)
            if digest in hashes:
                raise ValueError(
                    "duplicate recovery content: "
                    f"{path} and {hashes[digest]}"
                )
            hashes[digest] = str(path.resolve())
            rows.append((path, digest))
        recovery_by_seed[seed] = rows

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for seed in seed_values:
        nominal_path = nominal_root / f"episode_seed_{seed:03d}.npz"
        if not nominal_path.is_file():
            raise FileNotFoundError(f"missing nominal episode: {nominal_path}")
        frame_parts, action_parts = [], []
        source_rows = []
        nominal_frames, nominal_actions = _load_episode(nominal_path)
        frame_parts.append(nominal_frames)
        action_parts.append(nominal_actions)
        source_rows.append(
            {
                "kind": "nominal",
                "path": str(nominal_path.resolve()),
                "samples": len(nominal_actions),
            }
        )
        for recovery_path, digest in recovery_by_seed[seed]:
            recovery_frames, recovery_actions = _load_episode(recovery_path)
            frame_parts.append(recovery_frames)
            action_parts.append(recovery_actions)
            source_rows.append(
                {
                    "kind": "dagger_teacher",
                    "path": str(recovery_path.resolve()),
                    "sha256": digest,
                    "samples": len(recovery_actions),
                }
            )
        frames = np.concatenate(frame_parts, axis=0)
        actions = np.concatenate(action_parts, axis=0)
        indexes = select_indices(
            actions,
            max_settle_fraction=max_settle_fraction,
            limit=samples_per_seed,
            seed=461 + seed,
        )
        selected_frames = frames[indexes]
        selected_actions = actions[indexes]
        target = output_root / f"episode_seed_{seed:03d}.npz"
        np.savez_compressed(
            target,
            frames=selected_frames,
            actions=selected_actions,
        )
        counts = {
            phase: sum(
                action_phase(action) == phase for action in selected_actions
            )
            for phase in PHASES
        }
        manifest_rows.append(
            {
                "seed": seed,
                "output": str(target.resolve()),
                "samples": len(selected_actions),
                "phase_counts": counts,
                "sources": source_rows,
            }
        )

    report = {
        "schema_version": "openvla_4l_visual_action_distillation_v1",
        "seeds": list(seed_values),
        "max_settle_fraction": max_settle_fraction,
        "samples_per_seed": samples_per_seed,
        "source_hashes": hashes,
        "rows": manifest_rows,
    }
    (output_root / "distillation_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,5")
    parser.add_argument("--max-settle-fraction", type=float, default=0.15)
    parser.add_argument("--samples-per-seed", type=int, default=900)
    args = parser.parse_args()
    seeds = tuple(
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    )
    report = build_distillation_dataset(
        args.nominal_root,
        args.recovery_root,
        args.output_root,
        seeds=seeds,
        max_settle_fraction=args.max_settle_fraction,
        samples_per_seed=args.samples_per_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
