"""Aggregate nominal ``organizing::toy`` episodes with DAgger recovery labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOMINAL = (
    ROOT
    / "datasets"
    / "formal_skills"
    / "organizing_toy"
    / "training_manifest.json"
)
DEFAULT_RECOVERY_ROOT = (
    ROOT / "datasets" / "formal_skills" / "organizing_toy_dagger_r2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "datasets"
    / "formal_skills"
    / "organizing_toy"
    / "training_manifest_dagger_r2.json"
)


def build_dagger_manifest(
    nominal: dict[str, Any],
    recovery: list[dict[str, Any]],
    *,
    recovery_repeat: int,
) -> dict[str, Any]:
    if nominal.get("schema_version") != "formal_skill_training_manifest_v1":
        raise ValueError("unsupported nominal manifest schema")
    if nominal.get("relation_key") != "organizing::toy":
        raise ValueError("nominal manifest relation mismatch")
    if recovery_repeat < 1:
        raise ValueError("recovery_repeat must be positive")
    if not recovery:
        raise ValueError("at least one recovery episode is required")
    train = nominal.get("train")
    held_out = nominal.get("held_out")
    if not isinstance(train, list) or not isinstance(held_out, list):
        raise ValueError("nominal train and held_out must be lists")
    heldout_seeds = {int(item["seed"]) for item in held_out}

    combined: list[dict[str, Any]] = []
    for raw in train:
        item = dict(raw)
        item["source"] = str(item.get("source", "nominal_expert"))
        item["repeat_index"] = int(item.get("repeat_index", 0))
        combined.append(item)

    unique_recovery_samples = 0
    recovery_seeds: set[int] = set()
    for raw in recovery:
        item = dict(raw)
        seed = int(item["seed"])
        if seed in heldout_seeds:
            raise ValueError(
                f"held-out seed {seed} cannot enter DAgger recovery training"
            )
        if item.get("source") != "dagger_recovery":
            raise ValueError("recovery entry source must be dagger_recovery")
        samples = int(item.get("samples", 0))
        if samples < 1:
            raise ValueError("recovery entry must contain positive samples")
        unique_recovery_samples += samples
        recovery_seeds.add(seed)
        for repeat_index in range(recovery_repeat):
            repeated = dict(item)
            repeated["repeat_index"] = repeat_index
            combined.append(repeated)

    training_seeds = {int(item["seed"]) for item in combined}
    overlap = training_seeds & heldout_seeds
    if overlap:
        raise ValueError(f"train and held-out seeds overlap: {sorted(overlap)}")
    training_samples = sum(int(item["samples"]) for item in combined)
    result = dict(nominal)
    result.update(
        {
            "minimum_training_episodes": len(combined),
            "training_episode_count": len(combined),
            "training_sample_count": training_samples,
            "train": combined,
            "held_out": [dict(item) for item in held_out],
            "dagger": {
                "source_manifest": str(nominal.get("source_manifest", "")),
                "recovery_repeat": recovery_repeat,
                "unique_recovery_episodes": len(recovery),
                "unique_recovery_samples": unique_recovery_samples,
                "weighted_recovery_samples": (
                    unique_recovery_samples * recovery_repeat
                ),
                "recovery_seeds": sorted(recovery_seeds),
            },
        }
    )
    return result


def discover_recovery(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for sidecar in sorted(Path(root).glob("seed_*/round_*/manifest_entry.json")):
        item = json.loads(sidecar.read_text(encoding="utf-8"))
        if item.get("source") != "dagger_recovery":
            raise ValueError(f"invalid DAgger sidecar source: {sidecar}")
        episode = Path(str(item.get("episode", "")))
        report = Path(str(item.get("report", "")))
        if not episode.is_file() or not report.is_file():
            raise FileNotFoundError(f"incomplete DAgger sidecar: {sidecar}")
        entries.append(item)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-manifest", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument(
        "--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT
    )
    parser.add_argument("--recovery-repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    nominal = json.loads(args.nominal_manifest.read_text(encoding="utf-8"))
    nominal["source_manifest"] = str(args.nominal_manifest.resolve())
    recovery = discover_recovery(args.recovery_root)
    result = build_dagger_manifest(
        nominal, recovery, recovery_repeat=args.recovery_repeat
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "training_episode_count": result["training_episode_count"],
                "training_sample_count": result["training_sample_count"],
                "heldout_seeds": [
                    int(item["seed"]) for item in result["held_out"]
                ],
                **result["dagger"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
