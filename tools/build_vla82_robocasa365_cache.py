"""Decode and stride-sample the 19-class training videos into NPZ caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "datasets" / "vla82_robocasa365_19class" / "training_manifest.json"
)


def sample_indices(length: int, stride: int) -> list[int]:
    if length < 1 or stride < 1:
        raise ValueError("length and stride must be positive")
    return list(range(0, length, stride))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args(argv)
    if args.stride < 1:
        raise ValueError("stride must be positive")
    import imageio.v3 as iio

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cached = []
    total = 0
    for position, episode in enumerate(manifest["episodes"], start=1):
        actions = np.load(episode["actions"], allow_pickle=False)
        frames = np.stack(list(iio.imiter(episode["video"], plugin="pyav")))
        aligned = min(len(frames), len(actions))
        indices = np.asarray(sample_indices(aligned, args.stride), dtype=np.int64)
        cache_path = Path(episode["actions"]).with_name(f"training_stride_{args.stride}.npz")
        np.savez_compressed(
            cache_path,
            frames=frames[indices].astype(np.uint8),
            actions=actions[indices].astype(np.float32),
        )
        item = {
            **episode,
            "cache": str(cache_path.resolve()),
            "decoded_frames": int(len(frames)),
            "aligned_frames": aligned,
            "samples": int(len(indices)),
            "stride": args.stride,
        }
        cached.append(item)
        total += len(indices)
        print(
            f"[{position}/{len(manifest['episodes'])}] {episode['task_class']} "
            f"samples={len(indices)}",
            flush=True,
        )
    output = {
        "schema_version": "vla82_robocasa365_cache_v1",
        "source_manifest": str(args.manifest.resolve()),
        "task_class_count": manifest["task_class_count"],
        "episode_count": len(cached),
        "sample_count": total,
        "stride": args.stride,
        "episodes": cached,
    }
    path = args.manifest.with_name(f"training_cache_stride_{args.stride}.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
