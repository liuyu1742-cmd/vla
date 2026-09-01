from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tfrecord.reader import tfrecord_loader


FIELD = "steps/language_instruction"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample BridgeData language without TensorFlow"
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--episodes-per-shard", type=int, default=4)
    args = parser.parse_args()

    shard_paths = sorted(args.dataset_dir.glob("bridge_dataset-train.tfrecord-*"))
    selected = shard_paths[: args.shards]
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for shard_index, shard_path in enumerate(selected):
        loader = tfrecord_loader(shard_path, None, {FIELD: "byte"})
        for episode_index, record in enumerate(loader):
            if episode_index >= args.episodes_per_shard:
                break
            raw = record.get(FIELD, b"")
            instruction = (
                raw.decode("utf-8", errors="replace").strip()
                if isinstance(raw, bytes)
                else str(raw).strip()
            )
            if not instruction:
                continue
            counts[instruction] += 1
            rows.append(
                {
                    "dataset_id": "bridge_v2",
                    "split": "train",
                    "shard_index": shard_index,
                    "episode_index_in_shard": episode_index,
                    "instruction": instruction,
                    "source_path_or_url": str(shard_path.resolve()),
                    "source_kind": "sampled_episode",
                    "evidence_level": "sampled_episode",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "bridge_language_sample.jsonl"
    sample_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "dataset_id": "bridge_v2",
        "selected_shards": len(selected),
        "sampled_episodes": len(rows),
        "unique_instructions": len(counts),
        "top_instructions": [
            {"instruction": text, "count": count}
            for text, count in counts.most_common(50)
        ],
        "sampling_note": (
            "Deterministic prefix sample; intended for metadata validation, "
            "not an unbiased estimate of the full dataset."
        ),
        "output": str(sample_path.resolve()),
    }
    summary_path = args.output_dir / "bridge_language_sample_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
