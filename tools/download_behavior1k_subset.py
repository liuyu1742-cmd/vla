"""Download a bounded, paired BEHAVIOR-1K training subset with provenance.

The script deliberately keeps one RGB head camera plus the matching LeRobot
Parquet actions/state table for evenly spaced episodes of every official 2025
challenge task.  It never downloads the full 1.5 TB source dataset.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


REPO_ROOT = "https://huggingface.co/datasets/behavior-1k/2025-challenge-demos/resolve/main"


def download(url: str, destination: Path, retries: int = 3) -> None:
    """Download one file atomically, preserving an existing non-empty file."""
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "OpenVLA-Simulator/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(attempt * 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--limit-tasks", type=int, default=0, help="Non-zero only for a smoke download.")
    args = parser.parse_args()
    if args.episodes_per_task < 1:
        raise ValueError("episodes-per-task must be positive")

    tasks = [json.loads(line) for line in (args.metadata_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    episodes = [json.loads(line) for line in (args.metadata_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    episodes_by_instruction: dict[str, list[dict]] = {}
    for episode in episodes:
        for instruction in episode["tasks"]:
            episodes_by_instruction.setdefault(instruction, []).append(episode)

    selected: list[dict] = []
    for task in tasks[: args.limit_tasks or None]:
        candidates = sorted(episodes_by_instruction[task["task"]], key=lambda item: item["episode_index"])
        stride = max(1, len(candidates) // args.episodes_per_task)
        for episode in candidates[::stride][: args.episodes_per_task]:
            selected.append(
                {
                    "task_index": task["task_index"],
                    "task_name": task["task_name"],
                    "instruction": task["task"],
                    "episode_index": episode["episode_index"],
                    "frames": episode["length"],
                }
            )

    manifest_path = args.output_dir / "manifest.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in selected:
            episode = row["episode_index"]
            task_index = row["task_index"]
            stem = f"episode_{episode:08d}"
            parquet_rel = f"data/task-{task_index:04d}/{stem}.parquet"
            video_rel = f"videos/task-{task_index:04d}/observation.images.rgb.head/{stem}.mp4"
            row["parquet_path"] = parquet_rel
            row["head_rgb_path"] = video_rel
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            download(f"{REPO_ROOT}/{parquet_rel}?download=true", args.output_dir / parquet_rel)
            download(f"{REPO_ROOT}/{video_rel}?download=true", args.output_dir / video_rel)
            print(f"task={task_index:02d} episode={episode:08d} frames={row['frames']}", flush=True)
    print(json.dumps({"status": "complete", "episodes": len(selected), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
