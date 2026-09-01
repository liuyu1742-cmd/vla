"""Resumable download of the OpenVLA-supported BridgeData V2 RLDS dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


BASE_URL = "https://hf-mirror.com/datasets/quiet-storm/bridgev2-rlds-cache/resolve/main/bridge_dataset/1.0.0"
ROOT = Path(__file__).resolve().parents[1]


def filenames() -> list[str]:
    rows = [f"bridge_dataset-train.tfrecord-{index:05d}-of-01024" for index in range(1024)]
    rows += [f"bridge_dataset-val.tfrecord-{index:05d}-of-00128" for index in range(128)]
    rows += [
        "action_proprio_stats_7d6a416829d818b733e7342f225f3c522a8265a5224e0175f2ab28e26a932ff1.json",
        "action_proprio_stats_9ccaabde9b86d1b6d3bfdf6fcfc1ab326e3265b928ae4dd4cab47e2067b9ecab.json",
        "action_proprio_stats_4203a9a0c2e65f37c33c477d42e8f9d87929e12a4df4e51dbf1479f417d7f3e5.json",
        "action_proprio_stats_eb41baf7d978d166b787f61ef00e0c6a1b747c3dc68e2d19167bf750b025c417.json",
    ]
    rows += ["dataset_info.json", "features.json"]
    return rows


def default_target() -> Path:
    return ROOT / "datasets" / "oxe" / "bridge_orig" / "1.0.0"


def download_one(name: str, target: Path) -> str:
    output = target / name
    subprocess.run(
        [
            "curl.exe", "--fail", "--location", "--continue-at", "-",
            "--retry", "10", "--retry-all-errors", "--output", str(output),
            f"{BASE_URL}/{name}",
        ],
        check=True,
    )
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=default_target())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only for a small connectivity smoke download.")
    args = parser.parse_args()
    names = filenames()[:args.limit] if args.limit else filenames()
    args.target.mkdir(parents=True, exist_ok=True)
    (args.target.parent / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps({"dataset": "BridgeData V2", "source": BASE_URL, "files": len(names)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "starting", "files": len(names), "target": str(args.target)}, ensure_ascii=False), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, name in enumerate(pool.map(lambda item: download_one(item, args.target), names), start=1):
            print(json.dumps({"status": "downloaded", "index": index, "file": name}, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "completed", "files": len(names)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
