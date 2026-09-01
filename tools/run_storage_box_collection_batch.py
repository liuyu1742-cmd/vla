"""Resume-safe worker for formal ``organizing::storage_box`` collection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELATION_KEY = "organizing::storage_box"


def _is_successful(root: Path, seed: int) -> bool:
    episode_dir = Path(root) / f"seed_{seed:03d}"
    report_path = episode_dir / "report.json"
    episode_path = episode_dir / "episode.npz"
    if not report_path.is_file() or not episode_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    predicates = report.get("success_predicates")
    return bool(
        report.get("success") is True
        and report.get("relation_key") == RELATION_KEY
        and report.get("counts_toward_task2_coverage") is True
        and isinstance(predicates, dict)
        and predicates.get("simulator_success") is True
    )


def _write_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_command(
    *,
    python: str,
    skill_ir: Path,
    seed: int,
    max_steps: int,
    output_root: Path,
) -> list[str]:
    return [
        python,
        "-u",
        "-m",
        "tools.collect_storage_box_expert",
        "--skill-ir",
        str(skill_ir),
        "--seed",
        str(seed),
        "--frame-stride",
        "1",
        "--max-steps",
        str(max_steps),
        "--output-root",
        str(output_root),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--skill-ir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=ROOT / "outputs" / "storage_box_collection",
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("worker seeds must be unique non-negative integers")
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if not args.skill_ir.is_file():
        raise FileNotFoundError(f"storage-box Skill IR is missing: {args.skill_ir}")

    status_path = args.status_dir / f"worker_{args.worker}.json"
    status: dict[str, object] = {
        "schema_version": "storage_box_collection_worker_v1",
        "relation_key": RELATION_KEY,
        "worker": args.worker,
        "skill_ir": str(args.skill_ir.resolve()),
        "seeds": args.seeds,
        "current_seed": None,
        "completed": [],
        "skipped": [],
        "failed": [],
        "finished": False,
    }
    _write_status(status_path, status)
    for seed in args.seeds:
        if _is_successful(args.output_root, seed):
            status["skipped"].append(seed)
            _write_status(status_path, status)
            print(f"worker={args.worker} seed={seed} status=already_successful", flush=True)
            continue
        status["current_seed"] = seed
        _write_status(status_path, status)
        print(f"worker={args.worker} seed={seed} status=starting", flush=True)
        command = build_command(
            python=sys.executable,
            skill_ir=args.skill_ir,
            seed=seed,
            max_steps=args.max_steps,
            output_root=args.output_root,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode == 0 and _is_successful(args.output_root, seed):
            status["completed"].append(seed)
            print(f"worker={args.worker} seed={seed} status=success", flush=True)
        else:
            status["failed"].append(
                {"seed": seed, "returncode": completed.returncode}
            )
            print(
                f"worker={args.worker} seed={seed} status=failed "
                f"returncode={completed.returncode}",
                flush=True,
            )
        _write_status(status_path, status)
    status["current_seed"] = None
    status["finished"] = True
    status["finished_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    _write_status(status_path, status)
    return 0 if not status["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
