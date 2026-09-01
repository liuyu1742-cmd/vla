"""Run the unmodified official OpenVLA-OFT LIBERO evaluator with manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from tools.check_openvla_oft_environment import CHECKPOINTS, required_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "third_party" / "LIBERO"
SUITES = tuple(CHECKPOINTS)


def build_command(
    python: Path,
    source_dir: Path,
    suite: str,
    trials_per_task: int,
    seed: int,
) -> list[str]:
    if trials_per_task < 1:
        raise ValueError("trials_per_task must be positive")
    return [
        str(python),
        str(source_dir / "experiments" / "robot" / "libero" / "run_libero_eval.py"),
        "--pretrained_checkpoint",
        required_checkpoint(suite),
        "--task_suite_name",
        suite,
        "--num_trials_per_task",
        str(trials_per_task),
        "--seed",
        str(seed),
        "--center_crop",
        "True",
        "--num_open_loop_steps",
        "8",
        "--use_l1_regression",
        "True",
        "--use_diffusion",
        "False",
        "--use_film",
        "False",
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--use_wandb",
        "False",
    ]


def parse_success_counts(text: str) -> dict[str, float | None]:
    overall = re.findall(
        r"Overall success rate:\s*([0-9.]+)(?:\s*\(([0-9.]+)%\))?", text
    )
    if overall:
        fraction, percent = overall[-1]
        return {"success_rate": float(percent) if percent else 100.0 * float(fraction)}
    current = re.findall(r"Current total success rate:\s*([0-9.]+)", text)
    return {"success_rate": 100.0 * float(current[-1]) if current else None}


def _source_commit(source_dir: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_one(
    python: Path,
    source_dir: Path,
    suite: str,
    trials_per_task: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = output_dir / suite / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = build_command(python, source_dir, suite, trials_per_task, seed)
    env = os.environ.copy()
    python_paths = [
        str(source_dir.resolve()),
        str(LIBERO_ROOT.resolve()),
        str(REPO_ROOT.resolve()),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["LIBERO_CONFIG_PATH"] = str((REPO_ROOT / "outputs" / "libero_config").resolve())
    env["HF_HOME"] = str((REPO_ROOT / "outputs" / "hf_cache").resolve())
    env["TRANSFORMERS_CACHE"] = str(
        (REPO_ROOT / "outputs" / "hf_cache" / "transformers").resolve()
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=source_dir,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    ended = datetime.now(timezone.utc)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    parsed = parse_success_counts(completed.stdout + "\n" + completed.stderr)
    manifest = {
        "schema_version": "openvla_oft_libero_run_v1",
        "suite": suite,
        "checkpoint": required_checkpoint(suite),
        "source_commit": _source_commit(source_dir),
        "seed": seed,
        "trials_per_task": trials_per_task,
        "tasks": 10,
        "requested_trials": 10 * trials_per_task,
        "command": command,
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "returncode": completed.returncode,
        "success_rate": parsed["success_rate"],
        "evidence_kind": "local_measured",
        "uses_expert_recovery": False,
        "official_oft_flags": {
            "parallel_decoding": True,
            "action_chunk_size": 8,
            "continuous_actions": True,
            "l1_regression": True,
            "num_images_in_input": 2,
            "use_proprio": True,
            "center_crop": True,
        },
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--source-dir", type=Path, default=REPO_ROOT / "third_party" / "openvla-oft"
    )
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiment_4_6" / "oft",
    )
    args = parser.parse_args()
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    invalid = sorted(set(suites) - set(SUITES))
    if invalid:
        raise ValueError(f"unsupported suites: {invalid}")
    manifests = [
        _run_one(
            args.python,
            args.source_dir,
            suite,
            args.trials_per_task,
            args.seed,
            args.output_dir,
        )
        for suite in suites
    ]
    summary = {
        "schema_version": "openvla_oft_libero_batch_v1",
        "runs": manifests,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"summary_seed_{args.seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(row["returncode"] == 0 for row in manifests) else 1


if __name__ == "__main__":
    raise SystemExit(main())

