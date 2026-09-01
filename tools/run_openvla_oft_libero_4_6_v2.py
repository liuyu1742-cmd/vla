"""Run traceable, simulation-only OpenVLA-OFT LIBERO evaluations."""

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
from uuid import uuid4

from tools.check_openvla_oft_environment import CHECKPOINTS, required_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "third_party" / "LIBERO"
WINDOWS_COMPAT_ROOT = REPO_ROOT / "tools" / "windows_compat"
SUITES = tuple(CHECKPOINTS)


def build_command(
    python: Path,
    source_dir: Path,
    suite: str,
    trials_per_task: int,
    seed: int,
    *,
    checkpoint: Path | None = None,
    local_log_dir: Path | None = None,
) -> list[str]:
    """Build an official-policy command using filesystem-safe absolute paths."""
    if suite not in CHECKPOINTS:
        raise ValueError(f"unsupported suite: {suite}")
    if trials_per_task < 1:
        raise ValueError("trials_per_task must be positive")
    source_dir = source_dir.resolve()
    evaluator = (
        source_dir / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    ).resolve()
    checkpoint_value = (
        str(checkpoint.resolve()) if checkpoint is not None else required_checkpoint(suite)
    )
    log_dir = (local_log_dir or Path("official_logs")).resolve()
    return [
        str(python.resolve()),
        str(evaluator),
        "--pretrained_checkpoint",
        checkpoint_value,
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
        "--local_log_dir",
        str(log_dir),
        "--use_wandb",
        "False",
    ]


def parse_run_metrics(text: str) -> dict[str, Any]:
    """Parse official final counters without inferring missing episodes."""
    success_rate: float | None = None
    overall = re.findall(
        r"Overall success rate:\s*([0-9.]+)(?:\s*\(([0-9.]+)%\))?", text
    )
    if overall:
        fraction, percent = overall[-1]
        success_rate = float(percent) if percent else 100.0 * float(fraction)
    elif current := re.findall(r"Current total success rate:\s*([0-9.]+)", text):
        success_rate = 100.0 * float(current[-1])

    episode_matches = re.findall(r"Total episodes:\s*(\d+)", text)
    success_matches = re.findall(r"Total successes:\s*(\d+)", text)
    task_rows: list[dict[str, Any]] = []
    last_task: str | None = None
    for line in text.splitlines():
        clean = re.sub(r"^.*?\[INFO\]\s*", "", line).strip()
        if clean.startswith("Task: "):
            last_task = clean.removeprefix("Task: ").strip()
        elif clean.startswith("Current task success rate:") and last_task:
            value = float(clean.split(":", 1)[1].strip())
            task_rows.append({"task": last_task, "success_rate": 100.0 * value})

    result: dict[str, Any] = {
        "success_rate": success_rate,
        "total_episodes": int(episode_matches[-1]) if episode_matches else None,
        "total_successes": int(success_matches[-1]) if success_matches else None,
        "task_success_rates": task_rows,
    }
    return result


def _source_commit(source_dir: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _gpu_info() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        name, memory_mb, driver = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        return {"name": name, "memory_total_mb": int(memory_mb), "driver": driver}
    except Exception:
        return None


def run_one(
    python: Path,
    source_dir: Path,
    suite: str,
    trials_per_task: int,
    seed: int,
    output_dir: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run one suite and write an immutable manifest beside its videos."""
    source_dir = source_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run_{timestamp}_{uuid4().hex[:8]}"
    run_dir = (output_dir / suite / f"seed_{seed}" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = build_command(
        python,
        source_dir,
        suite,
        trials_per_task,
        seed,
        checkpoint=checkpoint,
        local_log_dir=run_dir / "official_logs",
    )
    env = os.environ.copy()
    python_paths = [
        str(WINDOWS_COMPAT_ROOT.resolve()),
        str(source_dir),
        str(LIBERO_ROOT.resolve()),
        str(REPO_ROOT.resolve()),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["LIBERO_CONFIG_PATH"] = str((REPO_ROOT / "outputs" / "libero_config").resolve())
    env["HF_HOME"] = str((REPO_ROOT / "outputs" / "hf_cache").resolve())
    env["TRANSFORMERS_CACHE"] = str((REPO_ROOT / "outputs" / "hf_cache" / "transformers").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env.setdefault("MUJOCO_GL", "glfw")

    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    ended = datetime.now(timezone.utc)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    combined = completed.stdout + "\n" + completed.stderr
    metrics = parse_run_metrics(combined)
    videos = [str(path.resolve()) for path in sorted(run_dir.rglob("*.mp4"))]
    manifest = {
        "schema_version": "openvla_oft_libero_run_v2",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "suite": suite,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else required_checkpoint(suite),
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
        **metrics,
        "completed": completed.returncode == 0 and metrics["total_episodes"] == 10 * trials_per_task,
        "evidence_kind": "local_measured",
        "uses_expert_recovery": False,
        "gpu": _gpu_info(),
        "official_oft_flags": {
            "parallel_decoding": True,
            "action_chunk_size": 8,
            "continuous_actions": True,
            "l1_regression": True,
            "num_images_in_input": 2,
            "use_proprio": True,
            "center_crop": True,
        },
        "videos": videos,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "third_party" / "openvla-oft")
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "experiment_4_6" / "oft_v2")
    args = parser.parse_args()
    manifest = run_one(
        args.python,
        args.source_dir,
        args.suite,
        args.trials_per_task,
        args.seed,
        args.output_dir,
        args.checkpoint,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

