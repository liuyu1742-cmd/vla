"""Run RoboCasa demonstration playback one episode at a time.

Each episode gets an isolated process, video, and log. A stuck MuJoCo or
FFmpeg process is terminated after the configured timeout without preventing
the remaining episodes from running.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOCASA_ROOT = WORKSPACE / "third_party" / "robocasa"
ROBOSUITE_ROOT = WORKSPACE / "third_party" / "robosuite"
DEFAULT_PLAYBACK = (
    ROBOCASA_ROOT
    / "robocasa"
    / "scripts"
    / "dataset_scripts"
    / "playback_dataset.py"
)
DEFAULT_DATASET = (
    ROBOCASA_ROOT
    / "datasets"
    / "v1.0"
    / "pretrain"
    / "atomic"
    / "PickPlaceCounterToCabinet"
    / "20250819"
    / "lerobot"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "robocasa_official_per_episode"
FAILURE_MARKERS = ("Exception!", "Playback failed with", "Traceback (most recent call last)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--playback-script", type=Path, default=DEFAULT_PLAYBACK)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of episodes to test. Defaults to all remaining episodes.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum seconds allowed for one episode.",
    )
    parser.add_argument("--video-skip", type=int, default=10)
    parser.add_argument("--camera", default="robot0_agentview_left")
    parser.add_argument("--camera-height", type=int, default=320)
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Run episodes again even when the existing report marks them PASS.",
    )
    return parser.parse_args()


def read_total_episodes(dataset: Path) -> int:
    info_path = dataset / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    total = int(info["total_episodes"])
    if total <= 0:
        raise ValueError(f"Invalid total_episodes in {info_path}: {total}")
    return total


def validate_inputs(args: argparse.Namespace) -> int:
    for label, path in (
        ("Python interpreter", args.python),
        ("playback script", args.playback_script),
        ("dataset", args.dataset),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.count is not None and args.count <= 0:
        raise ValueError("--count must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    return read_total_episodes(args.dataset)


def load_results(report_path: Path) -> dict[int, dict[str, Any]]:
    if not report_path.exists():
        return {}
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        return {int(item["episode_index"]): item for item in report.get("results", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def write_reports(
    output_dir: Path,
    dataset: Path,
    timeout: float,
    results: dict[int, dict[str, Any]],
) -> None:
    ordered = [results[index] for index in sorted(results)]
    counts = {status: 0 for status in ("PASS", "FAIL", "TIMEOUT", "SKIPPED")}
    for item in ordered:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "timeout_seconds": timeout,
        "summary": counts,
        "results": ordered,
    }
    json_path = output_dir / "batch_report.json"
    temp_path = json_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(json_path)

    fieldnames = [
        "episode_index",
        "episode_name",
        "status",
        "elapsed_seconds",
        "return_code",
        "video_path",
        "log_path",
        "error",
    ]
    csv_path = output_dir / "batch_report.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: item.get(key, "") for key in fieldnames} for item in ordered)


def stop_process_tree(process: subprocess.Popen[Any], grace_seconds: float = 8.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=grace_seconds)
            return
        except (OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace_seconds)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass


def video_looks_complete(video_path: Path) -> bool:
    if not video_path.exists() or video_path.stat().st_size < 1024:
        return False
    with video_path.open("rb") as handle:
        header = handle.read(64)
    return b"ftyp" in header


def run_episode(
    args: argparse.Namespace,
    episode_index: int,
    videos_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    episode_name = f"episode_{episode_index:06d}"
    video_path = videos_dir / f"{episode_name}.mp4"
    log_path = logs_dir / f"{episode_name}.log"
    video_path.unlink(missing_ok=True)

    command = [
        str(args.python),
        str(args.playback_script),
        "--dataset",
        str(args.dataset),
        "--start-index",
        str(episode_index),
        "--n",
        "1",
        "--video_path",
        str(video_path),
        "--video_skip",
        str(args.video_skip),
        "--render_image_names",
        args.camera,
        "--camera_height",
        str(args.camera_height),
        "--camera_width",
        str(args.camera_width),
        "--verbose",
    ]
    env = os.environ.copy()
    local_paths = os.pathsep.join((str(ROBOCASA_ROOT), str(ROBOSUITE_ROOT)))
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        local_paths if not existing_pythonpath else local_paths + os.pathsep + existing_pythonpath
    )
    env["PYTHONUTF8"] = "1"
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    start = time.monotonic()
    timed_out = False
    return_code: int | None = None

    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ROBOCASA_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        try:
            return_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process_tree(process)
            try:
                return_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                return_code = None
            log_handle.write(f"\nTIMEOUT after {args.timeout:.1f} seconds\n")

    elapsed = round(time.monotonic() - start, 2)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    failure_marker = next((marker for marker in FAILURE_MARKERS if marker in log_text), None)
    video_ok = video_looks_complete(video_path)

    if timed_out:
        status = "TIMEOUT"
        error = f"Exceeded {args.timeout:.1f} seconds"
    elif return_code != 0:
        status = "FAIL"
        error = f"Playback exited with code {return_code}"
    elif failure_marker:
        status = "FAIL"
        error = f"Log contains: {failure_marker}"
    elif not video_ok:
        status = "FAIL"
        error = "Video is missing, too small, or lacks an MP4 header"
    else:
        status = "PASS"
        error = ""

    return {
        "episode_index": episode_index,
        "episode_name": episode_name,
        "status": status,
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "video_path": str(video_path),
        "log_path": str(log_path),
        "error": error,
    }


def main() -> int:
    args = parse_args()
    try:
        total_episodes = validate_inputs(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    end_index = total_episodes
    if args.count is not None:
        end_index = min(end_index, args.start_index + args.count)
    if args.start_index >= total_episodes:
        print(
            f"ERROR: --start-index {args.start_index} is outside 0..{total_episodes - 1}",
            file=sys.stderr,
        )
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output / "videos"
    logs_dir = args.output / "logs"
    videos_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    report_path = args.output / "batch_report.json"
    results = load_results(report_path)
    selected = list(range(args.start_index, end_index))

    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {selected[0]}..{selected[-1]} ({len(selected)} total)")
    print(f"Per-episode timeout: {args.timeout:.1f}s")
    print(f"Output: {args.output}")

    for position, episode_index in enumerate(selected, start=1):
        previous = results.get(episode_index)
        if not args.no_resume and previous and previous.get("status") == "PASS":
            print(
                f"[{position}/{len(selected)}] episode_{episode_index:06d}: SKIP (already PASS)",
                flush=True,
            )
            continue

        print(
            f"[{position}/{len(selected)}] episode_{episode_index:06d}: running...",
            flush=True,
        )
        result = run_episode(args, episode_index, videos_dir, logs_dir)
        results[episode_index] = result
        write_reports(args.output, args.dataset, args.timeout, results)
        print(
            f"[{position}/{len(selected)}] {result['episode_name']}: "
            f"{result['status']} ({result['elapsed_seconds']:.2f}s) "
            f"{result['error']}",
            flush=True,
        )

    selected_results = [results[index] for index in selected if index in results]
    failures = [item for item in selected_results if item["status"] != "PASS"]
    print(
        f"Finished: {len(selected_results) - len(failures)} PASS, "
        f"{len(failures)} FAIL/TIMEOUT"
    )
    print(f"Report: {args.output / 'batch_report.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
