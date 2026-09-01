"""Run multi-seed RoboCasa DAgger collection against one OpenVLA-4L checkpoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENVLA_PYTHON = Path(
    r"C:\Users\sjtu101\miniconda3\envs\openvla\python.exe"
)
DEFAULT_ROBOCASA_PYTHON = Path(
    r"C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"
)


def collector_command(
    robocasa_python: Path,
    *,
    seed: int,
    round_index: int,
    beta: float,
    port: int,
    steps: int,
    output_dir: Path,
) -> list[str]:
    return [
        str(robocasa_python),
        "-m",
        "tools.collect_water_cup_dagger_v2",
        "--seed",
        str(seed),
        "--round",
        str(round_index),
        "--beta",
        str(beta),
        "--port",
        str(port),
        "--steps",
        str(steps),
        "--object-scale",
        "0.7",
        "--output-dir",
        output_dir.as_posix(),
    ]


def wait_until_ready(process: subprocess.Popen[bytes], stdout_path: Path) -> None:
    deadline = time.monotonic() + 360.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("inference server exited before becoming ready")
        if stdout_path.exists():
            text = stdout_path.read_text(encoding="utf-8", errors="replace")
            if "OPENVLA_4L_SERVER_READY" in text:
                return
        time.sleep(1.0)
    raise TimeoutError("inference server did not become ready within 360 seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("full", "lora_r32", "last_layer_only", "frozen_vision"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seeds", default="0,1,3,5")
    parser.add_argument("--port", type=int, default=8933)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--openvla-python", type=Path, default=DEFAULT_OPENVLA_PYTHON)
    parser.add_argument("--robocasa-python", type=Path, default=DEFAULT_ROBOCASA_PYTHON)
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.log_dir / "server.stdout.log"
    stderr_path = args.log_dir / "server.stderr.log"
    server_command = [
        str(args.openvla_python),
        "-m",
        "tools.openvla_4l_tcp_server",
        "--mode",
        args.mode,
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--base",
        str(args.base.resolve()),
        "--port",
        str(args.port),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        server = subprocess.Popen(
            server_command,
            cwd=ROOT,
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creation_flags,
        )
        try:
            wait_until_ready(server, stdout_path)
            print(f"server ready on port {args.port}", flush=True)
            for seed in seeds:
                log_path = args.log_dir / (
                    f"collector_seed_{seed:03d}_round_{args.round:02d}.log"
                )
                command = collector_command(
                    args.robocasa_python,
                    seed=seed,
                    round_index=args.round,
                    beta=args.beta,
                    port=args.port,
                    steps=args.steps,
                    output_dir=args.output_dir,
                )
                with log_path.open("w", encoding="utf-8") as collector_log:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=collector_log,
                        stderr=subprocess.STDOUT,
                        creationflags=creation_flags,
                        check=False,
                    )
                episode_path = (
                    args.output_dir
                    / f"episode_seed_{seed:03d}_round_{args.round:02d}.npz"
                )
                report_path = (
                    args.output_dir
                    / f"episode_seed_{seed:03d}_round_{args.round:02d}_report.json"
                )
                if not episode_path.exists() or not report_path.exists():
                    raise RuntimeError(
                        f"collector seed {seed} did not persist its episode; "
                        f"see {log_path}"
                    )
                print(
                    f"seed={seed} collector_exit={result.returncode} "
                    f"episode={episode_path}",
                    flush=True,
                )
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

