"""Evaluate one OpenVLA-4L method with disclosed simulator-state expert recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from tools.openvla_4l_dagger_orchestrator import (
    DEFAULT_OPENVLA_PYTHON,
    DEFAULT_ROBOCASA_PYTHON,
    ROOT,
    wait_until_ready,
)


def guarded_command(
    robocasa_python: Path,
    *,
    seed: int,
    port: int,
    steps: int,
    output_dir: Path,
    max_translation_error: float,
) -> list[str]:
    return [
        str(robocasa_python),
        "-m",
        "tools.openvla_gym_water_cup_guarded_rollout",
        "--seed",
        str(seed),
        "--port",
        str(port),
        "--steps",
        str(steps),
        "--object-scale",
        "0.7",
        "--max-translation-error",
        str(max_translation_error),
        "--output-dir",
        str(output_dir),
    ]


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
    parser.add_argument("--seeds", default="0,1,2,3,5")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--max-translation-error", type=float, default=0.15)
    parser.add_argument("--openvla-python", type=Path, default=DEFAULT_OPENVLA_PYTHON)
    parser.add_argument("--robocasa-python", type=Path, default=DEFAULT_ROBOCASA_PYTHON)
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "server.stdout.log"
    stderr_path = args.output_dir / "server.stderr.log"
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
    rows = []
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
                seed_dir = args.output_dir / f"seed_{seed:03d}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                log_path = seed_dir / "client.log"
                with log_path.open("w", encoding="utf-8") as client_log:
                    result = subprocess.run(
                        guarded_command(
                            args.robocasa_python,
                            seed=seed,
                            port=args.port,
                            steps=args.steps,
                            output_dir=seed_dir,
                            max_translation_error=args.max_translation_error,
                        ),
                        cwd=ROOT,
                        env=environment,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        creationflags=creation_flags,
                        check=False,
                    )
                report_path = seed_dir / "guarded_report.json"
                if not report_path.exists():
                    raise RuntimeError(
                        f"hybrid seed {seed} did not write a report; see {log_path}"
                    )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                row = {
                    "seed": seed,
                    "success": bool(report["success"]),
                    "steps": int(report["steps"]),
                    "ever_grasped": bool(report["ever_grasped"]),
                    "openvla_direct_fraction": float(
                        report["openvla_direct_fraction"]
                    ),
                    "source_counts": report["source_counts"],
                    "report": str(report_path.resolve()),
                    "client_exit": result.returncode,
                }
                rows.append(row)
                print(
                    f"seed={seed} success={row['success']} "
                    f"openvla_direct_fraction={row['openvla_direct_fraction']:.3f}",
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

    successes = sum(row["success"] for row in rows)
    summary = {
        "protocol": "hybrid_expert_recovery",
        "uses_simulator_state_supervisor": True,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "seeds": seeds,
        "max_steps": args.steps,
        "max_translation_error": args.max_translation_error,
        "successes": successes,
        "episodes": len(rows),
        "success_rate": successes / len(rows) if rows else 0.0,
        "mean_openvla_direct_fraction": (
            sum(row["openvla_direct_fraction"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "rows": rows,
    }
    summary_path = args.output_dir / "hybrid_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}))
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
