"""Wait for formal training, then serve and evaluate three held-out seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from tools.formal_skill_evaluation_evidence import validate_model_evaluation


ROOT = Path(__file__).resolve().parents[1]


def build_pipeline_summary(
    reports: Mapping[int, Mapping[str, Any]], *, required_successes: int
) -> dict[str, Any]:
    if required_successes < 1:
        raise ValueError("required_successes must be positive")
    passed: list[int] = []
    failed: list[int] = []
    invalid: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    for seed in sorted(int(value) for value in reports):
        report = reports[seed]
        if report.get("success") is not True:
            failed.append(seed)
            continue
        try:
            validated = validate_model_evaluation(report)
        except (KeyError, TypeError, ValueError) as error:
            failed.append(seed)
            invalid[str(seed)] = str(error)
            continue
        passed.append(seed)
        evidence[str(seed)] = validated
    attempted = len(reports)
    return {
        "schema_version": "formal_skill_posttrain_pipeline_v1",
        "relation_key": "organizing::toy",
        "attempted_seeds": sorted(int(seed) for seed in reports),
        "passed_seeds": passed,
        "failed_seeds": failed,
        "invalid_evidence": invalid,
        "required_successes": required_successes,
        "success_count": len(passed),
        "success_rate": len(passed) / attempted if attempted else 0.0,
        "acceptance_passed": len(passed) >= required_successes,
        "validated_evidence": evidence,
    }


def _process_exists(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _wait_for_training(
    report_path: Path,
    *,
    training_pid: int | None,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    announced = False
    while time.monotonic() < deadline:
        if report_path.is_file():
            report = _read_json(report_path)
            planned = int(report.get("planned_updates", -1))
            completed = int(report.get("completed_updates", -2))
            if planned > 0 and completed == planned:
                while _process_exists(training_pid) and time.monotonic() < deadline:
                    time.sleep(min(poll_seconds, 5.0))
                return report
        if training_pid is not None and not _process_exists(training_pid):
            raise RuntimeError(
                f"training process {training_pid} exited without a complete report: {report_path}"
            )
        if not announced:
            print(
                f"PIPELINE_WAITING_FOR_TRAINING report={report_path} pid={training_pid}",
                flush=True,
            )
            announced = True
        time.sleep(poll_seconds)
    raise TimeoutError(f"timed out waiting for formal training: {report_path}")


def _wait_for_server(
    process: subprocess.Popen,
    log_path: Path,
    *,
    timeout_seconds: float = 240.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"formal inference server exited with code {process.returncode}")
        if log_path.is_file() and "OPENVLA_FORMAL_SKILL_SERVER_READY" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(2.0)
    raise TimeoutError("formal inference server did not become ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--wait-timeout-hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--required-successes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument(
        "--openvla-python", type=Path, default=Path(sys.executable)
    )
    parser.add_argument(
        "--robocasa-python",
        type=Path,
        default=Path(r"C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "models" / "openvla-organizing-toy-lora",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "formal_skill_posttrain_pipeline",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps != 300:
        raise ValueError("formal post-training acceptance requires exactly 300 decisions")
    if args.poll_seconds <= 0 or args.wait_timeout_hours <= 0:
        raise ValueError("wait and polling durations must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("evaluation seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_report_path = args.adapter_dir / "training_report.json"
    training_report = _wait_for_training(
        training_report_path,
        training_pid=args.training_pid,
        timeout_seconds=args.wait_timeout_hours * 3600.0,
        poll_seconds=args.poll_seconds,
    )
    print(
        f"PIPELINE_TRAINING_COMPLETE updates={training_report['completed_updates']} "
        f"final_loss={training_report.get('final_loss')}",
        flush=True,
    )
    for required in (
        args.adapter_dir / "adapter_config.json",
        args.adapter_dir / "adapter_model.safetensors",
        args.adapter_dir / "formal_skill_action_stats.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"completed training artifact is missing: {required}")

    server_log_path = args.output_dir / "server.log"
    server_error_path = args.output_dir / "server.err.log"
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server_log = server_log_path.open("w", encoding="utf-8")
    server_error = server_error_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [
            str(args.openvla_python),
            "-u",
            "-m",
            "tools.openvla_formal_skill_tcp_server",
            "--port",
            str(args.port),
            "--adapter-dir",
            str(args.adapter_dir.resolve()),
        ],
        cwd=ROOT,
        stdout=server_log,
        stderr=server_error,
        creationflags=creation_flags,
    )
    reports: dict[int, Mapping[str, Any]] = {}
    exit_codes: dict[str, int] = {}
    try:
        _wait_for_server(server, server_log_path)
        print(f"PIPELINE_SERVER_READY pid={server.pid} port={args.port}", flush=True)
        for seed in args.seeds:
            evaluation_log = args.output_dir / f"eval_seed_{seed}.log"
            report_path = ROOT / "outputs" / "formal_skill_eval" / f"seed_{seed:03d}" / "report.json"
            print(
                f"PIPELINE_EVALUATION_START seed={seed} log={evaluation_log}", flush=True
            )
            with evaluation_log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    [
                        str(args.robocasa_python),
                        "-u",
                        "-m",
                        "tools.formal_skill_rollout",
                        "--seed",
                        str(seed),
                        "--steps",
                        str(args.steps),
                        "--action-repeat",
                        str(args.action_repeat),
                        "--port",
                        str(args.port),
                        "--adapter-dir",
                        str(args.adapter_dir.resolve()),
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                    check=False,
                )
            exit_codes[str(seed)] = result.returncode
            if report_path.is_file():
                reports[seed] = _read_json(report_path)
            else:
                reports[seed] = {
                    "seed": seed,
                    "success": False,
                    "error": "evaluation did not produce a report",
                }
            print(
                f"PIPELINE_EVALUATION_END seed={seed} exit_code={result.returncode} "
                f"success={reports[seed].get('success')}",
                flush=True,
            )
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.close()
        server_error.close()

    summary = build_pipeline_summary(
        reports, required_successes=args.required_successes
    )
    summary.update(
        {
            "training_report": str(training_report_path.resolve()),
            "training_manifest_sha256": training_report.get("manifest_sha256"),
            "training_skill_ir_sha256": training_report.get("skill_ir_sha256"),
            "training_completed_updates": training_report.get("completed_updates"),
            "training_final_loss": training_report.get("final_loss"),
            "evaluation_exit_codes": exit_codes,
            "server_log": str(server_log_path.resolve()),
            "server_error_log": str(server_error_path.resolve()),
        }
    )
    summary_path = args.output_dir / "report.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(summary_path.resolve(), flush=True)
    return 0 if summary["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
