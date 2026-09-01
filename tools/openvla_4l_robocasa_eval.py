"""Run RoboCasa in its NumPy-2.2.5 environment against an OpenVLA TCP server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tools.openvla_4l_evaluator import render_rollout_evidence
from tools.openvla_4l_experiment import success_rate
from tools.openvla_4l_runner import INSTRUCTION, METHOD_LABELS
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict


ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _patch_scale(environment_class: type, object_scale: float):
    original = environment_class._get_obj_cfgs

    def scaled(environment):
        configurations = original(environment)
        for configuration in configurations:
            if configuration.get("name") == "obj":
                configuration["object_scale"] = object_scale
        return configurations

    environment_class._get_obj_cfgs = scaled
    return original


def evaluate(
    mode: str,
    checkpoint: Path,
    base: Path,
    output_dir: Path,
    seeds: list[int],
    max_steps: int,
    query_interval: int,
    object_scale: float,
    port: int,
) -> dict:
    third_party = ROOT / "third_party"
    sys.path.insert(0, str(third_party / "robosuite"))
    sys.path.insert(0, str(third_party / "robocasa"))
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
        PickPlaceCounterToCabinet,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    original = _patch_scale(PickPlaceCounterToCabinet, object_scale)
    episodes = []
    try:
        for seed in seeds:
            episode_dir = output_dir / f"seed_{seed:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            environment = gym.make(
                "robocasa/PickPlaceCounterToCabinet",
                split="pretrain",
                seed=seed,
                obj_registries=("lightwheel",),
                obj_groups="glass_cup",
                disable_env_checker=True,
            )
            observation, _ = environment.reset(seed=seed)
            current = [0.0] * 7
            records, frame_paths, timings = [], [], []
            succeeded = False
            try:
                for step in range(max_steps):
                    if step % query_interval == 0:
                        frame = np.asarray(
                            observation["video.robot0_agentview_left"]
                        )
                        frame_path = episode_dir / f"frame_{step:04d}.png"
                        Image.fromarray(frame.astype(np.uint8)).save(frame_path)
                        frame_paths.append(str(frame_path.resolve()))
                        started = time.perf_counter()
                        current = predict(
                            "127.0.0.1",
                            port,
                            {
                                "image_path": str(frame_path.resolve()),
                                "instruction": INSTRUCTION,
                            },
                        )
                        timings.append(time.perf_counter() - started)
                    observation, _, _, _, info = environment.step(
                        to_robocasa_action(current)
                    )
                    succeeded = bool(info.get("success", False))
                    records.append(
                        {"step": step, "action": current, "success": succeeded}
                    )
                    if succeeded:
                        break
            finally:
                environment.close()
            episodes.append(
                {
                    "seed": seed,
                    "success": succeeded,
                    "steps": len(records),
                    "mean_inference_seconds": (
                        float(np.mean(timings)) if timings else None
                    ),
                    "frame_paths": frame_paths,
                    "records": records,
                }
            )
            print(
                f"EVAL mode={mode} seed={seed} success={succeeded} "
                f"steps={len(records)}",
                flush=True,
            )
    finally:
        PickPlaceCounterToCabinet._get_obj_cfgs = original
    report = {
        "schema_version": "openvla_4l_closed_loop_v1",
        "mode": mode,
        "method_label": METHOD_LABELS[mode],
        "checkpoint": str(checkpoint.resolve()),
        "base": str(base.resolve()),
        "task": INSTRUCTION,
        "uses_expert_recovery": False,
        "object_scale": object_scale,
        "max_steps": max_steps,
        "query_interval": query_interval,
        "episodes": episodes,
        "successes": sum(row["success"] for row in episodes),
        "trials": len(episodes),
        "success_rate": success_rate(episodes),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_rollout_evidence(
        report_path, output_dir / f"{mode}_rollout_process.png"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(METHOD_LABELS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,5")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--query-interval", type=int, default=1)
    parser.add_argument("--object-scale", type=float, default=0.7)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.mode,
        args.checkpoint,
        args.base,
        args.output_dir,
        parse_seeds(args.seeds),
        args.max_steps,
        args.query_interval,
        args.object_scale,
        args.port,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("mode", "successes", "trials", "success_rate")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

