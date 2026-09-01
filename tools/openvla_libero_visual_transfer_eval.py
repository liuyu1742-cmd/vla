"""Run a reproducible visual sim-transfer evaluation for a local OpenVLA model.

The source domain is the native LIBERO camera image.  The target domain applies
a fixed camera-like photometric shift (per-channel gain, bias, and sensor
noise).  The optional calibration module applies the corresponding source-side
camera calibration before OpenVLA inference.  No expert actions, environment
state, or real-robot data are used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.openvla_libero_goal_pure_eval import (
    _configure_libero,
    _environment_for_task,
    _libero_image,
    _load_model,
    _predict_openvla_action,
    _set_seed,
    parse_task_ids,
)


GAINS = np.array([0.72, 1.05, 0.82], dtype=np.float32)
BIAS = np.array([0.04, 0.04, 0.04], dtype=np.float32)
NOISE_STD = 0.02


def target_camera_shift(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Emulate an unseen RGB camera response in the target simulation domain."""
    value = image.astype(np.float32) / 255.0
    value = value * GAINS + BIAS
    value += rng.normal(0.0, NOISE_STD, size=value.shape).astype(np.float32)
    return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def calibrate_target_image(image: np.ndarray) -> np.ndarray:
    """Restore the known affine camera response; noise is deliberately retained."""
    value = image.astype(np.float32) / 255.0
    value = (value - BIAS) / GAINS
    return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def transform(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "source":
        return image
    shifted = target_camera_shift(image, rng)
    if condition == "target_calibrated":
        return calibrate_target_image(shifted)
    return shifted


def evaluate(model_dir: Path, task_ids: list[int], trials: int, seed: int, condition: str, max_steps: int) -> dict:
    _configure_libero()
    from libero.libero import benchmark

    _set_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    model, processor = _load_model(model_dir)
    task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
    episodes: list[dict] = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env = _environment_for_task(task)
        for episode_index in range(trials):
            env.reset()
            observation = env.set_init_state(initial_states[episode_index])
            done = False
            error = None
            decisions = 0
            started = time.perf_counter()
            try:
                for step in range(max_steps):
                    if step < 10:
                        observation, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                        continue
                    image = transform(_libero_image(observation), condition, rng)
                    action = _predict_openvla_action(model, processor, image, task.language, "libero_goal")
                    action[-1] = np.sign(2.0 * action[-1] - 1.0) * -1.0
                    observation, _, done, _ = env.step(action.tolist())
                    decisions += 1
                    if done:
                        break
            except Exception as exc:  # recorded as a failed episode
                error = f"{type(exc).__name__}: {exc}"
            episodes.append({
                "task_id": task_id,
                "task_description": task.language,
                "episode_index": episode_index,
                "success": bool(done),
                "decision_steps": decisions,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": error,
            })
        env.close()

    return {
        "evaluation_kind": "openvla_visual_sim_transfer",
        "condition": condition,
        "seed": seed,
        "trials_per_task": trials,
        "max_steps": max_steps,
        "task_ids": task_ids,
        "target_camera_shift": {
            "gains_rgb": GAINS.tolist(), "bias_rgb": BIAS.tolist(), "gaussian_noise_std": NOISE_STD,
        },
        "uses_real_robot": False,
        "uses_expert_recovery": False,
        "episodes": episodes,
        "successes": sum(item["success"] for item in episodes),
        "episodes_total": len(episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default="1,2,5,7")
    parser.add_argument("--trials-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--condition", choices=("source", "target_shifted", "target_calibrated"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.model_dir, parse_task_ids(args.task_ids), args.trials_per_task, args.seed, args.condition, args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("condition", "successes", "episodes_total", "success_rate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

