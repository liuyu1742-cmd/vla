"""Evaluate visual sim-transfer conditions and persist actual keyframes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.openvla_libero_goal_pure_eval import (
    _configure_libero,
    _environment_for_task,
    _libero_image,
    _load_model,
    _predict_openvla_action,
    _set_seed,
    parse_task_ids,
)
from tools.openvla_libero_visual_transfer_eval import transform


def run_condition(model_dir: Path, condition: str, output_dir: Path, seed: int) -> dict:
    _configure_libero()
    from libero.libero import benchmark

    _set_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    model, processor = _load_model(model_dir)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task_ids = [1, 2, 5, 7]
    episodes = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        env = _environment_for_task(task)
        env.reset()
        observation = env.set_init_state(suite.get_task_init_states(task_id)[0])
        task_dir = output_dir / condition / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        initial_source = _libero_image(observation)
        initial_model_input = transform(initial_source, condition, rng)
        Image.fromarray(initial_source).save(task_dir / "source_initial.png")
        Image.fromarray(initial_model_input).save(task_dir / "model_input_initial.png")

        done = False
        error = None
        decisions = 0
        start = time.perf_counter()
        try:
            for step in range(310):
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
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        final_source = _libero_image(observation)
        final_model_input = transform(final_source, condition, rng)
        Image.fromarray(final_source).save(task_dir / "source_final.png")
        Image.fromarray(final_model_input).save(task_dir / "model_input_final.png")
        episodes.append({
            "task_id": task_id,
            "task_description": task.language,
            "success": bool(done),
            "decision_steps": decisions,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "error": error,
            "keyframes": {
                "source_initial": str(task_dir / "source_initial.png"),
                "model_input_initial": str(task_dir / "model_input_initial.png"),
                "source_final": str(task_dir / "source_final.png"),
                "model_input_final": str(task_dir / "model_input_final.png"),
            },
        })
        env.close()

    report = {
        "evaluation_kind": "openvla_visual_sim_transfer_keyframes",
        "condition": condition,
        "seed": seed,
        "tasks": task_ids,
        "uses_real_robot": False,
        "uses_expert_recovery": False,
        "episodes": episodes,
        "successes": sum(item["success"] for item in episodes),
        "episodes_total": len(episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
    }
    (output_dir / condition / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("condition", "successes", "episodes_total", "success_rate")}, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("source", "target_shifted", "target_calibrated"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    run_condition(args.model_dir, args.condition, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
