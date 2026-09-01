"""Run a selected subset of official LIBERO-Goal tasks with OpenVLA.

This runner deliberately preserves the official OpenVLA/LIBERO observation,
action-normalization, and success protocol while allowing a bounded set of task
IDs for a small, reproducible evaluation.  It does not use recovery actions or
expert control: every post-settling action is predicted by the VLA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = REPO_ROOT / "third_party" / "openvla"
LIBERO_ROOT = REPO_ROOT / "third_party" / "LIBERO"


def parse_task_ids(value: str) -> list[int]:
    """Parse a comma-separated, unique sequence of non-negative task IDs."""
    if not value or not value.strip():
        raise ValueError("task IDs must not be empty")
    try:
        task_ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("task IDs must be integers") from exc
    if any(task_id < 0 for task_id in task_ids):
        raise ValueError("task IDs must be non-negative")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task IDs are not allowed")
    return task_ids


def _configure_import_paths() -> None:
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "outputs" / "libero_config"))
    for path in (str(OPENVLA_ROOT), str(LIBERO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_openvla(model_dir: Path):
    """Load the local checkpoint, fetching only its public custom-code dependency."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    with (model_dir / "dataset_statistics.json").open("r", encoding="utf-8") as handle:
        model.norm_stats = json.load(handle)
    return model, processor


def run_subset(
    model_dir: Path,
    task_ids: list[int],
    trials_per_task: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate selected LIBERO-Goal tasks with direct OpenVLA actions only."""
    _configure_import_paths()
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
    )
    from experiments.robot.openvla_utils import get_vla_action
    from experiments.robot.robot_utils import (
        get_image_resize_size,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )

    if trials_per_task < 1:
        raise ValueError("trials_per_task must be at least one")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    set_seed_everywhere(seed)
    model, processor = _load_openvla(model_dir)
    unnorm_key = "libero_goal"
    if unnorm_key not in model.norm_stats:
        raise KeyError(f"missing action normalization key: {unnorm_key}")

    task_suite = benchmark.get_benchmark_dict()[unnorm_key]()
    if any(task_id >= task_suite.n_tasks for task_id in task_ids):
        raise ValueError(f"task IDs must be smaller than {task_suite.n_tasks}")

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(model_dir),
        center_crop=True,
        unnorm_key=unnorm_key,
        num_steps_wait=10,
    )
    resize_size = get_image_resize_size(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        initial_states = task_suite.get_task_init_states(task_id)
        if trials_per_task > len(initial_states):
            raise ValueError(
                f"task {task_id} has only {len(initial_states)} supplied initial states; "
                f"cannot run {trials_per_task} trials"
            )

        for episode_idx in range(trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            done = False
            error: str | None = None
            decision_steps = 0
            inference_seconds: list[float] = []

            for step in range(300 + cfg.num_steps_wait):
                try:
                    if step < cfg.num_steps_wait:
                        obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                        continue

                    image = get_libero_image(obs, resize_size)
                    observation = {
                        "full_image": image,
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                    }
                    started = time.perf_counter()
                    action = get_vla_action(
                        model,
                        processor,
                        cfg.pretrained_checkpoint,
                        observation,
                        task_description,
                        cfg.unnorm_key,
                        center_crop=cfg.center_crop,
                    )
                    inference_seconds.append(time.perf_counter() - started)
                    decision_steps += 1
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                except Exception as exc:  # Record simulator/model errors as failed evidence.
                    error = f"{type(exc).__name__}: {exc}"
                    break

            episodes.append(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_index": episode_idx,
                    "success": bool(done),
                    "decision_steps": decision_steps,
                    "mean_inference_seconds": float(np.mean(inference_seconds)) if inference_seconds else None,
                    "error": error,
                }
            )
        env.close()

    task_summaries = []
    for task_id in task_ids:
        rows = [row for row in episodes if row["task_id"] == task_id]
        task_summaries.append(
            {
                "task_id": task_id,
                "task_description": rows[0]["task_description"],
                "episodes": len(rows),
                "successes": sum(row["success"] for row in rows),
                "success_rate": sum(row["success"] for row in rows) / len(rows),
            }
        )
    report = {
        "evaluation_kind": "official_libero_goal_pure_vla",
        "uses_expert_recovery": False,
        "checkpoint": str(model_dir),
        "seed": seed,
        "trials_per_task": trials_per_task,
        "tasks": task_summaries,
        "episodes": episodes,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default="1,2,5,7")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_subset(
        model_dir=args.model_dir,
        task_ids=parse_task_ids(args.task_ids),
        trials_per_task=args.trials_per_task,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["tasks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
