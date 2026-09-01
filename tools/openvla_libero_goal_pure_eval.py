"""Evaluate selected official LIBERO-Goal tasks with direct OpenVLA actions.

The runner mirrors the observation, prompt, action, gripper, settling, and
success logic of OpenVLA's official ``run_libero_eval.py``.  It intentionally
avoids importing the upstream RLDS training package, which is not required for
inference and is unavailable in the isolated evaluation environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "third_party" / "LIBERO"
ACTION_DIM = 7


def build_openvla_prompt(task: str) -> str:
    """Return the OpenVLA-v1 instruction template used by the official helper."""
    return f"In: What action should the robot take to {task.lower()}?\nOut:"


def parse_task_ids(value: str) -> list[int]:
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


def _configure_libero() -> None:
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "outputs" / "libero_config"))
    libero_root = str(LIBERO_ROOT)
    if libero_root not in sys.path:
        sys.path.insert(0, libero_root)


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_model(model_dir: Path):
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


def _center_crop_image(image: np.ndarray) -> np.ndarray:
    """Apply the official 0.9-area center crop and resize back to input size."""
    import tensorflow as tf

    original_dtype = image.dtype
    height, width = image.shape[:2]
    tensor = tf.convert_to_tensor(image)
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    crop_scale = math.sqrt(0.9)
    offset = (1.0 - crop_scale) / 2.0
    tensor = tf.image.crop_and_resize(
        tf.expand_dims(tensor, axis=0),
        boxes=[[offset, offset, offset + crop_scale, offset + crop_scale]],
        box_indices=[0],
        crop_size=(height, width),
    )[0]
    tensor = tf.clip_by_value(tensor, 0, 1)
    return tf.image.convert_image_dtype(tensor, original_dtype, saturate=True).numpy()


def _resize_official_libero_image(image: np.ndarray) -> np.ndarray:
    """Replicate the JPEG round-trip and Lanczos resize from official LIBERO eval."""
    import tensorflow as tf

    encoded = tf.image.encode_jpeg(image)
    decoded = tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8)
    resized = tf.image.resize(decoded, (224, 224), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()


def _libero_image(observation: dict[str, Any]) -> np.ndarray:
    return _resize_official_libero_image(observation["agentview_image"][::-1, ::-1])


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def _predict_openvla_action(model, processor, image: np.ndarray, task: str, unnorm_key: str) -> np.ndarray:
    import torch
    from PIL import Image

    cropped = _center_crop_image(image)
    pil_image = Image.fromarray(cropped).convert("RGB")
    inputs = processor(build_openvla_prompt(task), pil_image).to("cuda:0", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"unexpected OpenVLA action shape: {action.shape}")
    return action


def _environment_for_task(task):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env


def run_subset(
    model_dir: Path,
    task_ids: list[int],
    trials_per_task: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run direct OpenVLA decisions for selected fixed-state LIBERO-Goal episodes."""
    _configure_libero()
    from libero.libero import benchmark

    if trials_per_task < 1:
        raise ValueError("trials_per_task must be at least one")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    _set_seed(seed)
    model, processor = _load_model(model_dir)
    unnorm_key = "libero_goal"
    if unnorm_key not in model.norm_stats:
        raise KeyError(f"missing action normalization key: {unnorm_key}")

    task_suite = benchmark.get_benchmark_dict()[unnorm_key]()
    if any(task_id >= task_suite.n_tasks for task_id in task_ids):
        raise ValueError(f"task IDs must be smaller than {task_suite.n_tasks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        if trials_per_task > len(initial_states):
            raise ValueError(f"task {task_id} exposes only {len(initial_states)} initial states")
        env = _environment_for_task(task)

        for episode_idx in range(trials_per_task):
            env.reset()
            observation = env.set_init_state(initial_states[episode_idx])
            done = False
            error: str | None = None
            decision_steps = 0
            inference_seconds: list[float] = []

            for step in range(310):
                try:
                    if step < 10:
                        observation, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                        continue
                    image = _libero_image(observation)
                    started = time.perf_counter()
                    action = _predict_openvla_action(model, processor, image, task_description, unnorm_key)
                    inference_seconds.append(time.perf_counter() - started)
                    decision_steps += 1
                    action[-1] = np.sign(2.0 * action[-1] - 1.0)
                    action[-1] *= -1.0
                    observation, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                except Exception as exc:
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
