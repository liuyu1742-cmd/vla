"""Pure-policy RoboCasa evaluation with raw OpenVLA-4L rollout evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from tools.openvla_4l_experiment import success_rate
from tools.openvla_4l_runner import INSTRUCTION, METHOD_LABELS, _font
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.water_cup_action_codec import ACTION_NORM_KEY, install_water_cup_action_stats


ROOT = Path(__file__).resolve().parents[1]


def render_rollout_evidence(report_path: Path, output_png: Path) -> None:
    """Render start, middle, and terminal frames saved by a real rollout."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episode = report["episodes"][0]
    paths = [Path(value) for value in episode["frame_paths"]]
    if not paths:
        raise ValueError("episode must contain saved simulation frames")
    indexes = tuple(dict.fromkeys((0, len(paths) // 2, len(paths) - 1)))
    canvas = Image.new("RGB", (1700, 820), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (45, 30),
        f"OpenVLA-4L 仿真闭环过程证据 · {report['method_label']}",
        fill="#10233f",
        font=_font(34),
    )
    draw.text(
        (50, 90),
        f"checkpoint: {report['checkpoint']}",
        fill="#26354a",
        font=_font(24),
    )
    draw.text(
        (50, 130),
        f"纯策略: 是 · 专家恢复: 否 · seed={episode['seed']} · success={episode['success']}",
        fill="#26354a",
        font=_font(24),
    )
    labels = ("初始观察", "闭环执行中", "终止状态")
    for order, index in enumerate(indexes):
        with Image.open(paths[index]) as source:
            frame = source.convert("RGB")
            frame.thumbnail((500, 420))
        x = 50 + order * 545
        canvas.paste(frame, (x, 210))
        draw.rectangle(
            (x, 210, x + frame.width, 210 + frame.height),
            outline="#40566f",
            width=3,
        )
        draw.text((x, 650), labels[order], fill="#26354a", font=_font(24))
    terminal = episode["records"][-1]
    action = ", ".join(f"{float(value):+.3f}" for value in terminal["action"])
    draw.rounded_rectangle((45, 710, 1655, 785), radius=10, fill="#111827")
    draw.text(
        (70, 730),
        f"terminal step={terminal['step']}  action=[{action}]",
        fill="#e5e7eb",
        font=_font(20),
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)


def _load_policy(mode: str, checkpoint: Path, base: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(base if mode == "lora_r32" else checkpoint),
        trust_remote_code=True,
        local_files_only=True,
    )
    if mode == "lora_r32":
        model = AutoModelForVision2Seq.from_pretrained(
            str(base),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload()
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            str(checkpoint),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    install_water_cup_action_stats(model)
    return model.to("cuda:0").eval(), processor


def _predict(model: Any, processor: Any, frame: np.ndarray) -> list[float]:
    import torch

    prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
    inputs = processor(prompt, Image.fromarray(frame).convert("RGB")).to(
        "cuda:0", dtype=torch.bfloat16
    )
    with torch.inference_mode():
        action = model.predict_action(
            **inputs, unnorm_key=ACTION_NORM_KEY, do_sample=False
        )
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f"invalid OpenVLA action: {values}")
    return [float(value) for value in values]


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
) -> dict[str, Any]:
    """Evaluate one checkpoint and retain concrete episode booleans."""
    third_party = ROOT / "third_party"
    sys.path.insert(0, str(third_party / "robosuite"))
    sys.path.insert(0, str(third_party / "robocasa"))
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
        PickPlaceCounterToCabinet,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = _load_policy(mode, checkpoint, base)
    original = _patch_scale(PickPlaceCounterToCabinet, object_scale)
    episodes = []
    try:
        for seed in seeds:
            episode_dir = output_dir / f"seed_{seed:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            env = gym.make(
                "robocasa/PickPlaceCounterToCabinet",
                split="pretrain",
                seed=seed,
                obj_registries=("lightwheel",),
                obj_groups="glass_cup",
                disable_env_checker=True,
            )
            observation, _ = env.reset(seed=seed)
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
                        current = _predict(model, processor, frame)
                        timings.append(time.perf_counter() - started)
                    observation, _, _, _, info = env.step(
                        to_robocasa_action(current)
                    )
                    succeeded = bool(info.get("success", False))
                    records.append(
                        {"step": step, "action": current, "success": succeeded}
                    )
                    if succeeded:
                        break
            finally:
                env.close()
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
                f"EVAL mode={mode} seed={seed} success={succeeded} steps={len(records)}",
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
    args = parser.parse_args()
    report = evaluate(
        args.mode,
        args.checkpoint,
        args.base,
        args.output_dir,
        [int(value) for value in args.seeds.split(",")],
        args.max_steps,
        args.query_interval,
        args.object_scale,
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

