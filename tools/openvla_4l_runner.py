"""Run the auditable project-adapted OpenVLA-4L fine-tuning experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.openvla_4l_experiment import (
    FIVE_SHOT_SEEDS,
    MODES,
    is_trainable,
    reduce_language_layers,
)
from tools.openvla_4l_distillation_dataset import (
    action_phase,
    select_indices as _select_distillation_indices,
)


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "pick up the glass cup and place it in the cabinet"
METHOD_LABELS = {
    "full": "Full Fine-tuning",
    "lora_r32": "LoRA (rank=32)",
    "last_layer_only": "Last-Layer-Only",
    "frozen_vision": "Frozen-Vision",
}


def balanced_indices(
    actions: list[np.ndarray] | np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[int]:
    """Backward-compatible equal-phase sampler used by earlier experiments."""

    return _select_distillation_indices(
        np.asarray(actions, dtype=np.float32),
        max_settle_fraction=1.0 / 3.0,
        limit=samples,
        seed=seed,
    )


def collect_five_shot_paths(data_root: Path) -> list[Path]:
    """Return the exact project five-shot split or fail with all missing paths."""
    paths = [
        data_root / f"episode_seed_{seed:03d}.npz" for seed in FIVE_SHOT_SEEDS
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing five-shot demonstrations: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_training_evidence(report_path: Path, output_png: Path) -> None:
    """Render actual training samples, losses, and GPU telemetry as one evidence image."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    canvas = Image.new("RGB", (1600, 900), "#f6f8fb")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34)
    body_font = _font(24)
    mono_font = _font(22)
    draw.text(
        (45, 30),
        f"OpenVLA-4L 瀹為檯璁粌杩囩▼璇佹嵁 路 {report['method_label']}",
        fill="#10233f",
        font=title_font,
    )
    metadata = [
        f"run_id: {report['run_id']}",
        f"checkpoint: {report['checkpoint']}",
        f"GPU: {report['gpu']}",
        f"dataset: water_cup_expert 路 seeds={list(FIVE_SHOT_SEEDS)}",
    ]
    for index, line in enumerate(metadata):
        draw.text((50, 95 + 38 * index), line, fill="#26354a", font=body_font)

    frames = [Path(value) for value in report.get("training_frames", [])]
    x = 50
    for index, path in enumerate(frames[:3]):
        with Image.open(path) as source:
            frame = source.convert("RGB")
            frame.thumbnail((420, 300))
        canvas.paste(frame, (x, 280))
        draw.rectangle((x, 280, x + frame.width, 280 + frame.height), outline="#40566f", width=3)
        draw.text((x, 590), f"鐪熷疄绀烘暀甯?{index + 1}", fill="#26354a", font=body_font)
        x += 500

    draw.rounded_rectangle((45, 650, 1555, 855), radius=12, fill="#111827")
    headers = "step    loss       peak VRAM    GPU used"
    draw.text((75, 675), headers, fill="#7dd3fc", font=mono_font)
    steps = report.get("steps", [])
    visible = steps[-4:] if steps else []
    for row_index, row in enumerate(visible):
        line = (
            f"{int(row['step']):>4}    {float(row['loss']):>8.5f}    "
            f"{float(row['peak_vram_gb']):>7.2f} GB    "
            f"{int(row['gpu_memory_used_mib']):>6} MiB"
        )
        draw.text(
            (75, 720 + 32 * row_index),
            line,
            fill="#e5e7eb",
            font=mono_font,
        )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)


def gpu_snapshot() -> dict[str, float | int | str]:
    """Read one synchronous NVIDIA telemetry sample."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, encoding="utf-8").strip()
    name, used, total, utilization = [part.strip() for part in output.split(",")]
    return {
        "gpu_name": name,
        "gpu_memory_used_mib": int(used),
        "gpu_memory_total_mib": int(total),
        "gpu_utilization_percent": int(utilization),
    }


def action_text(tokenizer: Any, action: np.ndarray) -> str:
    """Encode one continuous seven-dimensional action with OpenVLA bins."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (7,):
        raise ValueError("OpenVLA action must have seven dimensions")
    bins = np.linspace(-1.0, 1.0, 256)
    token_ids = tokenizer.vocab_size - np.digitize(
        np.clip(values, -1.0, 1.0), bins
    )
    return tokenizer.decode(token_ids.tolist())


def build_base(source: Path, output: Path, keep_layers: int) -> dict[str, Any]:
    """Build and reload-validate a reduced OpenVLA base from local weights."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        str(source), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(source),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    reduction = reduce_language_layers(model, keep_layers)
    model.config._name_or_path = str(output.resolve())
    model.language_model.config._name_or_path = str(output.resolve())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    processor.save_pretrained(output)
    del model
    reloaded = AutoModelForVision2Seq.from_pretrained(
        str(output),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    reload_layers = len(reloaded.language_model.model.layers)
    reload_parameters = sum(parameter.numel() for parameter in reloaded.parameters())
    del reloaded
    if reload_layers != keep_layers or reload_parameters != total_parameters:
        raise RuntimeError("OpenVLA-4L reload validation failed")
    report = {
        "schema_version": "openvla_4l_base_v1",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        **reduction,
        "total_parameters": total_parameters,
        "reload_layers": reload_layers,
        "reload_parameters": reload_parameters,
        "status": "validated",
    }
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _load_items(paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray, Path, int]]:
    items: list[tuple[np.ndarray, np.ndarray, Path, int]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            frames = episode["frames"]
            actions = episode["actions"]
            for index in range(len(actions)):
                items.append((frames[index], actions[index], path, index))
    return items


def _save_training_frames(
    items: list[tuple[np.ndarray, np.ndarray, Path, int]], run_dir: Path
) -> list[str]:
    sample_dir = run_dir / "training_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    indexes = (0, len(items) // 2, len(items) - 1)
    paths: list[str] = []
    for order, item_index in enumerate(indexes):
        path = sample_dir / f"sample_{order + 1}.png"
        Image.fromarray(items[item_index][0].astype(np.uint8)).save(path)
        paths.append(str(path.resolve()))
    return paths


def train(
    mode: str,
    base: Path,
    data_root: Path,
    output: Path,
    run_dir: Path,
    steps: int,
    accumulation_steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Train one exact fine-tuning mode and save its independent checkpoint."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Adafactor, AutoModelForVision2Seq, AutoProcessor

    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if steps < 1 or accumulation_steps < 1:
        raise ValueError("steps and accumulation_steps must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    paths = collect_five_shot_paths(data_root)
    items = _load_items(paths)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items)).tolist()
    run_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    training_frames = _save_training_frames(items, run_dir)

    processor = AutoProcessor.from_pretrained(
        str(base), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    last_layer_index = len(model.language_model.model.layers) - 1
    if mode == "lora_r32":
        model = get_peft_model(
            model,
            LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules="all-linear",
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        )
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = is_trainable(name, mode, last_layer_index)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise RuntimeError("selected no trainable parameters")

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.to("cuda:0")
    model.train()
    optimizer = Adafactor(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    step_rows: list[dict[str, Any]] = []
    cursor = 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(steps):
        micro_losses: list[float] = []
        for _ in range(accumulation_steps):
            if cursor >= len(order):
                order = rng.permutation(len(items)).tolist()
                cursor = 0
            frame, action, _, _ = items[order[cursor]]
            cursor += 1
            response = action_text(processor.tokenizer, action)
            prompt = (
                "In: What action should the robot take to "
                f"{INSTRUCTION}?\nOut: {response}</s>"
            )
            input_ids = torch.tensor(
                processor.tokenizer(
                    prompt, add_special_tokens=True
                ).input_ids,
                dtype=torch.long,
                device="cuda:0",
            ).unsqueeze(0)
            labels = input_ids.clone()
            labels[:, :-8] = -100
            pixels = processor.image_processor.apply_transform(
                Image.fromarray(frame.astype(np.uint8))
            ).unsqueeze(0).to("cuda:0", dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(
                    input_ids=input_ids,
                    attention_mask=(
                        input_ids != processor.tokenizer.pad_token_id
                    ),
                    pixel_values=pixels,
                    labels=labels,
                    use_cache=False,
                )
                loss = result.loss / accumulation_steps
            loss.backward()
            micro_losses.append(float(result.loss.detach().cpu()))
            del result, loss, input_ids, labels, pixels
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        telemetry = gpu_snapshot()
        row = {
            "step": update + 1,
            "loss": float(np.mean(micro_losses)),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "elapsed_seconds": time.time() - started,
            **telemetry,
        }
        step_rows.append(row)
        print(
            f"TRAIN mode={mode} step={row['step']}/{steps} "
            f"loss={row['loss']:.6f} peak_vram={row['peak_vram_gb']:.2f}GB "
            f"gpu_used={row['gpu_memory_used_mib']}MiB",
            flush=True,
        )
        with (run_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    model.to("cpu")
    torch.cuda.empty_cache()
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    processor.save_pretrained(output)
    snapshot = gpu_snapshot()
    report = {
        "schema_version": "openvla_4l_training_v1",
        "run_id": run_dir.name,
        "mode": mode,
        "method_label": METHOD_LABELS[mode],
        "base": str(base.resolve()),
        "checkpoint": str(output.resolve()),
        "dataset": str(data_root.resolve()),
        "demonstration_seeds": list(FIVE_SHOT_SEEDS),
        "samples": len(items),
        "optimizer": "Adafactor",
        "learning_rate": learning_rate,
        "physical_batch_size": 1,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": accumulation_steps,
        "optimizer_steps": steps,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "lora_rank": 32 if mode == "lora_r32" else None,
        "gpu": snapshot["gpu_name"],
        "training_frames": training_frames,
        "steps": step_rows,
        "status": "completed",
    }
    report_path = run_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_training_evidence(
        report_path, run_dir / f"{mode}_training_process.png"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-base")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--layers", type=int, default=4)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--mode", choices=MODES, required=True)
    train_parser.add_argument("--base", type=Path, required=True)
    train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=1)
    train_parser.add_argument("--accumulation-steps", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=461)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build-base":
        result = build_base(args.source, args.output, args.layers)
    else:
        result = train(
            mode=args.mode,
            base=args.base,
            data_root=args.data_root,
            output=args.output,
            run_dir=args.run_dir,
            steps=args.steps,
            accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


