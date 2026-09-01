"""Run one real OpenVLA training update for each Section 4.6.1 method.

The probe uses one frame/action pair from the project's five-shot water-cup
training split. Physical batch size is one and gradients are accumulated 16
times, giving the user-approved effective batch size of 16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODES = ("full", "lora_r32", "last_layer_only", "frozen_vision")
FIVE_SHOT_SEEDS = (0, 1, 2, 3, 5)


def classify_parameter(name: str, mode: str) -> bool:
    lowered = name.lower()
    if mode == "full":
        return True
    if mode == "frozen_vision":
        return not lowered.startswith("vision_backbone.")
    if mode == "last_layer_only":
        return "language_model.model.layers.31." in lowered or "language_model.lm_head." in lowered
    if mode == "lora_r32":
        return False
    raise ValueError(f"unknown mode: {mode}")


def summarize_named_parameters(named_parameters: Iterable[tuple[str, object]], mode: str) -> dict:
    def size(value: object) -> int:
        return int(value if isinstance(value, int) else value.numel())

    rows = [(name, size(parameter)) for name, parameter in named_parameters]
    return {
        "mode": mode,
        "total_parameters": sum(value for _, value in rows),
        "trainable_parameters": sum(
            value for name, value in rows if classify_parameter(name, mode)
        ),
    }


def action_text(tokenizer, action) -> str:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (7,):
        raise ValueError("OpenVLA action must have seven dimensions")
    bins = np.linspace(-1.0, 1.0, 256)
    ids = tokenizer.vocab_size - np.digitize(np.clip(values, -1.0, 1.0), bins)
    return tokenizer.decode(ids.tolist())


def load_real_sample(data_root: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    paths = [data_root / f"episode_seed_{seed:03d}.npz" for seed in FIVE_SHOT_SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing five-shot demonstrations: {missing}")
    with np.load(paths[0], allow_pickle=False) as episode:
        return episode["frames"][0], episode["actions"][0], paths[0]


def run_probe(
    mode: str,
    model_dir: Path,
    data_root: Path,
    accumulation_steps: int,
) -> dict:
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    record = {
        "schema_version": "openvla_finetune_probe_v2",
        "mode": mode,
        "method_label": {
            "full": "Full Fine-tuning",
            "lora_r32": "LoRA (rank=32)",
            "last_layer_only": "Last-Layer-Only",
            "frozen_vision": "Frozen-Vision",
        }[mode],
        "dataset": "water_cup_expert",
        "task": "pick up the glass cup and place it in the cabinet",
        "demonstration_seeds": list(FIVE_SHOT_SEEDS),
        "demonstrations": len(FIVE_SHOT_SEEDS),
        "physical_batch_size": 1,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": accumulation_steps,
        "lora_rank": 32 if mode == "lora_r32" else None,
        "total_parameters": None,
        "trainable_parameters": None,
        "peak_vram_gb": None,
        "loss": None,
        "status": "error",
        "error": None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        frame, action, sample_path = load_real_sample(data_root)
        record["sample"] = str(sample_path.resolve())
        processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        response = action_text(processor.tokenizer, action)
        prompt = (
            "In: What action should the robot take to pick up the glass cup "
            f"and place it in the cabinet?\nOut: {response}</s>"
        )
        ids = torch.tensor(
            processor.tokenizer(prompt, add_special_tokens=True).input_ids,
            dtype=torch.long,
        ).unsqueeze(0)
        labels = ids.clone()
        labels[:, -8:] = ids[:, -8:]
        labels[:, :-8] = -100
        pixels = processor.image_processor.apply_transform(
            Image.fromarray(frame.astype(np.uint8))
        ).unsqueeze(0)

        model = AutoModelForVision2Seq.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
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
                parameter.requires_grad = classify_parameter(name, mode)
        record["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        record["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if record["trainable_parameters"] == 0:
            raise RuntimeError("selected no trainable parameters")

        model.to("cuda:0")
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-5,
        )
        ids = ids.cuda()
        labels = labels.cuda()
        pixels = pixels.to("cuda", dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(accumulation_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=ids,
                    attention_mask=(ids != processor.tokenizer.pad_token_id),
                    pixel_values=pixels,
                    labels=labels,
                )
                loss = output.loss / accumulation_steps
            loss.backward()
            losses.append(float(output.loss.detach().cpu()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        record["loss"] = float(np.mean(losses))
        record["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        record["status"] = "measured"
    except torch.cuda.OutOfMemoryError as exc:
        record["status"] = "oom"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "openvla-7b")
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "datasets" / "water_cup_expert"
    )
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run_probe(args.mode, args.model_dir, args.data_root, args.accumulation_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["status"] in {"measured", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

