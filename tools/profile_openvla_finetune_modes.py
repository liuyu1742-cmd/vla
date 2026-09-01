"""Profile OpenVLA fine-tuning parameter modes on the local GPU.

Each mode runs in a fresh subprocess so an OOM in a large configuration cannot
contaminate the CUDA allocator state of the next measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


VALID_MODES = ("full", "lora_r32", "last_layer_only", "frozen_vision")
REPO_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = REPO_ROOT / "third_party" / "openvla"


def classify_parameter(name: str, mode: str) -> bool:
    """Return whether an existing base-model parameter is trainable."""
    if mode == "full":
        return True
    if mode == "frozen_vision":
        lowered = name.lower()
        return "vision_backbone" not in lowered and "vision_tower" not in lowered
    if mode == "last_layer_only":
        lowered = name.lower()
        return (
            "layers.31." in lowered
            or "layer.31." in lowered
            or "lm_head." in lowered
            or lowered.endswith("lm_head.weight")
        )
    if mode == "lora_r32":
        return False
    raise ValueError(f"unknown fine-tuning mode: {mode}")


def summarize_named_parameters(named_parameters: Iterable[tuple[str, object]], mode: str) -> dict:
    """Count total and selected parameters from tensors or integer sizes."""

    def count(value: object) -> int:
        if isinstance(value, int):
            return value
        numel = getattr(value, "numel", None)
        if numel is None:
            raise TypeError("parameter values must be integer sizes or expose numel()")
        return int(numel())

    rows = [(name, count(parameter)) for name, parameter in named_parameters]
    return {
        "mode": mode,
        "total_parameters": sum(size for _, size in rows),
        "trainable_parameters": sum(
            size for name, size in rows if classify_parameter(name, mode)
        ),
    }


def _first_training_sample(data_root: Path) -> Path:
    paths = sorted(data_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no NPZ demonstration found under {data_root}")
    return paths[0]


def _worker(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    sys.path.insert(0, str(OPENVLA_ROOT))
    from peft import LoraConfig, get_peft_model
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer

    record = {
        "mode": args.worker_mode,
        "effective_batch_size": args.effective_batch_size,
        "physical_batch_size": 1,
        "gradient_accumulation_steps": args.effective_batch_size,
        "trainable_parameters": None,
        "total_parameters": None,
        "peak_vram_gb": None,
        "status": "unsupported",
        "error": None,
        "model_dir": str(args.model_dir.resolve()),
        "sample": None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        sample_path = _first_training_sample(args.data_root)
        record["sample"] = str(sample_path.resolve())
        with np.load(sample_path) as episode:
            frame = episode["frames"][0]
            action = episode["actions"][0]

        processor = AutoProcessor.from_pretrained(
            str(args.model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        prompt = PurePromptBuilder("openvla")
        prompt.add_turn("human", "What action should the robot take to complete the task?")
        prompt.add_turn("gpt", action_tokenizer(action))
        ids = torch.tensor(
            processor.tokenizer(prompt.get_prompt(), add_special_tokens=True).input_ids,
            dtype=torch.long,
        ).unsqueeze(0)
        labels = ids.clone()
        labels[:, : -(len(action) + 1)] = -100
        pixels = processor.image_processor.apply_transform(
            Image.fromarray(frame.astype(np.uint8))
        ).unsqueeze(0)

        model = AutoModelForVision2Seq.from_pretrained(
            str(args.model_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        if args.worker_mode == "lora_r32":
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
                parameter.requires_grad = classify_parameter(name, args.worker_mode)

        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        record["total_parameters"] = total
        record["trainable_parameters"] = trainable
        if trainable == 0:
            raise RuntimeError("fine-tuning mode selected zero parameters")

        model = model.to("cuda:0")
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
        for _ in range(args.effective_batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=ids,
                    attention_mask=(ids != processor.tokenizer.pad_token_id),
                    pixel_values=pixels,
                    labels=labels,
                )
                loss = output.loss / args.effective_batch_size
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        record["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        record["status"] = "measured"
    except torch.cuda.OutOfMemoryError as exc:
        record["status"] = "oom"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    except Exception as exc:
        record["status"] = "unsupported"
        record["error"] = f"{type(exc).__name__}: {exc}"

    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))
    return 0 if record["status"] in {"measured", "oom"} else 1


def _parent(args: argparse.Namespace) -> int:
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    invalid = sorted(set(modes) - set(VALID_MODES))
    if invalid:
        raise ValueError(f"unknown modes: {invalid}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in modes:
        output = args.output_dir / f"{mode}.json"
        command = [
            sys.executable,
            "-m",
            "tools.profile_openvla_finetune_modes",
            "--worker-mode",
            mode,
            "--worker-output",
            str(output),
            "--model-dir",
            str(args.model_dir),
            "--data-root",
            str(args.data_root),
            "--effective-batch-size",
            str(args.effective_batch_size),
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if output.exists():
            records.append(json.loads(output.read_text(encoding="utf-8")))
        else:
            records.append(
                {
                    "mode": mode,
                    "status": "unsupported",
                    "error": f"worker exited {completed.returncode} without a record",
                }
            )
    payload = {
        "schema_version": "openvla_finetune_profile_v1",
        "effective_batch_size": args.effective_batch_size,
        "records": records,
    }
    (args.output_dir / "profile.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models" / "openvla-7b")
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "datasets" / "robocasa_vla"
    )
    parser.add_argument("--modes", default=",".join(VALID_MODES))
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiment_4_6" / "finetune_modes",
    )
    parser.add_argument("--worker-mode", choices=VALID_MODES)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.effective_batch_size < 1:
        raise ValueError("--effective-batch-size must be positive")
    if args.worker_mode:
        if args.worker_output is None:
            raise ValueError("--worker-output is required with --worker-mode")
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

