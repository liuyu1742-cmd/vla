"""Run one local OpenVLA action prediction without reading BridgeData V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_model_dir(model_dir: Path) -> None:
    """Require the local Hugging Face checkpoint configuration."""
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "openvla-7b",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "openvla_inference_smoke_report.json",
    )
    arguments = parser.parse_args()
    validate_model_dir(arguments.model_dir)

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA smoke inference requires CUDA")

    device = "cuda:0"
    dtype = torch.bfloat16
    instruction = "pick up the cup"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    processor = AutoProcessor.from_pretrained(str(arguments.model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(arguments.model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    inputs = processor(prompt, image).to(device, dtype=dtype)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    report: dict[str, Any] = {
        "model_dir": str(arguments.model_dir.resolve()),
        "instruction": instruction,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "predicted_action": [float(value) for value in action],
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
