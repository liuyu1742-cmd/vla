"""Predict one OpenVLA action from a JSON request produced by RoboCasa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    """Require an existing robot-camera image and a phase instruction."""
    image_path = request.get("image_path")
    if not image_path:
        raise ValueError("image_path is required")
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"image_path not found: {image_path}")
    if not request.get("instruction"):
        raise ValueError("instruction is required")
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "openvla-7b")
    args = parser.parse_args()
    request = validate_request(json.loads(args.request.read_text(encoding="utf-8")))
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(str(args.model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True).to("cuda:0")
    prompt = f"In: What action should the robot take to {request['instruction']}?\nOut:"
    inputs = processor(prompt, Image.open(request["image_path"]).convert("RGB")).to("cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    args.response.write_text(json.dumps({"raw_action": [float(x) for x in action]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
