"""Serve the audited organizing::toy LoRA with phase-conditioned inference."""

from __future__ import annotations

import argparse
import socketserver
from pathlib import Path
from typing import Any

from tools.finetune_formal_skill_openvla import build_instruction
from tools.formal_skill_action_codec import install_formal_skill_action_stats
from tools.openvla_ipc_predict import validate_request
from tools.openvla_tcp_protocol import decode_message, encode_message


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PHASES = frozenset({"locate", "move", "grasp", "place"})


def conditioned_instruction(request: dict[str, Any]) -> str:
    instruction = request.get("instruction")
    if not instruction:
        raise ValueError("instruction is required")
    phase = str(request.get("canonical_phase", "")).strip().lower()
    if not phase:
        raise ValueError("canonical_phase is required for formal skill inference")
    if phase not in CANONICAL_PHASES:
        raise ValueError(f"unsupported canonical_phase: {phase!r}")
    return build_instruction(str(instruction), phase)


class Predictor:
    def __init__(self, model_dir: Path, adapter_dir: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if not Path(adapter_dir).is_dir():
            raise FileNotFoundError(f"formal LoRA adapter is missing: {adapter_dir}")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        base = AutoModelForVision2Seq.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).to("cuda:0")
        self.model = PeftModel.from_pretrained(
            base, str(adapter_dir), local_files_only=True
        ).merge_and_unload()
        self.unnorm_key = install_formal_skill_action_stats(self.model, adapter_dir)

    def predict(self, request: dict[str, Any]) -> list[float]:
        from PIL import Image

        validated = validate_request(request)
        instruction = conditioned_instruction(validated)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(
            prompt, Image.open(validated["image_path"]).convert("RGB")
        ).to("cuda:0", dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        result = [float(value) for value in action]
        if len(result) != 7:
            raise RuntimeError(f"formal OpenVLA returned {len(result)} action values")
        return result


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = decode_message(self.rfile.readline())
            response = {
                "raw_action": self.server.predictor.predict(request),
                "relation_key": "organizing::toy",
                "canonical_phase": request.get("canonical_phase"),
            }
        except Exception as error:
            response = {"error": str(error)}
        self.wfile.write(encode_message(response))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "openvla-7b")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "models" / "openvla-organizing-toy-lora",
    )
    args = parser.parse_args()
    with Server((args.host, args.port), Handler) as server:
        server.predictor = Predictor(args.model_dir, args.adapter_dir)
        print(
            f"OPENVLA_FORMAL_SKILL_SERVER_READY {args.host}:{args.port} "
            "relation=organizing::toy",
            flush=True,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
