"""Serve the water-cup LoRA with RoboCasa-compatible action decoding."""

from __future__ import annotations

import argparse
import socketserver
from pathlib import Path
from typing import Any

from tools.openvla_ipc_predict import validate_request
from tools.openvla_tcp_protocol import decode_message, encode_message
from tools.water_cup_action_codec import ACTION_NORM_KEY, install_water_cup_action_stats


ROOT = Path(__file__).resolve().parents[1]


def configure_water_cup_model(model: Any) -> str:
    """Install the action statistics used by the local water-cup dataset."""
    install_water_cup_action_stats(model)
    return ACTION_NORM_KEY


class Predictor:
    def __init__(self, model_dir: Path, adapter_dir: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor

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
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        self.unnorm_key = configure_water_cup_model(self.model)

    def predict(self, request: dict) -> list[float]:
        from PIL import Image

        request = validate_request(request)
        prompt = f"In: What action should the robot take to {request['instruction']}?\nOut:"
        inputs = self.processor(
            prompt, Image.open(request["image_path"]).convert("RGB")
        ).to("cuda:0", dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        return [float(value) for value in action]


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            response = {"raw_action": self.server.predictor.predict(decode_message(self.rfile.readline()))}
        except Exception as error:
            response = {"error": str(error)}
        self.wfile.write(encode_message(response))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "openvla-7b")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "models" / "openvla-water-cup-lora-full",
    )
    args = parser.parse_args()
    with Server(("127.0.0.1", args.port), Handler) as server:
        server.predictor = Predictor(args.model_dir, args.adapter_dir)
        print(f"OPENVLA_WATER_CUP_SERVER_READY 127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
