"""Serve one OpenVLA-4L checkpoint to the separate RoboCasa environment."""

from __future__ import annotations

import argparse
import socketserver
from pathlib import Path

from tools.openvla_ipc_predict import validate_request
from tools.openvla_tcp_protocol import decode_message, encode_message
from tools.water_cup_action_codec import ACTION_NORM_KEY, install_water_cup_action_stats


def processor_root(mode: str, checkpoint: Path, base: Path) -> Path:
    return base if mode == "lora_r32" else checkpoint


class Predictor:
    def __init__(self, mode: str, checkpoint: Path, base: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(processor_root(mode, checkpoint, base)),
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
            model = PeftModel.from_pretrained(
                model, str(checkpoint)
            ).merge_and_unload()
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                str(checkpoint),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        install_water_cup_action_stats(model)
        self.model = model.to("cuda:0").eval()

    def predict(self, request: dict) -> list[float]:
        from PIL import Image

        request = validate_request(request)
        prompt = (
            "In: What action should the robot take to "
            f"{request['instruction']}?\nOut:"
        )
        inputs = self.processor(
            prompt, Image.open(request["image_path"]).convert("RGB")
        ).to("cuda:0", dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=ACTION_NORM_KEY,
                do_sample=False,
            )
        values = [float(value) for value in action]
        if len(values) != 7:
            raise ValueError("OpenVLA policy must return seven action values")
        return values


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = decode_message(self.rfile.readline())
            response = {"raw_action": self.server.predictor.predict(request)}
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        self.wfile.write(encode_message(response))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "lora_r32", "last_layer_only", "frozen_vision"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    with Server(("127.0.0.1", args.port), Handler) as server:
        server.predictor = Predictor(args.mode, args.checkpoint, args.base)
        print(
            f"OPENVLA_4L_SERVER_READY mode={args.mode} port={args.port}",
            flush=True,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

