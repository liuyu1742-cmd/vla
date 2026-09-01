"""Serve the 19-class VLA82 RoboCasa LoRA adapter."""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.openvla_ipc_predict import validate_request
from tools.openvla_tcp_protocol import decode_message, encode_message


def load_action_stats(adapter_dir: Path) -> tuple[str, dict]:
    payload = json.loads((Path(adapter_dir) / "vla82_action_stats.json").read_text(encoding="utf-8"))
    key, action = str(payload["key"]), payload["action"]
    if any(len(action[field]) != 7 for field in ("q01", "q99", "mask")):
        raise ValueError("VLA82 action statistics must have seven channels")
    return key, action


class Predictor:
    def __init__(self, model_dir: Path, adapter_dir: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        base = AutoModelForVision2Seq.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, local_files_only=True,
        ).to("cuda:0")
        self.model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True).merge_and_unload()
        self.unnorm_key, action = load_action_stats(adapter_dir)
        self.model.norm_stats = {self.unnorm_key: {"action": action}}

    def predict(self, request: dict) -> list[float]:
        from PIL import Image
        request = validate_request(request)
        prompt = f"In: What action should the robot take to {request['instruction']}?\nOut:"
        inputs = self.processor(prompt, Image.open(request["image_path"]).convert("RGB")).to(
            "cuda:0", dtype=self.torch.bfloat16
        )
        with self.torch.inference_mode():
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "openvla-7b")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "models" / "openvla-vla82-robocasa365-lora-r1")
    args = parser.parse_args()
    with Server((args.host, args.port), Handler) as server:
        server.predictor = Predictor(args.model_dir, args.adapter_dir)
        print(f"OPENVLA_VLA82_SERVER_READY {args.host}:{args.port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
