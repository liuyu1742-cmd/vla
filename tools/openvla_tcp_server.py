"""Serve OpenVLA inference persistently to the separate RoboCasa environment."""

from __future__ import annotations

import argparse
import socketserver
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.openvla_ipc_predict import validate_request
from tools.openvla_tcp_protocol import decode_message, encode_message


class OpenVLAPredictor:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, local_files_only=True,
        ).to("cuda:0")

    def predict(self, request: dict) -> list[float]:
        from PIL import Image

        request = validate_request(request)
        prompt = f"In: What action should the robot take to {request['instruction']}?\nOut:"
        inputs = self.processor(prompt, Image.open(request["image_path"]).convert("RGB")).to("cuda:0", dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return [float(value) for value in action]


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = decode_message(self.rfile.readline())
            response = {"raw_action": self.server.predictor.predict(request)}
        except Exception as error:
            response = {"error": str(error)}
        self.wfile.write(encode_message(response))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "openvla-7b")
    args = parser.parse_args()
    predictor = OpenVLAPredictor(args.model_dir)
    with Server((args.host, args.port), Handler) as server:
        server.predictor = predictor
        print(f"OPENVLA_SERVER_READY {args.host}:{args.port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
