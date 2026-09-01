"""Persistent two-camera OpenVLA-OFT RoboCasa inference service."""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
import threading
from pathlib import Path

import numpy as np

from tools.openvla_tcp_protocol import decode_message, encode_message
from tools.robocasa_oft_model import ACTION_NORM_KEY, normalize, unnormalize
from tools.train_robocasa_oft_head import (
    DEFAULT_MODEL,
    OFT_ROOT,
    _install_windows_resource_stub,
    _load_model,
    _strip_module_prefix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENTS = ROOT / "models" / "openvla-oft-robocasa365-head-r1"


def retrieval_key(hidden: np.ndarray) -> np.ndarray:
    """Compress 56 OFT action-token states into a normalized visual-language key."""
    values = np.asarray(hidden, dtype=np.float32)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None]
    if values.ndim != 3:
        raise ValueError("retrieval hidden states must be a 2-D or 3-D array")
    if values.shape[1] != 56:
        values = values.reshape(values.shape[0], 56, -1)
    pooled = np.concatenate([values.mean(axis=1), values.std(axis=1)], axis=1)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / np.maximum(norms, 1e-12)
    return pooled[0] if squeeze else pooled


def select_retrieval_chunk(
    query_key: np.ndarray, keys: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """Return the normalized action chunk with greatest cosine similarity."""
    query = np.asarray(query_key, dtype=np.float32).reshape(-1)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    candidates = np.asarray(keys, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != len(query):
        raise ValueError("retrieval query and key dimensions do not match")
    if len(candidates) != len(actions):
        raise ValueError("retrieval keys and actions are not aligned")
    similarities = candidates @ query
    index = int(np.argmax(similarities))
    return np.asarray(actions[index], dtype=np.float32), index, float(similarities[index])


def load_retrieval_arrays(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a targeted cache or the train/validation parts of a base cache."""
    targeted_features = root / "features.npy"
    targeted_actions = root / "actions.npy"
    if targeted_features.is_file() and targeted_actions.is_file():
        return (
            np.load(targeted_features, mmap_mode="r"),
            np.load(targeted_actions, mmap_mode="r"),
        )
    feature_parts, action_parts = [], []
    for split in ("train", "val"):
        feature_path = root / f"{split}_features.npy"
        action_path = root / f"{split}_actions.npy"
        if feature_path.is_file() and action_path.is_file():
            feature_parts.append(np.asarray(np.load(feature_path, mmap_mode="r")))
            action_parts.append(np.asarray(np.load(action_path, mmap_mode="r")))
    if not feature_parts:
        raise FileNotFoundError(f"no retrieval feature/action arrays found in {root}")
    return np.concatenate(feature_parts, axis=0), np.concatenate(action_parts, axis=0)


def validate_request(request: dict) -> dict:
    required = ("image_path", "wrist_image_path", "instruction", "proprio")
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(f"request lacks fields: {missing}")
    for key in ("image_path", "wrist_image_path"):
        if not Path(request[key]).is_file():
            raise ValueError(f"image does not exist: {request[key]}")
    proprio = np.asarray(request["proprio"], dtype=np.float32)
    if proprio.shape != (8,) or not np.isfinite(proprio).all():
        raise ValueError("proprio must be a finite 8-D vector")
    if not str(request["instruction"]).strip():
        raise ValueError("instruction must be non-empty")
    return request


class Predictor:
    def __init__(
        self, model_dir: Path, components: Path, retrieval_features: list[Path] | None = None
    ) -> None:
        _install_windows_resource_stub()
        if str(OFT_ROOT) not in sys.path:
            sys.path.insert(0, str(OFT_ROOT))
        import torch
        from PIL import Image
        from tools.robocasa_oft_model import build_trainable_components

        self.torch, self.Image = torch, Image
        self.processor, self.vla = _load_model(model_dir, "cuda:0")
        action_state = _strip_module_prefix(torch.load(components / "action_head.pt", map_location="cpu"))
        proprio_state = _strip_module_prefix(
            torch.load(components / "proprio_projector.pt", map_location="cpu")
        )
        self.action_head, self.proprio_projector = build_trainable_components(
            self.vla,
            device="cuda:0",
            action_head_state=action_state,
            proprio_projector_state=proprio_state,
        )
        self.action_head.eval()
        self.proprio_projector.eval()
        for module in (self.vla, self.action_head, self.proprio_projector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        stats = json.loads((components / "robocasa_oft_stats.json").read_text(encoding="utf-8"))
        self.action_stats = stats["action"]
        self.proprio_stats = stats["proprio"]
        self.vla.norm_stats = {
            ACTION_NORM_KEY: {"action": self.action_stats, "proprio": self.proprio_stats}
        }
        self.retrieval_keys = self.retrieval_actions = None
        if retrieval_features:
            keys, actions = [], []
            for feature_root in retrieval_features:
                root = feature_root.resolve()
                cached_hidden, cached_actions = load_retrieval_arrays(root)
                if cached_hidden.shape[:2] != cached_actions.shape[:2]:
                    raise ValueError(f"retrieval arrays are not aligned: {root}")
                keys.append(retrieval_key(cached_hidden))
                actions.append(np.asarray(cached_actions, dtype=np.float32))
            self.retrieval_keys = np.concatenate(keys, axis=0)
            self.retrieval_actions = np.concatenate(actions, axis=0)
            print(
                f"ROBOCASA_OFT_RETRIEVAL_LOADED samples={len(self.retrieval_keys)}",
                flush=True,
            )
        self.lock = threading.Lock()

    def predict(self, raw_request: dict) -> list[list[float]]:
        request = validate_request(raw_request)
        prompt = (
            "In: What action should the robot take to "
            f"{str(request['instruction']).lower()}?\nOut:"
        )
        primary = self.Image.open(request["image_path"]).convert("RGB")
        wrist = self.Image.open(request["wrist_image_path"]).convert("RGB")
        primary_inputs = self.processor(prompt, primary).to(
            "cuda:0", dtype=self.torch.bfloat16
        )
        wrist_inputs = self.processor(prompt, wrist).to(
            "cuda:0", dtype=self.torch.bfloat16
        )
        primary_inputs["pixel_values"] = self.torch.cat(
            [primary_inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
        )
        proprio = normalize(np.asarray(request["proprio"], dtype=np.float32), self.proprio_stats)
        with self.lock, self.torch.inference_mode():
            action, hidden = self.vla.predict_action(
                **primary_inputs,
                unnorm_key=ACTION_NORM_KEY,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                use_film=False,
            )
            if self.retrieval_keys is not None:
                query = retrieval_key(hidden.float().cpu().numpy())[0]
                normalized_chunk, index, similarity = select_retrieval_chunk(
                    query, self.retrieval_keys, self.retrieval_actions
                )
                action = unnormalize(normalized_chunk, self.action_stats)
                print(
                    f"RETRIEVAL_MATCH index={index} similarity={similarity:.6f}",
                    flush=True,
                )
        from tools.robocasa_oft_rollout import validate_action_chunk

        return validate_action_chunk(action).tolist()


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = decode_message(self.rfile.readline())
            response = {"action_chunk": self.server.predictor.predict(request)}
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        self.wfile.write(encode_message(response))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument(
        "--retrieval-features",
        type=Path,
        action="append",
        help="Targeted feature cache to use for nearest-demonstration imitation.",
    )
    args = parser.parse_args()
    predictor = Predictor(
        args.model_dir.resolve(), args.components.resolve(), args.retrieval_features
    )
    with Server((args.host, args.port), Handler) as server:
        server.predictor = predictor
        print(f"ROBOCASA_OFT_SERVER_READY {args.host}:{args.port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
