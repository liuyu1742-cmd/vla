"""Train a continuous OpenVLA-OFT action head on the aligned RoboCasa365 cache.

The VLA backbone is always frozen.  By default the proprio projector and L1
action head are adapted; ``--freeze-proprio`` is available as a memory-safe
fallback after a measured smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from types import ModuleType
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.robocasa_oft_model import normalize, quantile_stats


ROOT = Path(__file__).resolve().parents[1]
OFT_ROOT = ROOT / "third_party" / "openvla-oft"
DEFAULT_MODEL = ROOT / "models" / "openvla-7b-oft-combined-rtx3090-merged"
LOCAL_HF_CACHE = ROOT / "outputs" / "hf_cache"

# Hugging Face dynamic modules are generated even for a fully local checkpoint.
# Keep those writes inside the workspace on restricted Windows installations.
os.environ.setdefault("HF_HOME", str(LOCAL_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(LOCAL_HF_CACHE / "transformers"))


def _install_windows_resource_stub() -> None:
    """Satisfy TFDS's unused POSIX resource import on Windows."""
    if sys.platform != "win32" or "resource" in sys.modules:
        return
    resource = ModuleType("resource")
    resource.RLIMIT_NOFILE = 7
    resource.getrlimit = lambda _limit: (2048, 2048)
    resource.setrlimit = lambda _limit, _value: None
    sys.modules["resource"] = resource


def improvement(value: float, best: float, minimum: float) -> bool:
    """Return whether validation loss improved by the configured margin."""
    return value <= best - minimum


@dataclass
class EarlyStop:
    patience: int
    minimum: float
    best: float = float("inf")
    misses: int = 0

    def update(self, value: float) -> bool:
        if improvement(value, self.best, self.minimum):
            self.best, self.misses = value, 0
        else:
            self.misses += 1
        return self.misses >= self.patience


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("alignment_failures") != 0 or not result.get("episodes"):
        raise ValueError("cache manifest is empty or contains alignment failures")
    return result


def _training_stats(episodes: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [item for item in episodes if item["split"] == "train"]
    if not train:
        raise ValueError("cache has no training episodes")
    proprios = np.concatenate(
        [np.load(item["artifacts"]["proprio"], mmap_mode="r") for item in train], axis=0
    )
    actions = np.concatenate(
        [np.load(item["artifacts"]["actions"], mmap_mode="r") for item in train], axis=0
    )
    return quantile_stats(actions), quantile_stats(proprios)


def _sample_index(
    episodes: list[dict[str, Any]], split: str, stride: int, limit: int | None = None
) -> list[tuple[dict[str, Any], int]]:
    result: list[tuple[dict[str, Any], int]] = []
    for episode in episodes:
        if episode["split"] == split:
            result.extend((episode, frame) for frame in range(0, int(episode["frames"]), stride))
    if limit is not None:
        result = result[:limit]
    return result


def _strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("module."): value for key, value in state.items()}


class FrameReader:
    """Small LRU video reader; sequential episode ordering avoids repeated decoding."""

    def __init__(self) -> None:
        self._path: str | None = None
        self._capture: Any = None
        self._next = 0

    def read(self, path: str, frame: int) -> np.ndarray:
        import cv2

        if path != self._path:
            if self._capture is not None:
                self._capture.release()
            self._capture = cv2.VideoCapture(path)
            if not self._capture.isOpened():
                raise RuntimeError(f"cannot open video: {path}")
            self._path, self._next = path, 0
        if frame != self._next:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self._capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame} from {path}")
        self._next = frame + 1
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()


def _build_transform(processor: Any) -> tuple[Any, Any]:
    _install_windows_resource_stub()
    if str(OFT_ROOT) not in sys.path:
        sys.path.insert(0, str(OFT_ROOT))
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform

    transform = RLDSBatchTransform(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    return transform, collator


def _make_instance(
    episode: dict[str, Any],
    frame: int,
    transform: Any,
    primary_reader: FrameReader,
    wrist_reader: FrameReader,
    action_stats: dict[str, Any],
    proprio_stats: dict[str, Any],
) -> dict[str, Any]:
    actions = np.load(episode["artifacts"]["action_chunks"], mmap_mode="r")[frame]
    proprio = np.load(episode["artifacts"]["proprio"], mmap_mode="r")[frame]
    primary = primary_reader.read(episode["artifacts"]["primary_video"], frame)
    wrist = wrist_reader.read(episode["artifacts"]["wrist_video"], frame)
    rlds = {
        "dataset_name": b"robocasa365_oft",
        "action": normalize(actions, action_stats),
        "observation": {
            "image_primary": primary[None],
            "image_wrist": wrist[None],
            "proprio": normalize(proprio, proprio_stats)[None],
        },
        "task": {"language_instruction": episode["instruction"].encode("utf-8")},
    }
    return transform(rlds)


def _batches(
    index: list[tuple[dict[str, Any], int]],
    batch_size: int,
    transform: Any,
    collator: Any,
    action_stats: dict[str, Any],
    proprio_stats: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    primary_reader, wrist_reader = FrameReader(), FrameReader()
    try:
        instances: list[dict[str, Any]] = []
        for episode, frame in index:
            instances.append(
                _make_instance(
                    episode,
                    frame,
                    transform,
                    primary_reader,
                    wrist_reader,
                    action_stats,
                    proprio_stats,
                )
            )
            if len(instances) == batch_size:
                batch = collator(instances)
                if batch["proprio"].ndim == 1:
                    batch["proprio"] = batch["proprio"].unsqueeze(0)
                yield batch
                instances = []
        if instances:
            batch = collator(instances)
            if batch["proprio"].ndim == 1:
                batch["proprio"] = batch["proprio"].unsqueeze(0)
            yield batch
    finally:
        primary_reader.close()
        wrist_reader.close()


def _load_model(model_path: Path, device: str) -> tuple[Any, Any]:
    _install_windows_resource_stub()
    if str(OFT_ROOT) not in sys.path:
        sys.path.insert(0, str(OFT_ROOT))
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)
    return processor, vla


def _forward_loss(
    vla: Any,
    action_head: Any,
    proprio_projector: Any,
    batch: dict[str, Any],
    device: str,
) -> tuple[Any, Any]:
    import torch
    inputs, action_hidden = _extract_action_hidden(
        vla, proprio_projector, batch, device, gradients=True
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = action_head.predict_action(action_hidden)
        target = batch["actions"].to(device=device, dtype=torch.bfloat16)
        loss = torch.nn.functional.l1_loss(predicted, target)
    return loss, predicted


def _extract_action_hidden(
    vla: Any,
    proprio_projector: Any,
    batch: dict[str, Any],
    device: str,
    *,
    gradients: bool,
) -> tuple[dict[str, Any], Any]:
    """Run the official multimodal path and select its 56 action-token states."""
    import torch
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pixel_values": batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
        "labels": batch["labels"].to(device),
        "output_hidden_states": True,
        "proprio": batch["proprio"].to(device=device, dtype=torch.bfloat16),
        "proprio_projector": proprio_projector,
        "use_film": False,
    }
    context = torch.enable_grad() if gradients else torch.no_grad()
    with context, torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(**inputs)
        token_ids = inputs["labels"][:, 1:]
        action_mask = get_current_action_mask(token_ids) | get_next_actions_mask(token_ids)
        num_patches = vla.vision_backbone.get_num_patches() * 2 + 1
        hidden = output.hidden_states[-1][:, num_patches:-1]
        batch_size = inputs["input_ids"].shape[0]
        action_hidden = hidden[action_mask].reshape(batch_size, 8 * 7, -1).to(torch.bfloat16)
    return inputs, action_hidden


def _component_states(model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    action_path = model_path / "action_head--300000_checkpoint.pt"
    proprio_path = model_path / "proprio_projector--300000_checkpoint.pt"
    if not action_path.exists() or not proprio_path.exists():
        raise FileNotFoundError("combined OFT component checkpoints are missing")
    return (
        _strip_module_prefix(torch.load(action_path, map_location="cpu")),
        _strip_module_prefix(torch.load(proprio_path, map_location="cpu")),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OpenVLA-OFT training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(Path(args.cache).resolve())
    action_stats, proprio_stats = _training_stats(manifest["episodes"])
    stats_payload = {
        "dataset_key": "robocasa365_oft",
        "action": action_stats,
        "proprio": proprio_stats,
        "source_cache": str(Path(args.cache).resolve()),
    }
    _write_json(output / "robocasa_oft_stats.json", stats_payload)

    device = "cuda:0"
    model_path = Path(args.model).resolve()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    processor, vla = _load_model(model_path, device)
    action_state, proprio_state = _component_states(model_path)
    from tools.robocasa_oft_model import build_trainable_components

    action_head, proprio_projector = build_trainable_components(
        vla,
        device=device,
        action_head_state=action_state,
        proprio_projector_state=proprio_state,
    )
    if args.freeze_proprio:
        for parameter in proprio_projector.parameters():
            parameter.requires_grad_(False)
        proprio_projector.eval()
    trainable = [p for p in action_head.parameters() if p.requires_grad]
    trainable += [p for p in proprio_projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    transform, collator = _build_transform(processor)

    train_index = _sample_index(manifest["episodes"], "train", args.stride)
    val_index = _sample_index(manifest["episodes"], "val", args.stride, args.val_samples)
    if not train_index or not val_index:
        raise ValueError("train/validation sample index is empty")
    initial_validation: float | None = None
    history: list[dict[str, Any]] = []
    prediction_samples: list[np.ndarray] = []
    start = time.time()
    smoke_batches = args.smoke_batches if args.smoke_batches > 0 else None
    train_batch_limit = smoke_batches or (
        args.max_train_batches if args.max_train_batches > 0 else None
    )
    epochs = 1 if smoke_batches else args.epochs
    early_stop = EarlyStop(args.patience, args.minimum_improvement)
    best = float("inf")

    # Measure the warm-start checkpoint before making any RoboCasa update.
    action_head.eval()
    proprio_projector.eval()
    baseline_losses: list[float] = []
    with torch.no_grad():
        for val_number, batch in enumerate(
            _batches(
                val_index,
                args.batch_size,
                transform,
                collator,
                action_stats,
                proprio_stats,
            )
        ):
            loss, predicted = _forward_loss(vla, action_head, proprio_projector, batch, device)
            baseline_losses.append(float(loss.detach().cpu()))
            prediction_samples.append(predicted.detach().float().cpu().numpy())
            if smoke_batches is not None and val_number + 1 >= smoke_batches:
                break
    initial_validation = float(np.mean(baseline_losses))

    for epoch in range(epochs):
        action_head.train()
        if not args.freeze_proprio:
            proprio_projector.train()
        train_losses: list[float] = []
        batch_count = 0
        for batch in _batches(
            train_index,
            args.batch_size,
            transform,
            collator,
            action_stats,
            proprio_stats,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, predicted = _forward_loss(vla, action_head, proprio_projector, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            prediction_samples.append(predicted.detach().float().cpu().numpy())
            batch_count += 1
            if train_batch_limit is not None and batch_count >= train_batch_limit:
                break

        action_head.eval()
        proprio_projector.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for val_number, batch in enumerate(
                _batches(
                    val_index,
                    args.batch_size,
                    transform,
                    collator,
                    action_stats,
                    proprio_stats,
                )
            ):
                loss, predicted = _forward_loss(vla, action_head, proprio_projector, batch, device)
                val_losses.append(float(loss.detach().cpu()))
                prediction_samples.append(predicted.detach().float().cpu().numpy())
                if smoke_batches is not None and val_number + 1 >= smoke_batches:
                    break
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        record = {
            "epoch": epoch + 1,
            "train_l1": train_loss,
            "validation_l1": val_loss,
            "train_batches": len(train_losses),
            "validation_batches": len(val_losses),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(action_head.state_dict(), output / "action_head.pt")
            torch.save(proprio_projector.state_dict(), output / "proprio_projector.pt")
        if early_stop.update(val_loss):
            break

    samples = np.concatenate(prediction_samples, axis=0)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    report = {
        "status": "SMOKE_PASS" if smoke_batches else "TRAINING_COMPLETE",
        "model": str(model_path),
        "cache": str(Path(args.cache).resolve()),
        "frozen_vla": True,
        "freeze_proprio": bool(args.freeze_proprio),
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "batch_size": args.batch_size,
        "stride": args.stride,
        "train_samples": len(train_index),
        "validation_samples": len(val_index),
        "initial_validation_l1": initial_validation,
        "best_validation_l1": best,
        "prediction_variance": np.var(samples, axis=0).tolist(),
        "finite_predictions": bool(np.isfinite(samples).all()),
        "peak_allocated_gb": peak_gb,
        "memory_gate_gb": 23.5,
        "memory_gate_pass": peak_gb < 23.5,
        "elapsed_seconds": time.time() - start,
        "history": history,
        "arguments": vars(args),
    }
    _write_json(output / "training_report.json", report)
    if smoke_batches:
        _write_json(output / "memory_smoke.json", report)
    if peak_gb >= 23.5:
        raise RuntimeError(f"memory gate failed: {peak_gb:.3f} GB >= 23.5 GB")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="datasets/vla82_robocasa365_oft/cache_manifest.json")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default="models/openvla-oft-robocasa365-head-r1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--minimum-improvement", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--freeze-proprio", action="store_true")
    parser.add_argument("--seed", type=int, default=82)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
