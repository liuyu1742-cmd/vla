"""Cache frozen RoboCasa OpenVLA-OFT action-token features exactly once."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.train_robocasa_oft_head import (
    DEFAULT_MODEL,
    _batches,
    _build_transform,
    _component_states,
    _extract_action_hidden,
    _load_manifest,
    _load_model,
    _sample_index,
    _strip_module_prefix,
    _training_stats,
    _write_json,
)


def run(args: argparse.Namespace) -> dict:
    import torch
    from tools.robocasa_oft_model import build_trainable_components

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cache_manifest = Path(args.cache).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(cache_manifest)
    action_stats, proprio_stats = _training_stats(manifest["episodes"])
    processor, vla = _load_model(Path(args.model).resolve(), "cuda:0")
    action_state, initial_projector = _component_states(Path(args.model).resolve())
    projector_state = initial_projector
    if args.projector:
        projector_state = _strip_module_prefix(torch.load(args.projector, map_location="cpu"))
    _, projector = build_trainable_components(
        vla,
        device="cuda:0",
        action_head_state=action_state,
        proprio_projector_state=projector_state,
    )
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    transform, collator = _build_transform(processor)
    started = time.time()
    split_reports = {}
    torch.cuda.reset_peak_memory_stats()

    for split in ("train", "val"):
        index = _sample_index(manifest["episodes"], split, args.stride)
        if args.limit > 0:
            index = index[: args.limit]
        if not index:
            raise ValueError(f"empty {split} index")
        feature_path = output / f"{split}_features.npy"
        action_path = output / f"{split}_actions.npy"
        features = None
        actions = np.lib.format.open_memmap(
            action_path, mode="w+", dtype=np.float16, shape=(len(index), 8, 7)
        )
        cursor = 0
        for batch_number, batch in enumerate(
            _batches(
                index,
                args.batch_size,
                transform,
                collator,
                action_stats,
                proprio_stats,
            )
        ):
            _, hidden = _extract_action_hidden(
                vla, projector, batch, "cuda:0", gradients=False
            )
            hidden_np = hidden.reshape(hidden.shape[0], 8, -1).float().cpu().numpy().astype(np.float16)
            if features is None:
                features = np.lib.format.open_memmap(
                    feature_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(index), 8, hidden_np.shape[-1]),
                )
            count = hidden_np.shape[0]
            features[cursor : cursor + count] = hidden_np
            actions[cursor : cursor + count] = batch["actions"].numpy().astype(np.float16)
            cursor += count
            if batch_number % 25 == 0:
                print(f"{split}: {cursor}/{len(index)}", flush=True)
        features.flush()
        actions.flush()
        split_reports[split] = {
            "samples": len(index),
            "features": str(feature_path),
            "actions": str(action_path),
            "feature_shape": list(features.shape),
        }

    report = {
        "status": "FEATURE_CACHE_COMPLETE",
        "source_cache": str(cache_manifest),
        "model": str(Path(args.model).resolve()),
        "projector": str(Path(args.projector).resolve()) if args.projector else "combined_oft_warm_start",
        "stride": args.stride,
        "batch_size": args.batch_size,
        "splits": split_reports,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.time() - started,
        "action_stats": action_stats,
        "proprio_stats": proprio_stats,
    }
    _write_json(output / "feature_cache_manifest.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="datasets/vla82_robocasa365_oft/cache_manifest.json")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--projector")
    parser.add_argument("--output", default="datasets/vla82_robocasa365_oft_features")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
