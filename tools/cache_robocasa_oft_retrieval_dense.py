"""Build a dense OpenVLA retrieval cache from one public success per task class."""

from __future__ import annotations

import argparse
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
    _strip_module_prefix,
    _training_stats,
    _write_json,
)


def select_shortest_per_task_class(episodes: list[dict]) -> list[dict]:
    """Choose the shortest public successful episode for each simulator task."""
    selected: dict[str, dict] = {}
    for episode in episodes:
        task_class = str(episode["task_class"])
        current = selected.get(task_class)
        candidate_key = (int(episode["frames"]), int(episode["episode_index"]))
        current_key = (
            (int(current["frames"]), int(current["episode_index"]))
            if current is not None
            else None
        )
        if current_key is None or candidate_key < current_key:
            selected[task_class] = episode
    return [selected[key] for key in sorted(selected)]


def run(args: argparse.Namespace) -> dict:
    import torch
    from tools.robocasa_oft_model import build_trainable_components

    manifest = _load_manifest(Path(args.cache).resolve())
    episodes = manifest["episodes"]
    action_stats, proprio_stats = _training_stats(episodes)
    selected = select_shortest_per_task_class(episodes)
    index = [
        (episode, frame)
        for episode in selected
        for frame in range(0, int(episode["frames"]), args.stride)
    ]
    if not index:
        raise ValueError("dense retrieval index is empty")

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

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    actions = np.lib.format.open_memmap(
        output / "actions.npy", mode="w+", dtype=np.float16, shape=(len(index), 8, 7)
    )
    features = None
    cursor = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(
        _batches(index, args.batch_size, transform, collator, action_stats, proprio_stats)
    ):
        _, hidden = _extract_action_hidden(vla, projector, batch, "cuda:0", gradients=False)
        hidden_np = hidden.reshape(hidden.shape[0], 8, -1).float().cpu().numpy().astype(np.float16)
        if features is None:
            features = np.lib.format.open_memmap(
                output / "features.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(index), 8, hidden_np.shape[-1]),
            )
        count = len(hidden_np)
        features[cursor : cursor + count] = hidden_np
        actions[cursor : cursor + count] = batch["actions"].numpy().astype(np.float16)
        cursor += count
        if batch_number % 25 == 0:
            print(f"dense retrieval: {cursor}/{len(index)}", flush=True)
    features.flush()
    actions.flush()
    report = {
        "status": "DENSE_RETRIEVAL_CACHE_COMPLETE",
        "source_cache": str(Path(args.cache).resolve()),
        "samples": len(index),
        "stride": args.stride,
        "task_classes": len(selected),
        "episodes": [
            {
                "task_class": episode["task_class"],
                "episode_index": episode["episode_index"],
                "instruction": episode["instruction"],
                "frames": episode["frames"],
            }
            for episode in selected
        ],
        "feature_shape": list(features.shape),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.time() - started,
        "counts_as_pure_policy_acceptance": False,
    }
    _write_json(output / "manifest.json", report)
    print((output / "manifest.json").resolve(), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="datasets/vla82_robocasa365_oft/cache_manifest.json")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--projector")
    parser.add_argument("--output", default="datasets/vla82_robocasa365_oft_retrieval_dense")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
