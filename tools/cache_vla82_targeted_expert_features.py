"""Encode one targeted simulator expert episode with the frozen OFT backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.robocasa_oft_contract import future_action_chunk
from tools.robocasa_oft_model import normalize
from tools.train_robocasa_oft_head import (
    DEFAULT_MODEL,
    _build_transform,
    _component_states,
    _extract_action_hidden,
    _load_model,
    _strip_module_prefix,
    _write_json,
)


def selected_indices(length: int, stride: int, max_samples: int | None) -> list[int]:
    """Return strided indices after limiting the usable source prefix."""
    if stride < 1:
        raise ValueError("stride must be positive")
    usable = length if max_samples is None else min(length, max_samples)
    return list(range(0, usable, stride))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Use only this many leading source frames before applying stride.",
    )
    args = parser.parse_args()
    import torch
    from tools.robocasa_oft_model import build_trainable_components

    episode = np.load(args.episode)
    primary = episode["primary"]
    wrist = episode["wrist"]
    proprio = episode["proprio"]
    raw_actions = episode["actions"]
    instruction = str(episode["instruction"].item())
    if not (len(primary) == len(wrist) == len(proprio) == len(raw_actions)):
        raise ValueError("targeted expert arrays are not aligned")
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    action_stats, proprio_stats = stats["action"], stats["proprio"]
    chunks = np.stack(
        [future_action_chunk(raw_actions, index) for index in range(len(raw_actions))]
    )
    indices = selected_indices(len(raw_actions), args.stride, args.max_samples)
    processor, vla = _load_model(args.model.resolve(), "cuda:0")
    action_state, _ = _component_states(args.model.resolve())
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
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    features = None
    target_actions = np.lib.format.open_memmap(
        output / "actions.npy", mode="w+", dtype=np.float16, shape=(len(indices), 8, 7)
    )
    cursor = 0
    for start in range(0, len(indices), args.batch_size):
        selected = indices[start : start + args.batch_size]
        instances = []
        for index in selected:
            rlds = {
                "dataset_name": b"vla82_targeted_expert",
                "action": normalize(chunks[index], action_stats),
                "observation": {
                    "image_primary": primary[index][None],
                    "image_wrist": wrist[index][None],
                    "proprio": normalize(proprio[index], proprio_stats)[None],
                },
                "task": {"language_instruction": instruction.encode("utf-8")},
            }
            instances.append(transform(rlds))
        batch = collator(instances)
        if batch["proprio"].ndim == 1:
            batch["proprio"] = batch["proprio"].unsqueeze(0)
        _, hidden = _extract_action_hidden(vla, projector, batch, "cuda:0", gradients=False)
        hidden_np = hidden.reshape(len(selected), 8, -1).float().cpu().numpy().astype(np.float16)
        if features is None:
            features = np.lib.format.open_memmap(
                output / "features.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(indices), 8, hidden_np.shape[-1]),
            )
        features[cursor : cursor + len(selected)] = hidden_np
        target_actions[cursor : cursor + len(selected)] = batch["actions"].numpy().astype(np.float16)
        cursor += len(selected)
        if cursor % 40 < args.batch_size:
            print(f"targeted features: {cursor}/{len(indices)}", flush=True)
    features.flush()
    target_actions.flush()
    report = {
        "status": "TARGETED_FEATURE_CACHE_COMPLETE",
        "source_episode": str(args.episode.resolve()),
        "source_expert_success_required": True,
        "samples": len(indices),
        "source_prefix_samples": (
            min(len(raw_actions), args.max_samples)
            if args.max_samples is not None
            else len(raw_actions)
        ),
        "stride": args.stride,
        "feature_shape": list(features.shape),
        "features": str((output / "features.npy").resolve()),
        "actions": str((output / "actions.npy").resolve()),
        "instruction": instruction,
        "counts_as_pure_policy_acceptance": False,
    }
    _write_json(output / "manifest.json", report)
    print((output / "manifest.json").resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
