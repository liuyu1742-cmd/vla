"""Train the OFT L1 action head from frozen multimodal feature arrays."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from tools.train_robocasa_oft_head import (
    DEFAULT_MODEL,
    OFT_ROOT,
    _component_states,
    _install_windows_resource_stub,
    _strip_module_prefix,
    _write_json,
)


def _objective(torch, prediction, target, kind):
    if kind == "mse":
        return torch.nn.functional.mse_loss(prediction, target)
    if kind == "l1":
        return torch.nn.functional.l1_loss(prediction, target)
    raise ValueError(f"unsupported loss: {kind}")


def load_extra_feature_arrays(roots, expected_feature_shape):
    """Load and concatenate one or more compatible targeted feature caches."""
    feature_parts, action_parts = [], []
    for root_value in roots:
        root = Path(root_value).resolve()
        features = np.load(root / "features.npy", mmap_mode="r")
        actions = np.load(root / "actions.npy", mmap_mode="r")
        if features.shape[:2] != actions.shape[:2]:
            raise ValueError(f"extra feature/action arrays are not aligned: {root}")
        if features.shape[1:] != tuple(expected_feature_shape):
            raise ValueError(f"extra feature shape is incompatible: {root}")
        feature_parts.append(np.asarray(features))
        action_parts.append(np.asarray(actions))
    return np.concatenate(feature_parts, axis=0), np.concatenate(action_parts, axis=0)


def _evaluate(torch, head, features, actions, indices, batch_size, device, loss_kind):
    losses, l1_losses, predictions = [], [], []
    head.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(indices), batch_size):
            ids = indices[start : start + batch_size]
            x = torch.from_numpy(np.asarray(features[ids])).to(device=device, dtype=torch.bfloat16)
            y = torch.from_numpy(np.asarray(actions[ids])).to(device=device, dtype=torch.bfloat16)
            pred = head.predict_action(x.reshape(len(ids), 56, -1))
            losses.append(float(_objective(torch, pred, y, loss_kind).cpu()))
            l1_losses.append(float(torch.nn.functional.l1_loss(pred, y).cpu()))
            predictions.append(pred.float().cpu().numpy())
    return float(np.mean(losses)), float(np.mean(l1_losses)), np.concatenate(predictions)


def run(args: argparse.Namespace) -> dict:
    import torch

    _install_windows_resource_stub()
    if str(OFT_ROOT) not in sys.path:
        sys.path.insert(0, str(OFT_ROOT))
    from prismatic.models.action_heads import L1RegressionActionHead

    feature_root = Path(args.features).resolve()
    manifest = json.loads((feature_root / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    train_x = np.load(feature_root / "train_features.npy", mmap_mode="r")
    train_y = np.load(feature_root / "train_actions.npy", mmap_mode="r")
    val_x = np.load(feature_root / "val_features.npy", mmap_mode="r")
    val_y = np.load(feature_root / "val_actions.npy", mmap_mode="r")
    if train_x.shape[:2] != train_y.shape[:2] or val_x.shape[:2] != val_y.shape[:2]:
        raise ValueError("feature/action arrays are not aligned")
    base_train_samples = len(train_x)
    targeted_samples = 0
    targeted_val_x = targeted_val_y = None
    if args.extra_features:
        extra_x, extra_y = load_extra_feature_arrays(
            args.extra_features, expected_feature_shape=train_x.shape[1:]
        )
        targeted_samples = len(extra_x)
        # Keep the reset frame in training: it is the critical first policy
        # decision.  Hold out a different temporal phase from each group of 5.
        targeted_val_mask = np.arange(targeted_samples) % 5 == 4
        targeted_val_x = np.asarray(extra_x[targeted_val_mask])
        targeted_val_y = np.asarray(extra_y[targeted_val_mask])
        targeted_train_x = np.asarray(extra_x[~targeted_val_mask])
        targeted_train_y = np.asarray(extra_y[~targeted_val_mask])
        feature_parts = [np.asarray(train_x)] * args.base_weight
        action_parts = [np.asarray(train_y)] * args.base_weight
        feature_parts += [targeted_train_x] * args.extra_weight
        action_parts += [targeted_train_y] * args.extra_weight
        if not feature_parts:
            raise ValueError("training requires base or targeted samples")
        train_x = np.concatenate(feature_parts, axis=0)
        train_y = np.concatenate(action_parts, axis=0)
        if args.extra_motion_weight > 1:
            high_motion = np.max(np.abs(targeted_train_y[:, 0, :3]), axis=1) >= 0.75
            motion_x = targeted_train_x[high_motion]
            motion_y = targeted_train_y[high_motion]
            train_x = np.concatenate(
                [train_x] + [motion_x] * (args.extra_motion_weight - 1), axis=0
            )
            train_y = np.concatenate(
                [train_y] + [motion_y] * (args.extra_motion_weight - 1), axis=0
            )
    state, _ = _component_states(Path(args.model).resolve())
    if args.action_head:
        state = _strip_module_prefix(torch.load(args.action_head, map_location="cpu"))
    head = L1RegressionActionHead(input_dim=4096, hidden_dim=4096, action_dim=7).to(
        device="cuda:0", dtype=torch.bfloat16
    )
    head.load_state_dict(state)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    train_ids = list(range(len(train_x)))
    val_ids = list(range(len(val_x)))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    base_objective, base_l1, baseline_predictions = _evaluate(
        torch, head, val_x, val_y, val_ids, args.batch_size, "cuda:0", args.loss
    )
    targeted_objective = targeted_l1 = None
    if targeted_val_x is not None:
        targeted_objective, targeted_l1, targeted_predictions = _evaluate(
            torch,
            head,
            targeted_val_x,
            targeted_val_y,
            list(range(len(targeted_val_x))),
            args.batch_size,
            "cuda:0",
            args.loss,
        )
        baseline_predictions = np.concatenate([baseline_predictions, targeted_predictions])
    baseline = (
        targeted_objective
        if args.selection == "targeted" and targeted_objective is not None
        else (
            0.5 * (base_objective + targeted_objective)
            if targeted_objective is not None
            else base_objective
        )
    )
    baseline_l1 = (
        0.5 * (base_l1 + targeted_l1) if targeted_l1 is not None else base_l1
    )
    best, misses, history = baseline, 0, []
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    for epoch in range(args.epochs):
        rng.shuffle(train_ids)
        head.train()
        losses = []
        for start in range(0, len(train_ids), args.batch_size):
            ids = train_ids[start : start + args.batch_size]
            x = torch.from_numpy(np.asarray(train_x[ids])).to("cuda:0", dtype=torch.bfloat16)
            y = torch.from_numpy(np.asarray(train_y[ids])).to("cuda:0", dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head.predict_action(x.reshape(len(ids), 56, -1))
                loss = _objective(torch, pred, y, args.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        base_validation, base_validation_l1, predictions = _evaluate(
            torch, head, val_x, val_y, val_ids, args.batch_size, "cuda:0", args.loss
        )
        targeted_validation = targeted_validation_l1 = None
        if targeted_val_x is not None:
            targeted_validation, targeted_validation_l1, targeted_predictions = _evaluate(
                torch,
                head,
                targeted_val_x,
                targeted_val_y,
                list(range(len(targeted_val_x))),
                args.batch_size,
                "cuda:0",
                args.loss,
            )
            predictions = np.concatenate([predictions, targeted_predictions])
        validation = (
            targeted_validation
            if args.selection == "targeted" and targeted_validation is not None
            else (
                0.5 * (base_validation + targeted_validation)
                if targeted_validation is not None
                else base_validation
            )
        )
        validation_l1 = (
            0.5 * (base_validation_l1 + targeted_validation_l1)
            if targeted_validation_l1 is not None
            else base_validation_l1
        )
        record = {
            "epoch": epoch + 1,
            "train_objective": float(np.mean(losses)),
            "validation_objective": validation,
            "validation_l1": validation_l1,
            "base_validation_objective": base_validation,
            "base_validation_l1": base_validation_l1,
            "targeted_validation_objective": targeted_validation,
            "targeted_validation_l1": targeted_validation_l1,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation < best - args.minimum_improvement:
            best, misses = validation, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            misses += 1
            if misses >= args.patience:
                break
    head.load_state_dict(best_state)
    torch.save(head.state_dict(), output / "action_head.pt")
    projector_source = Path(args.projector).resolve()
    projector_state = _strip_module_prefix(torch.load(projector_source, map_location="cpu"))
    torch.save(projector_state, output / "proprio_projector.pt")
    stats = {"dataset_key": "robocasa365_oft", "action": manifest["action_stats"], "proprio": manifest["proprio_stats"]}
    _write_json(output / "robocasa_oft_stats.json", stats)
    final_predictions = predictions if history else baseline_predictions
    report = {
        "status": "TRAINING_COMPLETE",
        "feature_cache": str(feature_root),
        "loss": args.loss,
        "selection": args.selection,
        "initial_validation_objective": baseline,
        "initial_validation_l1": baseline_l1,
        "best_validation_objective": best,
        "final_validation_l1": history[-1]["validation_l1"] if history else baseline_l1,
        "improved": bool(best < baseline),
        "prediction_variance": np.var(final_predictions, axis=0).tolist(),
        "finite_predictions": bool(np.isfinite(final_predictions).all()),
        "train_samples": len(train_x),
        "base_train_samples": base_train_samples,
        "base_repeat_weight": args.base_weight,
        "targeted_unique_samples": targeted_samples,
        "targeted_validation_samples": (
            len(targeted_val_x) if targeted_val_x is not None else 0
        ),
        "targeted_repeat_weight": args.extra_weight if args.extra_features else 0,
        "targeted_motion_weight": (
            args.extra_motion_weight if args.extra_features else 0
        ),
        "validation_samples": len(val_x),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "arguments": vars(args),
    }
    _write_json(output / "training_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="datasets/vla82_robocasa365_oft_features")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--action-head")
    parser.add_argument("--projector", required=True)
    parser.add_argument("--extra-features", action="append")
    parser.add_argument("--base-weight", type=int, default=1)
    parser.add_argument("--extra-weight", type=int, default=4)
    parser.add_argument("--extra-motion-weight", type=int, default=1)
    parser.add_argument("--output", default="models/openvla-oft-robocasa365-head-r1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--minimum-improvement", type=float, default=0.002)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--loss", choices=("l1", "mse"), default="mse")
    parser.add_argument("--selection", choices=("composite", "targeted"), default="composite")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=82)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
