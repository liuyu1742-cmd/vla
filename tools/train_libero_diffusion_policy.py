"""Training data and checkpoint contracts for the compact LIBERO diffusion policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tools.libero_diffusion_contracts import (
    ActionNormalizer,
    WindowIndex,
    build_window_indices,
)
from tools.libero_diffusion_model import (
    CompactDiffusionPolicy,
    DiffusionPolicyConfig,
)


class FeatureWindowDataset(Dataset):
    """Lazily load cached episodes and return boundary-safe policy windows."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        normalizer: ActionNormalizer,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        episode_cache_size: int = 4,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.cache_root = self.manifest_path.parent
        self.normalizer = normalizer
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.episode_cache_size = max(1, int(episode_cache_size))
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.episodes = list(manifest["episodes"])
        lengths = [int(row["length"]) for row in self.episodes]
        self.windows = build_window_indices(
            lengths,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
        )
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index in self._episode_cache:
            episode = self._episode_cache.pop(episode_index)
            self._episode_cache[episode_index] = episode
            return episode

        path = self.cache_root / self.episodes[episode_index]["path"]
        with np.load(path, allow_pickle=False) as archive:
            episode = {name: archive[name] for name in archive.files}
        self._episode_cache[episode_index] = episode
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    @staticmethod
    def _left_repeat(array: np.ndarray, count: int) -> np.ndarray:
        if count == 0:
            return array
        return np.concatenate(
            [np.repeat(array[:1], count, axis=0), array],
            axis=0,
        )

    @staticmethod
    def _right_repeat(array: np.ndarray, count: int) -> np.ndarray:
        if count == 0:
            return array
        return np.concatenate(
            [array, np.repeat(array[-1:], count, axis=0)],
            axis=0,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        window: WindowIndex = self.windows[index]
        episode = self._load_episode(window.episode_index)
        visual = episode["features"][
            window.observation_start : window.observation_stop
        ]
        state = episode["states"][
            window.observation_start : window.observation_stop
        ]
        actions = episode["actions"][window.action_start : window.action_stop]
        visual = self._left_repeat(visual, window.observation_pad_left)
        state = self._left_repeat(state, window.observation_pad_left)
        actions = self._right_repeat(actions, window.action_pad_right)
        normalized_actions = self.normalizer.normalize(actions)
        return {
            "visual": torch.from_numpy(np.asarray(visual, dtype=np.float32)),
            "state": torch.from_numpy(np.asarray(state, dtype=np.float32)),
            "instruction": str(np.asarray(episode["instruction"]).item()),
            "actions": torch.from_numpy(normalized_actions),
            "episode_index": window.episode_index,
        }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    normalizer: ActionNormalizer,
    seed: int,
    data_manifest_hash: str,
    model_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically save all state needed for a strict training resume."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": {
            "q01": normalizer.q01.tolist(),
            "q99": normalizer.q99.tolist(),
        },
        "seed": int(seed),
        "data_manifest_hash": str(data_manifest_hash),
        "model_config": dict(model_config),
        "extra": dict(extra or {}),
    }
    torch.save(payload, temporary)
    with temporary.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_data_manifest_hash: str,
    expected_model_config: dict[str, Any],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint only when immutable data/model contracts match."""

    payload = torch.load(Path(path), map_location=map_location)
    if payload["data_manifest_hash"] != expected_data_manifest_hash:
        raise RuntimeError("checkpoint data manifest hash does not match")
    if payload["model_config"] != expected_model_config:
        raise RuntimeError("checkpoint model config does not match")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def render_training_process(metrics_path: Path, output_path: Path) -> dict[str, int]:
    """Render exactly the finite loss points present in a JSONL metric log."""

    rows: list[dict[str, Any]] = []
    for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        train_loss = float(row["train_loss"])
        validation_loss = float(row["validation_loss"])
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise RuntimeError("training metric losses must be finite")
        rows.append(
            {
                "step": int(row["step"]),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
    if not rows:
        raise RuntimeError("training metric log contains no points")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    steps = [row["step"] for row in rows]
    axis.plot(steps, [row["train_loss"] for row in rows], label="train loss")
    axis.plot(
        steps,
        [row["validation_loss"] for row in rows],
        label="validation loss",
    )
    axis.set_xlabel("Training step")
    axis.set_ylabel("Diffusion MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return {
        "points": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_normalizer(manifest_path: Path) -> tuple[ActionNormalizer, dict[str, int]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    action_rows: list[np.ndarray] = []
    cache_root = Path(manifest_path).parent
    state_dim = action_dim = visual_dim = 0
    for row in manifest["episodes"]:
        with np.load(cache_root / row["path"], allow_pickle=False) as archive:
            action_rows.append(np.asarray(archive["actions"], dtype=np.float32))
            visual_dim = int(archive["features"].shape[-1])
            state_dim = int(archive["states"].shape[-1])
            action_dim = int(archive["actions"].shape[-1])
    if not action_rows:
        raise RuntimeError("feature cache manifest contains no episodes")
    return ActionNormalizer.fit(np.concatenate(action_rows, axis=0)), {
        "visual_dim": visual_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }


def _condition_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "condition": {
            "visual": batch["visual"].to(device, non_blocking=True),
            "state": batch["state"].to(device, non_blocking=True),
            "instructions": list(batch["instruction"]),
        },
        "actions": batch["actions"].to(device, non_blocking=True),
    }


def _cycle(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def _validation_loss(
    model: CompactDiffusionPolicy,
    loader: DataLoader,
    *,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    batch = _condition_batch(next(iter(loader)), device)
    generator = torch.Generator(device=device.type).manual_seed(seed)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        loss = model.training_loss(batch, generator=generator)
    model.train()
    value = float(loss.detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError("validation loss is not finite")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the compact language-conditioned LIBERO diffusion policy."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="joint_20k_seed42")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--validation-every", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if min(
        args.steps,
        args.batch_size,
        args.checkpoint_every,
        args.validation_every,
    ) <= 0:
        parser.error("steps, batch size, checkpoint interval, and validation interval must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    experiment_dir = args.experiment_dir.resolve()
    manifest_path = experiment_dir / "feature_cache" / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing feature-cache manifest: {manifest_path}")
    manifest_hash = _sha256(manifest_path)
    normalizer, dimensions = _fit_normalizer(manifest_path)
    config = DiffusionPolicyConfig(**dimensions)
    model_config = asdict(config)
    dataset = FeatureWindowDataset(manifest_path, normalizer=normalizer)
    if len(dataset) < 2:
        raise RuntimeError("at least two feature windows are required for training")

    indices = np.arange(len(dataset))
    split_generator = np.random.default_rng(args.seed)
    split_generator.shuffle(indices)
    validation_count = max(1, min(len(dataset) - 1, int(round(0.1 * len(dataset)))))
    validation_indices = indices[:validation_count].tolist()
    training_indices = indices[validation_count:].tolist()
    loader_generator = torch.Generator().manual_seed(args.seed)
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    training_loader = DataLoader(
        Subset(dataset, training_indices),
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    model = CompactDiffusionPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_step = 0
    if args.resume is not None:
        payload = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_data_manifest_hash=manifest_hash,
            expected_model_config=model_config,
            map_location=device,
        )
        restored = ActionNormalizer(
            q01=np.asarray(payload["normalizer"]["q01"], dtype=np.float32),
            q99=np.asarray(payload["normalizer"]["q99"], dtype=np.float32),
        )
        if not (
            np.array_equal(restored.q01, normalizer.q01)
            and np.array_equal(restored.q99, normalizer.q99)
        ):
            raise RuntimeError("checkpoint action normalization stats do not match")
        start_step = int(payload["step"])

    run_dir = experiment_dir / "runs" / args.run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "train_metrics.jsonl"
    if metrics_path.exists() and start_step == 0:
        raise FileExistsError(
            f"run already contains metrics and will not be overwritten: {run_dir}"
        )
    started_at = datetime.now(timezone.utc)
    run_manifest = {
        "status": "running",
        "run_name": args.run_name,
        "started_at": started_at.isoformat(),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "feature_manifest": str(manifest_path),
        "feature_manifest_sha256": manifest_hash,
        "model_config": model_config,
        "normalizer": {
            "q01": normalizer.q01.tolist(),
            "q99": normalizer.q99.tolist(),
        },
        "git_available": False,
    }
    _atomic_json(run_dir / "run_manifest.json", run_manifest)

    iterator = _cycle(training_loader)
    model.train()
    best_loss = math.inf
    last_log_time = time.perf_counter()
    examples_since_log = 0
    try:
        for step in range(start_step + 1, args.steps + 1):
            batch = _condition_batch(next(iterator), device)
            examples_since_log += int(batch["actions"].shape[0])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                train_loss = model.training_loss(batch)
            if not torch.isfinite(train_loss):
                raise RuntimeError(f"training loss is not finite at step {step}")
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            should_validate = (
                step % args.validation_every == 0 or step == args.steps
            )
            validation_loss: float | None = None
            if should_validate:
                validation_loss = _validation_loss(
                    model,
                    validation_loader,
                    device=device,
                    seed=args.seed + step,
                )
                elapsed = max(time.perf_counter() - last_log_time, 1e-9)
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "train_loss": float(train_loss.detach().cpu()),
                    "validation_loss": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "examples_per_second": examples_since_log / elapsed,
                    "gpu": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else None
                    ),
                    "allocated_vram_bytes": (
                        torch.cuda.memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    ),
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    ),
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                last_log_time = time.perf_counter()
                examples_since_log = 0

            should_checkpoint = (
                step % args.checkpoint_every == 0 or step == args.steps
            )
            if should_checkpoint:
                save_checkpoint(
                    checkpoints_dir / f"step_{step}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    normalizer=normalizer,
                    seed=args.seed,
                    data_manifest_hash=manifest_hash,
                    model_config=model_config,
                    extra={"validation_loss": validation_loss},
                )
            if validation_loss is not None and validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    normalizer=normalizer,
                    seed=args.seed,
                    data_manifest_hash=manifest_hash,
                    model_config=model_config,
                    extra={"validation_loss": validation_loss},
                )
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "CUDA out of memory; rerun this unchanged configuration with --batch-size 128"
        ) from error

    render_training_process(metrics_path, run_dir / "training_process.png")
    best_target = checkpoints_dir / "best.pt"
    if best_target.is_file():
        shutil.copy2(best_target, run_dir / "best.pt")
    run_manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": (
                datetime.now(timezone.utc) - started_at
            ).total_seconds(),
            "best_validation_loss": best_loss,
            "metrics": str(metrics_path),
            "best_checkpoint": str(run_dir / "best.pt"),
        }
    )
    _atomic_json(run_dir / "run_manifest.json", run_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
