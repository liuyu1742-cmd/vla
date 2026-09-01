"""LoRA fine-tune OpenVLA on an audited formal Skill-IR manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.formal_skill_dataset import FormalSkillDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "datasets" / "formal_skills" / "organizing_toy" / "training_manifest.json"
)
DEFAULT_MODEL = ROOT / "models" / "openvla-7b"
DEFAULT_OUTPUT = ROOT / "models" / "openvla-organizing-toy-lora"
ACTION_DIMENSION = 7
ACTION_NORM_KEY = "task2_organizing_toy"
ACTION_TOKEN_WEIGHTS = (3.0, 3.0, 3.0, 0.15, 0.15, 0.15, 2.0, 0.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planned_training_summary(
    manifest_path: Path, *, epochs: int, batch_size: int
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema_version") != "formal_skill_training_manifest_v1":
        raise ValueError("unsupported formal skill manifest schema")
    if manifest.get("relation_key") != "organizing::toy":
        raise ValueError("formal trainer only accepts organizing::toy")
    train = manifest.get("train")
    held_out = manifest.get("held_out")
    if not isinstance(train, list) or not isinstance(held_out, list):
        raise ValueError("manifest train and held_out must be lists")
    train_seeds = [int(item["seed"]) for item in train]
    heldout_seeds = [int(item["seed"]) for item in held_out]
    overlap = set(train_seeds) & set(heldout_seeds)
    if overlap:
        raise ValueError(f"train and held-out seeds overlap: {sorted(overlap)}")
    samples = int(manifest["training_sample_count"])
    if samples < 1:
        raise ValueError("training manifest has no samples")
    if int(manifest["training_episode_count"]) != len(train):
        raise ValueError("training episode count differs from train list")
    if int(manifest["heldout_episode_count"]) != len(held_out):
        raise ValueError("held-out episode count differs from held_out list")
    return {
        "schema_version": "formal_skill_openvla_training_v1",
        "relation_key": "organizing::toy",
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "skill_ir_sha256": str(manifest["skill_ir_sha256"]),
        "train_seeds": train_seeds,
        "heldout_seeds": heldout_seeds,
        "train_episodes": len(train),
        "heldout_episodes": len(held_out),
        "train_samples": samples,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "planned_updates": math.ceil(samples / batch_size) * epochs,
        "data_order": "episode-contiguous; deterministic per epoch",
        "loss": "visual-token-aligned weighted causal cross entropy",
    }


def action_statistics(
    actions: np.ndarray, *, lower: float = 0.01, upper: float = 0.99
) -> dict[str, list[Any]]:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIMENSION:
        raise ValueError(f"actions must have shape (N, {ACTION_DIMENSION})")
    if values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("actions must be non-empty and finite")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    q01 = np.quantile(values, lower, axis=0).astype(np.float32)
    q99 = np.quantile(values, upper, axis=0).astype(np.float32)
    mask = (q99 - q01) > 1e-6
    return {
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": mask.tolist(),
    }


def normalize_action(action: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (ACTION_DIMENSION,):
        raise ValueError(f"action must have shape ({ACTION_DIMENSION},)")
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    mask = np.asarray(statistics["mask"], dtype=bool)
    if low.shape != values.shape or high.shape != values.shape or mask.shape != values.shape:
        raise ValueError("action statistics must have seven entries")
    result = values.copy()
    result[mask] = 2.0 * (values[mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    result[mask] = np.clip(result[mask], -1.0, 1.0)
    result[~mask] = 0.0
    return result


def build_instruction(instruction: str, canonical_phase: str) -> str:
    task = " ".join(str(instruction).strip().split())
    phase = " ".join(str(canonical_phase).strip().lower().split())
    if not task or not phase:
        raise ValueError("instruction and canonical phase must be non-empty")
    return f"{task}; the current skill phase is {phase}"


def action_text(tokenizer: Any, normalized_action: np.ndarray) -> str:
    action = np.asarray(normalized_action, dtype=np.float32)
    if action.shape != (ACTION_DIMENSION,):
        raise ValueError("OpenVLA action must have seven dimensions")
    bins = np.linspace(-1.0, 1.0, 256)
    token_ids = tokenizer.vocab_size - np.digitize(np.clip(action, -1.0, 1.0), bins)
    return tokenizer.decode(token_ids.tolist())


def load_training_actions(dataset: FormalSkillDataset) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for entry in dataset.episodes:
        with np.load(entry["episode"], allow_pickle=False) as episode:
            chunk = np.asarray(episode["actions"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIMENSION:
            raise ValueError(f"invalid action matrix in {entry['episode']}: {chunk.shape}")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("formal training split is empty")
    return np.concatenate(chunks, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    summary = planned_training_summary(
        args.manifest, epochs=args.epochs, batch_size=args.batch_size
    )
    dataset = FormalSkillDataset(args.manifest, split="train")
    if len(dataset) != summary["train_samples"]:
        raise RuntimeError("dataset length differs from audited manifest")
    statistics = action_statistics(load_training_actions(dataset))
    norm_stats = {"key": ACTION_NORM_KEY, "action": statistics}
    summary.update(
        {
            "model_dir": str(Path(args.model_dir).resolve()),
            "output": str(Path(args.output).resolve()),
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "action_norm_key": ACTION_NORM_KEY,
            "action_statistics": statistics,
        }
    )
    if args.dry_run:
        summary["dry_run"] = True
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from tools.water_cup_multimodal_labels import expand_labels_for_visual_tokens

    if not torch.cuda.is_available():
        raise RuntimeError("formal OpenVLA fine-tuning requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True
    )

    class Demonstrations(Dataset):
        def __len__(self) -> int:
            return len(dataset)

        def __getitem__(self, index: int):
            sample = dataset[index]
            normalized = normalize_action(sample["action"], statistics)
            conditioned = build_instruction(
                sample["instruction"], sample["canonical_phase"]
            )
            prompt = (
                f"In: What action should the robot take to {conditioned}?\n"
                f"Out: {action_text(processor.tokenizer, normalized)}</s>"
            )
            ids = torch.tensor(
                processor.tokenizer(prompt, add_special_tokens=True).input_ids,
                dtype=torch.long,
            )
            labels = ids.clone()
            labels[: -(ACTION_DIMENSION + 1)] = -100
            return Image.fromarray(sample["frame"]), ids, labels

    def collate(batch):
        images, inputs, labels = zip(*batch)
        length = max(item.numel() for item in inputs)
        pad = processor.tokenizer.pad_token_id
        input_ids = torch.full((len(batch), length), pad, dtype=torch.long)
        target_ids = torch.full((len(batch), length), -100, dtype=torch.long)
        for index, (item, target) in enumerate(zip(inputs, labels)):
            input_ids[index, : item.numel()] = item
            target_ids[index, : target.numel()] = target
        pixels = torch.stack(
            [processor.image_processor.apply_transform(image) for image in images]
        )
        return input_ids, target_ids, pixels

    loader = DataLoader(
        Demonstrations(),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    token_weights = torch.tensor(ACTION_TOKEN_WEIGHTS, device="cuda")
    losses: list[float] = []
    step = 0
    model.train()
    summary.update(
        {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "lora": {"rank": 16, "alpha": 32, "target_modules": "all-linear"},
        }
    )
    for epoch in range(1, args.epochs + 1):
        for input_ids, labels, pixels in loader:
            output = model(
                input_ids=input_ids.cuda(),
                attention_mask=(input_ids != processor.tokenizer.pad_token_id).cuda(),
                pixel_values=pixels.cuda().to(torch.bfloat16),
            )
            expanded_labels = expand_labels_for_visual_tokens(
                labels.cuda(), logits_length=output.logits.shape[1]
            )
            logits = output.logits[:, :-1, :].float()
            targets = expanded_labels[:, 1:]
            token_loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(targets)
            weights = torch.zeros_like(token_loss)
            for row in range(targets.shape[0]):
                supervised = torch.nonzero(
                    targets[row] != -100, as_tuple=False
                ).flatten()
                if supervised.numel() != len(ACTION_TOKEN_WEIGHTS):
                    raise RuntimeError(
                        f"expected 8 supervised tokens, got {supervised.numel()}"
                    )
                weights[row, supervised] = token_weights
            loss = (token_loss * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            value = float(loss.detach().cpu())
            losses.append(value)
            if step == 1 or step % args.log_every == 0 or step == summary["planned_updates"]:
                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"step={step}/{summary['planned_updates']} loss={value:.5f}",
                    flush=True,
                )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    (args.output / "formal_skill_action_stats.json").write_text(
        json.dumps(norm_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary["completed_updates"] = step
    summary["final_loss"] = losses[-1] if losses else None
    summary["mean_loss"] = float(np.mean(losses)) if losses else None
    summary["losses"] = losses
    (args.output / "training_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(str((args.output / "training_report.json").resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
