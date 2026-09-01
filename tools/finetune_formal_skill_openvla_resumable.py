"""Checkpointed, automatically resumable formal OpenVLA LoRA training."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.finetune_formal_skill_openvla import (
    ACTION_DIMENSION,
    ACTION_NORM_KEY,
    ACTION_TOKEN_WEIGHTS,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    action_statistics,
    action_text,
    build_instruction,
    load_training_actions,
    normalize_action,
    planned_training_summary,
)
from tools.formal_skill_dataset import FormalSkillDataset


CHECKPOINT_SCHEMA = "formal_skill_training_checkpoint_v1"


def resume_position(completed_updates: int, updates_per_epoch: int) -> tuple[int, int]:
    if completed_updates < 0 or updates_per_epoch < 1:
        raise ValueError("completed updates must be non-negative and epoch size positive")
    return divmod(completed_updates, updates_per_epoch)


def find_latest_checkpoint(
    checkpoint_root: Path, *, manifest_sha256: str, planned_updates: int
) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for directory in Path(checkpoint_root).glob("checkpoint_step_*"):
        if not directory.is_dir():
            continue
        state_path = directory / "checkpoint_state.json"
        if not all(
            path.is_file()
            for path in (
                state_path,
                directory / "adapter_config.json",
                directory / "optimizer.pt",
            )
        ):
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            completed = int(state["completed_updates"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if state.get("schema_version") != CHECKPOINT_SCHEMA:
            continue
        if state.get("manifest_sha256") != manifest_sha256:
            continue
        if int(state.get("planned_updates", -1)) != planned_updates:
            continue
        if not 0 < completed <= planned_updates:
            continue
        candidates.append((completed, directory))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.log_every < 1 or args.checkpoint_every < 1:
        raise ValueError("logging and checkpoint intervals must be positive")
    summary = planned_training_summary(
        args.manifest, epochs=args.epochs, batch_size=args.batch_size
    )
    dataset = FormalSkillDataset(args.manifest, split="train")
    if len(dataset) != summary["train_samples"]:
        raise RuntimeError("dataset length differs from audited manifest")
    statistics = action_statistics(load_training_actions(dataset))
    summary.update(
        {
            "trainer": "checkpointed_resumable_v1",
            "model_dir": str(Path(args.model_dir).resolve()),
            "output": str(Path(args.output).resolve()),
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "action_norm_key": ACTION_NORM_KEY,
            "action_statistics": statistics,
            "checkpoint_every": args.checkpoint_every,
        }
    )
    checkpoint_root = args.output / "checkpoints"
    checkpoint = None
    if not args.no_resume:
        checkpoint = find_latest_checkpoint(
            checkpoint_root,
            manifest_sha256=summary["manifest_sha256"],
            planned_updates=summary["planned_updates"],
        )
    summary["resume_checkpoint"] = str(checkpoint.resolve()) if checkpoint else None
    if args.dry_run:
        summary["dry_run"] = True
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader, Dataset, Subset
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

    base = AutoModelForVision2Seq.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda")
    if checkpoint is None:
        model = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
        completed = 0
        loss_sum = 0.0
        loss_count = 0
        loss_tail: list[float] = []
    else:
        model = PeftModel.from_pretrained(
            base, str(checkpoint), is_trainable=True, local_files_only=True
        )
        state = json.loads(
            (checkpoint / "checkpoint_state.json").read_text(encoding="utf-8")
        )
        completed = int(state["completed_updates"])
        loss_sum = float(state.get("loss_sum", 0.0))
        loss_count = int(state.get("loss_count", completed))
        loss_tail = [float(value) for value in state.get("loss_tail", [])][-100:]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", map_location="cpu")
        )
    token_weights = torch.tensor(ACTION_TOKEN_WEIGHTS, device="cuda")
    updates_per_epoch = math.ceil(len(dataset) / args.batch_size)
    epoch_index, update_offset = resume_position(completed, updates_per_epoch)
    if epoch_index > args.epochs or (
        epoch_index == args.epochs and update_offset != 0
    ):
        raise RuntimeError("checkpoint is beyond the requested training schedule")
    summary.update(
        {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "lora": {"rank": 16, "alpha": 32, "target_modules": "all-linear"},
            "resumed_from_update": completed,
        }
    )
    print(
        f"TRAINING_RESUME checkpoint={checkpoint} completed={completed} "
        f"planned={summary['planned_updates']}",
        flush=True,
    )

    def save_checkpoint(step: int) -> Path:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        destination = checkpoint_root / f"checkpoint_step_{step:06d}"
        if destination.exists():
            destination = checkpoint_root / (
                f"checkpoint_step_{step:06d}_retry_{time.time_ns()}"
            )
        destination.mkdir(parents=True)
        model.save_pretrained(destination)
        torch.save(optimizer.state_dict(), destination / "optimizer.pt")
        _atomic_json(
            destination / "checkpoint_state.json",
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "completed_updates": step,
                "planned_updates": summary["planned_updates"],
                "manifest_sha256": summary["manifest_sha256"],
                "skill_ir_sha256": summary["skill_ir_sha256"],
                "loss_sum": loss_sum,
                "loss_count": loss_count,
                "loss_tail": loss_tail[-100:],
            },
        )
        print(f"CHECKPOINT_SAVED step={step} path={destination}", flush=True)
        return destination

    demonstrations = Demonstrations()
    model.train()
    for epoch in range(epoch_index, args.epochs):
        first_update = update_offset if epoch == epoch_index else 0
        sample_start = min(first_update * args.batch_size, len(dataset))
        epoch_data = Subset(demonstrations, range(sample_start, len(dataset)))
        loader = DataLoader(
            epoch_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
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
            completed += 1
            value = float(loss.detach().cpu())
            loss_sum += value
            loss_count += 1
            loss_tail.append(value)
            del loss_tail[:-100]
            if (
                completed == 1
                or completed % args.log_every == 0
                or completed == summary["planned_updates"]
            ):
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={completed}/{summary['planned_updates']} loss={value:.5f}",
                    flush=True,
                )
            if completed % args.checkpoint_every == 0:
                save_checkpoint(completed)
        update_offset = 0

    if completed != summary["planned_updates"]:
        raise RuntimeError(
            f"training ended at {completed}, expected {summary['planned_updates']} updates"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    _atomic_json(
        args.output / "formal_skill_action_stats.json",
        {"key": ACTION_NORM_KEY, "action": statistics},
    )
    summary.update(
        {
            "completed_updates": completed,
            "final_loss": loss_tail[-1] if loss_tail else None,
            "mean_loss": loss_sum / loss_count if loss_count else None,
            "loss_count": loss_count,
            "loss_tail": loss_tail,
        }
    )
    _atomic_json(args.output / "training_report.json", summary)
    print(str((args.output / "training_report.json").resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
