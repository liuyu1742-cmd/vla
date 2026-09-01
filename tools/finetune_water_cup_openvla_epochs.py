"""Train a water-cup OpenVLA LoRA for complete passes over local demonstrations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from tools.finetune_water_cup_openvla_full import INSTRUCTION, action_text
from tools.water_cup_action_codec import ACTION_NORM_KEY, ACTION_DIMENSION
from tools.water_cup_training_schedule import training_updates


ROOT = Path(__file__).resolve().parents[1]
ACTION_TOKEN_WEIGHTS = (3.0, 3.0, 3.0, 0.15, 0.15, 0.15, 2.0, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "openvla-water-cup-lora-e12",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_samples() -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    manifest = json.loads(
        (ROOT / "datasets" / "water_cup_expert" / "training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    paths = [ROOT / item["episode"] for item in manifest["train"]]
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for path in paths:
        with np.load(path) as episode:
            samples.extend(zip(episode["frames"], episode["actions"]))
    return samples, [path.name for path in paths]


def main() -> int:
    args = parse_args()
    samples, episodes = load_samples()
    planned_updates = training_updates(len(samples), args.batch_size, args.epochs)
    summary = {
        "train_episodes": episodes,
        "train_samples": len(samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "planned_updates": planned_updates,
        "action_norm_key": ACTION_NORM_KEY,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForVision2Seq, AutoProcessor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(
        str(ROOT / "models" / "openvla-7b"),
        trust_remote_code=True,
        local_files_only=True,
    )

    class Demonstrations(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            frame, action = samples[index]
            prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut: {action_text(processor.tokenizer, action)}</s>"
            ids = torch.tensor(
                processor.tokenizer(prompt, add_special_tokens=True).input_ids,
                dtype=torch.long,
            )
            labels = ids.clone()
            labels[: -(ACTION_DIMENSION + 1)] = -100
            return Image.fromarray(frame), ids, labels

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
        Demonstrations(), batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(ROOT / "models" / "openvla-7b"),
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
    for epoch in range(1, args.epochs + 1):
        for input_ids, labels, pixels in loader:
            output = model(
                input_ids=input_ids.cuda(),
                attention_mask=(input_ids != processor.tokenizer.pad_token_id).cuda(),
                pixel_values=pixels.cuda().to(torch.bfloat16),
            )
            logits = output.logits[:, :-1, :].float()
            targets = labels[:, 1:].cuda()
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(targets)
            weights = torch.zeros_like(token_loss)
            for row in range(targets.shape[0]):
                supervised = torch.nonzero(targets[row] != -100, as_tuple=False).flatten()
                weights[row, supervised] = token_weights[: supervised.numel()]
            loss = (token_loss * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            print(
                f"epoch={epoch}/{args.epochs} step={step}/{planned_updates} loss={loss_value:.5f}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    (args.output / "water_cup_action_stats.json").write_text(
        json.dumps(
            {
                "key": ACTION_NORM_KEY,
                "action": {
                    "q01": [-1.0] * ACTION_DIMENSION,
                    "q99": [1.0] * ACTION_DIMENSION,
                    "mask": [True] * ACTION_DIMENSION,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary["completed_updates"] = step
    summary["losses"] = losses
    (args.output / "training_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
