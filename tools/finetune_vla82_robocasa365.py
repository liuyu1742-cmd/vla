"""LoRA fine-tune OpenVLA on the balanced 19-class RoboCasa365 cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.finetune_formal_skill_openvla import (
    action_statistics,
    action_text,
    normalize_action,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    ROOT / "datasets" / "vla82_robocasa365_19class" / "training_cache_stride_8.json"
)
DEFAULT_MODEL = ROOT / "models" / "openvla-7b"
DEFAULT_OUTPUT = ROOT / "models" / "openvla-vla82-robocasa365-lora-r1"
ACTION_NORM_KEY = "vla82_robocasa365_19class"


def balanced_sample_weights(cells: Sequence[tuple[str, str]]) -> list[float]:
    """Inverse-frequency weights across both task classes and action phases."""
    if not cells:
        raise ValueError("sampling cells must not be empty")
    task_counts = Counter(task for task, _ in cells)
    cell_counts = Counter(cells)
    return [
        1.0 / (task_counts[task] * cell_counts[(task, phase)])
        for task, phase in cells
    ]


def action_phase(action: Sequence[float]) -> str:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (7,):
        raise ValueError("action must contain seven channels")
    if float(np.linalg.norm(values[:6])) >= 0.25:
        return "moving"
    return "gripper_closed" if float(values[6]) < 0 else "gripper_open"


def planned_training_summary(cache_path: Path, *, epochs: int) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "vla82_robocasa365_cache_v1":
        raise ValueError("unsupported cache schema")
    samples = int(payload["sample_count"])
    return {
        "task_class_count": int(payload["task_class_count"]),
        "episode_count": int(payload["episode_count"]),
        "sample_count": samples,
        "epochs": epochs,
        "planned_updates": samples * epochs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = planned_training_summary(args.cache, epochs=args.epochs)
    payload = json.loads(args.cache.read_text(encoding="utf-8"))
    all_actions = np.concatenate(
        [np.load(item["cache"], allow_pickle=False)["actions"] for item in payload["episodes"]]
    )
    statistics = action_statistics(all_actions)
    summary.update(
        {
            "cache": str(args.cache.resolve()),
            "model_dir": str(args.model_dir.resolve()),
            "output": str(args.output.resolve()),
            "learning_rate": args.learning_rate,
            "action_norm_key": ACTION_NORM_KEY,
            "action_statistics": statistics,
        }
    )
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from tools.water_cup_multimodal_labels import expand_labels_for_visual_tokens

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(82)
    np.random.seed(82)
    torch.manual_seed(82)
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True
    )

    class Demonstrations(Dataset):
        def __init__(self) -> None:
            self.index = []
            self.sampling_cells = []
            self.loaded_index = -1
            self.loaded = None
            for episode_index, episode in enumerate(payload["episodes"]):
                actions = np.load(episode["cache"], allow_pickle=False)["actions"]
                for frame_index, action in enumerate(actions):
                    self.index.append((episode_index, frame_index))
                    self.sampling_cells.append(
                        (str(episode["task_class"]), action_phase(action))
                    )

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, index: int):
            episode_index, frame_index = self.index[index]
            if self.loaded_index != episode_index:
                self.loaded = np.load(
                    payload["episodes"][episode_index]["cache"], allow_pickle=False
                )
                self.loaded_index = episode_index
            episode = payload["episodes"][episode_index]
            action = normalize_action(self.loaded["actions"][frame_index], statistics)
            prompt = (
                "In: What action should the robot take to "
                f"{episode['instruction']}\nOut: "
                f"{action_text(processor.tokenizer, action)}</s>"
            )
            ids = torch.tensor(
                processor.tokenizer(prompt, add_special_tokens=True).input_ids,
                dtype=torch.long,
            )
            labels = ids.clone()
            labels[: -8] = -100
            return Image.fromarray(self.loaded["frames"][frame_index]), ids, labels

    dataset = Demonstrations()

    def collate(batch):
        images, inputs, labels = zip(*batch)
        length = max(item.numel() for item in inputs)
        pad = processor.tokenizer.pad_token_id
        input_ids = torch.full((len(batch), length), pad, dtype=torch.long)
        target_ids = torch.full((len(batch), length), -100, dtype=torch.long)
        for row, (item, target) in enumerate(zip(inputs, labels)):
            input_ids[row, : item.numel()] = item
            target_ids[row, : target.numel()] = target
        pixels = torch.stack(
            [processor.image_processor.apply_transform(image) for image in images]
        )
        return input_ids, target_ids, pixels

    model = AutoModelForVision2Seq.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    sampler = WeightedRandomSampler(
        balanced_sample_weights(dataset.sampling_cells),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(82),
    )
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate)
    losses = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for input_ids, labels, pixels in loader:
            output = model(
                input_ids=input_ids.cuda(),
                attention_mask=(input_ids != processor.tokenizer.pad_token_id).cuda(),
                pixel_values=pixels.cuda().to(torch.bfloat16),
            )
            targets = expand_labels_for_visual_tokens(
                labels.cuda(), logits_length=output.logits.shape[1]
            )[:, 1:]
            logits = output.logits[:, :-1, :].float()
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{summary['planned_updates']} loss={losses[-1]:.5f}",
                    flush=True,
                )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    (args.output / "vla82_action_stats.json").write_text(
        json.dumps(
            {"key": ACTION_NORM_KEY, "action": statistics},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary.update(
        {
            "completed_updates": step,
            "final_loss": losses[-1],
            "mean_loss": float(np.mean(losses)),
            "losses": losses,
        }
    )
    (args.output / "training_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print((args.output / "training_report.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
