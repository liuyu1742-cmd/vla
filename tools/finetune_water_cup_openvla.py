"""LoRA fine-tune local OpenVLA on successful RoboCasa water-cup expert data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = ROOT / "third_party" / "openvla"
DATA_ROOT = ROOT / "datasets" / "water_cup_expert"
MODEL_ROOT = ROOT / "models" / "openvla-7b"
INSTRUCTION = "pick up the glass cup and place it in the cabinet"


def episode_paths(data_root: Path, *, held_out_seed: int, train: bool) -> list[Path]:
    paths = sorted(data_root.glob("episode_seed_*.npz"))
    selected = [path for path in paths if ((f"{held_out_seed:03d}" not in path.name) == train)]
    if not selected:
        raise ValueError("no episodes selected for requested split")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "openvla-water-cup-lora")
    parser.add_argument("--held-out-seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    sys.path.insert(0, str(OPENVLA_ROOT))
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer

    processor = AutoProcessor.from_pretrained(str(args.model_root), trust_remote_code=True, local_files_only=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    class ExpertDataset(Dataset):
        def __init__(self, paths: list[Path]) -> None:
            self.items: list[tuple[np.ndarray, np.ndarray]] = []
            for path in paths:
                with np.load(path) as episode:
                    self.items.extend(zip(episode["frames"], episode["actions"]))

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict:
            frame, action = self.items[index]
            builder = PurePromptBuilder("openvla")
            builder.add_turn("human", f"What action should the robot take to {INSTRUCTION}?")
            builder.add_turn("gpt", action_tokenizer(action))
            token_ids = processor.tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids
            input_ids = torch.tensor(token_ids)
            labels = input_ids.clone()
            labels[: -(len(action) + 1)] = -100
            return {"pixel_values": processor.image_processor.apply_transform(Image.fromarray(frame)), "input_ids": input_ids, "labels": labels, "dataset_name": "water_cup_expert"}

    train_data = ExpertDataset(episode_paths(args.data_root, held_out_seed=args.held_out_seed, train=True))
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collator)
    model = AutoModelForVision2Seq.from_pretrained(str(args.model_root), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True).to("cuda:0")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=8, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    model.train()
    iterator = iter(loader)
    losses: list[float] = []
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(), pixel_values=batch["pixel_values"].to("cuda", dtype=torch.bfloat16), labels=batch["labels"].cuda())
        output.loss.backward()
        optimizer.step(); optimizer.zero_grad()
        losses.append(float(output.loss.detach().cpu()))
        print(f"step={step + 1} loss={losses[-1]:.5f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    (args.output / "training_report.json").write_text(__import__("json").dumps({"steps": args.steps, "train_samples": len(train_data), "held_out_seed": args.held_out_seed, "losses": losses}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
