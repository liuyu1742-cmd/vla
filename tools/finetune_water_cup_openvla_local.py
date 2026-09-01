"""Local-only LoRA fine-tuning without RLDS/dlimp dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "pick up the glass cup and place it in the cabinet"


def action_text(tokenizer, action: np.ndarray) -> str:
    bins = np.linspace(-1.0, 1.0, 256)
    ids = tokenizer.vocab_size - np.digitize(np.clip(action, -1.0, 1.0), bins)
    return tokenizer.decode(list(ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "openvla-water-cup-lora")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model_dir = ROOT / "models" / "openvla-7b"
    paths = [ROOT / "datasets" / "water_cup_expert" / f"episode_seed_{seed:03d}.npz" for seed in (0, 1)]
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    items = []
    for path in paths:
        with np.load(path) as ep:
            items.extend(zip(ep["frames"], ep["actions"]))
    class Data(Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, idx):
            frame, action = items[idx]
            prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut: {action_text(processor.tokenizer, action)}</s>"
            ids = torch.tensor(processor.tokenizer(prompt, add_special_tokens=True).input_ids)
            labels = ids.clone(); labels[:-(len(action)+1)] = -100
            return Image.fromarray(frame), ids, labels
    def collate(batch):
        images, ids, labels = zip(*batch)
        max_len = max(x.numel() for x in ids); pad = processor.tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad, dtype=torch.long); out_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (x, y) in enumerate(zip(ids, labels)): input_ids[i,:x.numel()] = x; out_labels[i,:y.numel()] = y
        pixels = torch.stack([processor.image_processor.apply_transform(image) for image in images])
        return input_ids, out_labels, pixels
    loader = DataLoader(Data(), batch_size=1, shuffle=True, collate_fn=collate)
    model = AutoModelForVision2Seq.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True).to("cuda")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM")); model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4); losses=[]
    for step, (ids, labels, pixels) in zip(range(args.steps), loader):
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(input_ids=ids.cuda(), attention_mask=(ids != processor.tokenizer.pad_token_id).cuda(), pixel_values=pixels.cuda().to(torch.bfloat16), labels=labels.cuda())
        out.loss.backward(); opt.step(); opt.zero_grad(); losses.append(float(out.loss.detach().cpu())); print(f"step={step+1} loss={losses[-1]:.5f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True); model.save_pretrained(args.output); processor.save_pretrained(args.output)
    (args.output / "training_report.json").write_text(json.dumps({"steps":args.steps,"samples":len(items),"losses":losses},indent=2),encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
