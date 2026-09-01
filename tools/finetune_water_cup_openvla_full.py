"""Full local-only LoRA training on every successful episode in the manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
INSTRUCTION='pick up the glass cup and place it in the cabinet'
def action_text(tokenizer,action):
    return tokenizer.decode(list(tokenizer.vocab_size-np.digitize(np.clip(action,-1.,1.),np.linspace(-1.,1.,256))))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=200);ap.add_argument('--output',type=Path,default=ROOT/'models'/'openvla-water-cup-lora-full');args=ap.parse_args()
    import torch
    from PIL import Image
    from peft import LoraConfig,get_peft_model
    from torch.utils.data import Dataset,DataLoader
    from transformers import AutoModelForVision2Seq,AutoProcessor
    manifest=json.loads((ROOT/'datasets'/'water_cup_expert'/'training_manifest.json').read_text()); paths=[ROOT/x['episode'] for x in manifest['train']]
    processor=AutoProcessor.from_pretrained(str(ROOT/'models'/'openvla-7b'),trust_remote_code=True,local_files_only=True);samples=[]
    for p in paths:
        with np.load(p) as e:samples.extend(zip(e['frames'],e['actions']))
    class D(Dataset):
        def __len__(self):return len(samples)
        def __getitem__(self,i):
            f,a=samples[i];ids=torch.tensor(processor.tokenizer(f'In: What action should the robot take to {INSTRUCTION}?\nOut: {action_text(processor.tokenizer,a)}</s>',add_special_tokens=True).input_ids);y=ids.clone();y[:-(len(a)+1)]=-100;return Image.fromarray(f),ids,y
    def c(b):
        im,x,y=zip(*b);n=max(z.numel() for z in x);pad=processor.tokenizer.pad_token_id;X=torch.full((len(b),n),pad,dtype=torch.long);Y=torch.full((len(b),n),-100,dtype=torch.long)
        for i,(a,d) in enumerate(zip(x,y)):X[i,:a.numel()]=a;Y[i,:d.numel()]=d
        return X,Y,torch.stack([processor.image_processor.apply_transform(z) for z in im])
    loader=DataLoader(D(),batch_size=1,shuffle=True,collate_fn=c);it=iter(loader);base=ROOT/'models'/'openvla-7b';m=AutoModelForVision2Seq.from_pretrained(str(base),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True,local_files_only=True).to('cuda');m=get_peft_model(m,LoraConfig(r=8,lora_alpha=8,target_modules='all-linear',task_type='CAUSAL_LM'));o=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-4);loss=[]
    for s in range(args.steps):
        try:x,y,p=next(it)
        except StopIteration:it=iter(loader);x,y,p=next(it)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=m(input_ids=x.cuda(),attention_mask=(x!=processor.tokenizer.pad_token_id).cuda(),pixel_values=p.cuda().to(torch.bfloat16),labels=y.cuda())
        z.loss.backward();o.step();o.zero_grad();loss.append(float(z.loss.detach().cpu()));print(f'step={s+1} loss={loss[-1]:.5f}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True);m.save_pretrained(args.output);processor.save_pretrained(args.output);(args.output/'training_report.json').write_text(json.dumps({'steps':args.steps,'train_episodes':[p.name for p in paths],'train_samples':len(samples),'losses':loss},indent=2))
if __name__=='__main__':main()
