"""Synchronize object-level operation metadata after task-directory renaming."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tools.archive_chapter5_objects import task_dir

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); a=p.parse_args()
    m=json.loads(a.manifest.read_text(encoding='utf-8')); changed=0
    for row in m['objects']:
        op=a.dataset_root/task_dir(row['section'],row['task_name'])/row['object_name']/ 'operation.json'
        if op.is_file():
            data=json.loads(op.read_text(encoding='utf-8')); data['task_name']=row['task_name']; data['section']=row['section']; op.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); changed+=1
    print(json.dumps({'operation_metadata_updated':changed},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
