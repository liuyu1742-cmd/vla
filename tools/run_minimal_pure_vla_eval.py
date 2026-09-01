"""Run the held-out pure-VLA minimum-loop evaluation sequentially."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "minimal_pure_vla_eval"
PYTHON = Path(r"C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe")
MANIFEST = ROOT / "datasets" / "formal_skills" / "organizing_toy" / "training_manifest_dagger_r2.json"
ADAPTER = ROOT / "models" / "openvla-organizing-toy-lora-dagger-r3"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in (101, 102, 103):
        seed_out = OUT / f"seed_{seed}"
        cmd = [
            str(PYTHON), "-m", "tools.formal_skill_rollout",
            "--seed", str(seed), "--steps", "300", "--action-repeat", "2",
            "--host", "127.0.0.1", "--port", "8774",
            "--manifest", str(MANIFEST), "--adapter-dir", str(ADAPTER),
            "--output-root", str(seed_out),
        ]
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (OUT / f"seed_{seed}.log").write_text(completed.stdout, encoding="utf-8")
        (OUT / f"seed_{seed}.err.log").write_text(completed.stderr, encoding="utf-8")
        report_path = seed_out / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        results.append({"seed": seed, "returncode": completed.returncode, "report": report})
        if completed.returncode != 0:
            break
    successful = sum(1 for item in results if item["report"] and item["report"].get("success"))
    summary = {
        "evaluation_mode": "pure_vla_guarded_rollout",
        "hybrid_servo": False,
        "seeds_requested": [101, 102, 103],
        "results": results,
        "successful_seeds": successful,
        "evaluated_seeds": len(results),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if len(results) == 3 and all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
