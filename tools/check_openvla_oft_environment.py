"""Record whether the local environment can run official OpenVLA-OFT."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}
OFFICIAL_FLAGS = {
    "use_l1_regression": True,
    "use_diffusion": False,
    "use_film": False,
    "num_images_in_input": 2,
    "use_proprio": True,
    "center_crop": True,
    "num_open_loop_steps": 8,
}


def required_checkpoint(task_suite_name: str) -> str:
    try:
        return CHECKPOINTS[task_suite_name]
    except KeyError as exc:
        raise ValueError(f"unsupported LIBERO suite: {task_suite_name}") from exc


def _version(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _source_commit(source_dir: Path) -> str | None:
    if not (source_dir / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_environment_report(source_dir: Path) -> dict:
    dependencies = {
        name: _version(name)
        for name in ("torch", "transformers", "peft", "tensorflow", "libero")
    }
    torch_info = dependencies["torch"]
    gpu = None
    cuda = None
    if torch_info["available"]:
        import torch

        cuda = torch.version.cuda
        if torch.cuda.is_available():
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "total_vram_gb": torch.cuda.get_device_properties(0).total_memory
                / (1024**3),
            }
    return {
        "schema_version": "openvla_oft_environment_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "source_dir": str(source_dir.resolve()),
        "source_commit": _source_commit(source_dir),
        "source_present": (source_dir / "LIBERO.md").is_file(),
        "dependencies": dependencies,
        "cuda_version": cuda,
        "gpu": gpu,
        "official_flags": OFFICIAL_FLAGS,
        "checkpoints": CHECKPOINTS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=REPO_ROOT / "third_party" / "openvla-oft"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiment_4_6" / "environment.json",
    )
    args = parser.parse_args()
    report = build_environment_report(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

