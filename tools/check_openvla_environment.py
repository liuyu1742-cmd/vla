"""Report CUDA and local-checkpoint readiness for the OpenVLA smoke test."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any


def build_report(model_dir: Path) -> dict[str, Any]:
    """Return a read-only environment report without loading the model."""
    import torch

    cuda_available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "model_files_present": (model_dir / "config.json").is_file(),
    }
    if cuda_available:
        values = torch.ones((16, 16), device="cuda")
        report["cuda_matmul_ok"] = bool(torch.isfinite(values @ values).all().item())
    else:
        report["cuda_matmul_ok"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "openvla-7b",
    )
    arguments = parser.parse_args()
    print(json.dumps(build_report(arguments.model_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
