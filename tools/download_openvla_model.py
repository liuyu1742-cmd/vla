"""Download the public OpenVLA-7B checkpoint with resumable Hugging Face transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "openvla/openvla-7b"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
MODEL_FILES = (
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "config.json",
    "configuration_prismatic.py",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "modeling_prismatic.py",
    "preprocessor_config.json",
    "processing_prismatic.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def default_target() -> Path:
    return ROOT / "models" / "openvla-7b"


def build_download_metadata(target: Path, endpoint: str) -> dict[str, str]:
    return {"repo_id": REPO_ID, "endpoint": endpoint, "target": str(target)}


def model_files() -> tuple[str, ...]:
    return MODEL_FILES


def download_with_resumable_curl(target: Path, endpoint: str) -> None:
    """Use the mirror's HTTP range support when hub-client redirects are incompatible."""
    for filename in model_files():
        url = f"{endpoint.rstrip('/')}/{REPO_ID}/resolve/main/{quote(filename)}"
        output = target / filename
        print(json.dumps({"status": "downloading", "file": filename}, ensure_ascii=False), flush=True)
        subprocess.run(
            [
                "curl.exe",
                "--fail",
                "--location",
                "--continue-at",
                "-",
                "--retry",
                "10",
                "--retry-all-errors",
                "--output",
                str(output),
                url,
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=default_target())
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    args.target.mkdir(parents=True, exist_ok=True)
    metadata = build_download_metadata(args.target, args.endpoint)
    (args.target / "DOWNLOAD_REQUEST.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({"status": "starting", **metadata}, ensure_ascii=False), flush=True)
    download_with_resumable_curl(args.target, args.endpoint)
    print(json.dumps({"status": "completed", **metadata}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
