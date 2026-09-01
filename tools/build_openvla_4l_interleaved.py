"""Build an OpenVLA-4L base that retains representations across model depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def interleaved_indices(total_layers: int, keep_layers: int) -> tuple[int, ...]:
    if not 1 < keep_layers <= total_layers:
        raise ValueError("keep_layers must be between two and total_layers")
    return tuple(
        round(index * (total_layers - 1) / (keep_layers - 1))
        for index in range(keep_layers)
    )


def build(source: Path, output: Path, keep_layers: int) -> dict:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        str(source), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(source),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    layers = model.language_model.model.layers
    indices = interleaved_indices(len(layers), keep_layers)
    model.language_model.model.layers = torch.nn.ModuleList(
        [layers[index] for index in indices]
    )
    model.language_model.config.num_hidden_layers = keep_layers
    model.config.text_config.num_hidden_layers = keep_layers
    model.config._name_or_path = str(output.resolve())
    model.language_model.config._name_or_path = str(output.resolve())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(output)
    del model
    reloaded = AutoModelForVision2Seq.from_pretrained(
        str(output),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    report = {
        "schema_version": "openvla_4l_interleaved_base_v1",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_layers": 32,
        "kept_layers": keep_layers,
        "selected_layer_indices": list(indices),
        "total_parameters": total_parameters,
        "reload_layers": len(reloaded.language_model.model.layers),
        "reload_parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
        "status": "validated",
    }
    del reloaded
    if (
        report["reload_layers"] != keep_layers
        or report["reload_parameters"] != total_parameters
    ):
        raise RuntimeError("interleaved OpenVLA-4L reload validation failed")
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-layers", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source, args.output, args.keep_layers),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
