import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.openvla_4l_runner import (
    collect_five_shot_paths,
    render_training_evidence,
)


def test_collect_five_shot_paths_uses_exact_seeds(tmp_path):
    for seed in (0, 1, 2, 3, 5):
        np.savez(
            tmp_path / f"episode_seed_{seed:03d}.npz",
            frames=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            actions=np.zeros((1, 7), dtype=np.float32),
        )
    paths = collect_five_shot_paths(tmp_path)
    assert [path.name for path in paths] == [
        "episode_seed_000.npz",
        "episode_seed_001.npz",
        "episode_seed_002.npz",
        "episode_seed_003.npz",
        "episode_seed_005.npz",
    ]


def test_collect_five_shot_paths_reports_missing_seed(tmp_path):
    try:
        collect_five_shot_paths(tmp_path)
    except FileNotFoundError as error:
        assert "episode_seed_000.npz" in str(error)
    else:
        raise AssertionError("missing demonstrations must fail")


def test_render_training_evidence_creates_real_log_contact_sheet(tmp_path):
    frame = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), "navy").save(frame)
    report = {
        "run_id": "unit-run",
        "method_label": "LoRA (rank=32)",
        "checkpoint": "models/unit",
        "gpu": "NVIDIA GeForce RTX 3090",
        "training_frames": [str(frame)],
        "steps": [
            {
                "step": 1,
                "loss": 1.25,
                "peak_vram_gb": 3.5,
                "gpu_memory_used_mib": 4096,
            }
        ],
    }
    report_path = tmp_path / "training_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "evidence.png"
    render_training_evidence(report_path, output)
    with Image.open(output) as rendered:
        assert rendered.width >= 800
        assert rendered.height >= 500

