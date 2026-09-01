import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.libero_diffusion_contracts import ActionNormalizer
from tools.train_libero_diffusion_policy import (
    FeatureWindowDataset,
    load_checkpoint,
    main,
    render_training_process,
    save_checkpoint,
)


def write_episode(
    path: Path,
    *,
    value: float,
    length: int = 3,
) -> None:
    np.savez_compressed(
        path,
        features=np.full((length, 4), value, dtype=np.float16),
        states=np.full((length, 2), value, dtype=np.float32),
        actions=np.full((length, 2), value, dtype=np.float32),
        instruction=np.asarray(f"instruction {value}"),
        suite=np.asarray("suite"),
        episode_index=np.asarray(int(value)),
    )


def write_manifest(cache_root: Path) -> Path:
    episodes_dir = cache_root / "episodes"
    episodes_dir.mkdir(parents=True)
    write_episode(episodes_dir / "episode_00000.npz", value=0.0)
    write_episode(episodes_dir / "episode_00001.npz", value=1.0)
    manifest = {
        "episodes": [
            {"path": "episodes/episode_00000.npz", "length": 3},
            {"path": "episodes/episode_00001.npz", "length": 3},
        ]
    }
    path = cache_root / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_feature_windows_repeat_only_inside_each_episode(tmp_path: Path):
    manifest_path = write_manifest(tmp_path / "feature_cache")
    normalizer = ActionNormalizer.fit(
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    )
    dataset = FeatureWindowDataset(
        manifest_path,
        normalizer=normalizer,
        obs_horizon=2,
        action_horizon=4,
    )

    assert len(dataset) == 6
    for index in range(3):
        np.testing.assert_array_equal(dataset[index]["actions"].numpy(), -1.0)
    for index in range(3, 6):
        np.testing.assert_array_equal(dataset[index]["actions"].numpy(), 1.0)


def test_checkpoint_round_trip_restores_strict_training_state(tmp_path: Path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        optimizer.step()

    normalizer = ActionNormalizer.fit(
        np.asarray([[0.0, -1.0], [2.0, 3.0]], dtype=np.float32)
    )
    checkpoint = tmp_path / "step_2.pt"
    model_config = {"input_dim": 3, "output_dim": 2}
    manifest_hash = hashlib.sha256(b"manifest").hexdigest()
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=2,
        normalizer=normalizer,
        seed=42,
        data_manifest_hash=manifest_hash,
        model_config=model_config,
    )
    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(checkpoint.suffix + ".tmp").exists()

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    payload = load_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_data_manifest_hash=manifest_hash,
        expected_model_config=model_config,
    )

    assert payload["step"] == 2
    assert payload["seed"] == 42
    assert payload["data_manifest_hash"] == manifest_hash
    assert payload["model_config"] == model_config
    assert payload["normalizer"] == {
        "q01": normalizer.q01.tolist(),
        "q99": normalizer.q99.tolist(),
    }
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert restored_optimizer.state_dict()["state"]


def test_checkpoint_rejects_manifest_or_model_contract_drift(tmp_path: Path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    normalizer = ActionNormalizer.fit(
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    )
    checkpoint = tmp_path / "step_0.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=0,
        normalizer=normalizer,
        seed=42,
        data_manifest_hash="expected",
        model_config={"width": 3},
    )
    with pytest.raises(RuntimeError, match="manifest"):
        load_checkpoint(
            checkpoint,
            model=torch.nn.Linear(3, 2),
            expected_data_manifest_hash="changed",
            expected_model_config={"width": 3},
        )
    with pytest.raises(RuntimeError, match="config"):
        load_checkpoint(
            checkpoint,
            model=torch.nn.Linear(3, 2),
            expected_data_manifest_hash="expected",
            expected_model_config={"width": 4},
        )


def test_training_process_plot_contains_only_logged_points(tmp_path: Path):
    metrics = tmp_path / "train_metrics.jsonl"
    rows = [
        {"step": 1, "train_loss": 1.5, "validation_loss": 1.7},
        {"step": 2, "train_loss": 1.1, "validation_loss": 1.3},
    ]
    metrics.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "training_process.png"
    summary = render_training_process(metrics, output)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert summary == {"points": 2, "first_step": 1, "last_step": 2}


def test_two_step_cpu_training_smoke_writes_reloadable_artifacts(tmp_path: Path):
    experiment_dir = tmp_path / "experiment"
    write_manifest(experiment_dir / "feature_cache")

    exit_code = main(
        [
            "--experiment-dir",
            str(experiment_dir),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--checkpoint-every",
            "1",
            "--validation-every",
            "1",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--run-name",
            "cpu_smoke",
        ]
    )

    run_dir = experiment_dir / "runs" / "cpu_smoke"
    assert exit_code == 0
    assert (run_dir / "checkpoints" / "step_1.pt").is_file()
    assert (run_dir / "checkpoints" / "step_2.pt").is_file()
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "training_process.png").is_file()
    rows = [
        json.loads(line)
        for line in (run_dir / "train_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    assert all(np.isfinite(row["validation_loss"]) for row in rows)
