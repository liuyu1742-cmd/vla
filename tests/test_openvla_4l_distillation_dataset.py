from pathlib import Path

import numpy as np
import pytest

from tools.openvla_4l_distillation_dataset import (
    action_phase,
    build_distillation_dataset,
    select_indices,
)


def _episode(path: Path, actions: np.ndarray) -> None:
    frames = np.zeros((len(actions), 8, 8, 3), dtype=np.uint8)
    np.savez_compressed(path, frames=frames, actions=actions.astype(np.float32))


def test_action_phase_rejects_invalid_actions():
    with pytest.raises(ValueError, match="shape"):
        action_phase(np.zeros(6, dtype=np.float32))
    invalid = np.zeros(7, dtype=np.float32)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        action_phase(invalid)


def test_select_indices_caps_settle_fraction():
    actions = np.asarray(
        [[1, 0, 0, 0, 0, 0, 0]] * 4
        + [[1, 0, 0, 0, 0, 0, 1]] * 4
        + [[0, 0, 0, 0, 0, 0, 1]] * 20,
        dtype=np.float32,
    )
    indices = select_indices(
        actions,
        max_settle_fraction=0.15,
        limit=10,
        seed=7,
    )
    phases = [action_phase(actions[index]) for index in indices]
    assert len(indices) == 10
    assert phases.count("settle") <= 1
    assert "open_motion" in phases
    assert "closed_motion" in phases


def test_builder_writes_aligned_finite_dataset_and_manifest(tmp_path):
    nominal = tmp_path / "nominal"
    recovery = tmp_path / "recovery"
    output = tmp_path / "output"
    nominal.mkdir()
    recovery.mkdir()
    actions = np.asarray(
        [[1, 0, 0, 0, 0, 0, 0]] * 6
        + [[1, 0, 0, 0, 0, 0, 1]] * 6
        + [[0, 0, 0, 0, 0, 0, 1]] * 6,
        dtype=np.float32,
    )
    _episode(nominal / "episode_seed_000.npz", actions)
    _episode(recovery / "episode_seed_000_round_00.npz", actions)

    report = build_distillation_dataset(
        nominal,
        recovery,
        output,
        seeds=(0,),
        max_settle_fraction=0.15,
        samples_per_seed=20,
    )

    assert report["seeds"] == [0]
    assert len(report["source_hashes"]) == 1
    with np.load(output / "episode_seed_000.npz", allow_pickle=False) as episode:
        assert episode["frames"].shape[0] == episode["actions"].shape[0] == 20
        assert episode["actions"].shape[1:] == (7,)
        assert np.isfinite(episode["actions"]).all()
        phases = [action_phase(action) for action in episode["actions"]]
        assert phases.count("settle") <= 3
    assert (output / "distillation_manifest.json").is_file()


def test_builder_rejects_duplicate_recovery_content(tmp_path):
    nominal = tmp_path / "nominal"
    recovery = tmp_path / "recovery"
    output = tmp_path / "output"
    nominal.mkdir()
    recovery.mkdir()
    actions = np.asarray(
        [[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    _episode(nominal / "episode_seed_000.npz", actions)
    first = recovery / "episode_seed_000_round_00.npz"
    second = recovery / "episode_seed_000_round_01.npz"
    _episode(first, actions)
    second.write_bytes(first.read_bytes())

    with pytest.raises(ValueError, match="duplicate recovery"):
        build_distillation_dataset(
            nominal,
            recovery,
            output,
            seeds=(0,),
            max_settle_fraction=0.15,
            samples_per_seed=4,
        )
