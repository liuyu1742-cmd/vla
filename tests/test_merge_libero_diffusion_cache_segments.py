import json
from pathlib import Path

import numpy as np
import pytest

from tools.merge_libero_diffusion_cache_segments import merge_cache_segments
from tools.prepare_libero_diffusion_data import EXPECTED_TFRECORD_COUNTS


SHARED_CONFIG = {
    "encoder": "resnet18_imagenet1k_v1",
    "feature_dim": 512,
    "image_size": 128,
    "max_per_instruction": 10,
    "split": "train",
    "shuffle_files": False,
    "view": "observation.image static third-person",
}
DATASET_VALIDATION = {
    "dataset_root": "synthetic",
    "suite_counts": dict(EXPECTED_TFRECORD_COUNTS),
    "total_tfrecords": 96,
    "total_bytes": 123,
}


def write_segment(
    root: Path,
    suite: str,
    start: int,
    episode_rows: list[tuple[str, int]],
    *,
    selected_count: int | None = None,
    feature_dtype=np.float16,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    segment_root = root / f"{len(list(root.iterdir())):03d}_{suite}_{start:03d}"
    episode_root = segment_root / "episodes"
    episode_root.mkdir(parents=True)
    records = []
    for local_index, (instruction, episode_index) in enumerate(episode_rows):
        relative_path = Path("episodes") / f"{local_index:05d}.npz"
        np.savez_compressed(
            segment_root / relative_path,
            features=np.ones((1, 512), dtype=feature_dtype),
            states=np.ones((1, 8), dtype=np.float32),
            actions=np.ones((1, 7), dtype=np.float32),
            instruction=np.asarray(instruction),
            suite=np.asarray(suite),
            episode_index=np.asarray(episode_index, dtype=np.int64),
        )
        records.append(
            {
                "path": relative_path.as_posix(),
                "suite": suite,
                "instruction": instruction,
                "episode_index": episode_index,
                "length": 1,
                "boundary": {"start": 0, "stop": 1},
            }
        )
    manifest = {
        "schema_version": 1,
        "source_commit": "commit",
        "git_available": True,
        "config": dict(SHARED_CONFIG),
        "dataset_validation": dict(DATASET_VALIDATION),
        "segment": {
            "suite": suite,
            "selected_start": start,
            "selected_count": len(records) if selected_count is None else selected_count,
        },
        "episodes": records,
    }
    manifest_path = segment_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_merge_rejects_overlapping_segment_ranges(tmp_path: Path):
    manifests = [
        write_segment(tmp_path / "segments", "libero_spatial_no_noops", 0, [("a", 0)]),
        write_segment(tmp_path / "segments", "libero_spatial_no_noops", 0, [("a", 1)]),
    ]

    with pytest.raises(RuntimeError, match="overlap"):
        merge_cache_segments(manifests, tmp_path / "final")


def test_merge_rejects_missing_segment_range(tmp_path: Path):
    manifests = [
        write_segment(tmp_path / "segments", "libero_spatial_no_noops", 0, [("a", 0)]),
        write_segment(tmp_path / "segments", "libero_spatial_no_noops", 2, [("a", 2)]),
    ]

    with pytest.raises(RuntimeError, match="gap"):
        merge_cache_segments(manifests, tmp_path / "final")


def test_merge_rejects_duplicate_episode_identity(tmp_path: Path):
    manifests = [
        write_segment(tmp_path / "segments", "libero_spatial_no_noops", 0, [("a", 0)]),
        write_segment(tmp_path / "segments", "libero_spatial_no_noops", 1, [("a", 0)]),
    ]

    with pytest.raises(RuntimeError, match="duplicate"):
        merge_cache_segments(manifests, tmp_path / "final")


def test_merge_rejects_segment_count_mismatch(tmp_path: Path):
    manifest = write_segment(
        tmp_path / "segments",
        "libero_spatial_no_noops",
        0,
        [("a", 0)],
        selected_count=2,
    )

    with pytest.raises(RuntimeError, match="segment count"):
        merge_cache_segments([manifest], tmp_path / "final")


def test_merge_rejects_non_fp16_features(tmp_path: Path):
    manifest = write_segment(
        tmp_path / "segments",
        "libero_spatial_no_noops",
        0,
        [("a", 0)],
        feature_dtype=np.float32,
    )

    with pytest.raises(RuntimeError, match="float16"):
        merge_cache_segments([manifest], tmp_path / "final")


def test_merge_publishes_complete_400_episode_cache(tmp_path: Path):
    manifests = []
    segment_root = tmp_path / "segments"
    for suite in EXPECTED_TFRECORD_COUNTS:
        suite_rows = [
            (f"instruction {instruction_index}", instruction_index * 10 + occurrence)
            for instruction_index in range(10)
            for occurrence in range(10)
        ]
        for start in range(0, 100, 20):
            manifests.append(
                write_segment(segment_root, suite, start, suite_rows[start : start + 20])
            )

    manifest = merge_cache_segments(manifests, tmp_path / "final")

    canonical = tmp_path / "final" / "feature_cache"
    persisted = json.loads((canonical / "dataset_manifest.json").read_text())
    assert len(manifest["episodes"]) == 400
    assert len(persisted["episodes"]) == 400
    assert persisted["episodes"][0]["boundary"] == {"start": 0, "stop": 1}
    assert persisted["episodes"][-1]["boundary"] == {"start": 399, "stop": 400}
    assert len(persisted["source_segment_manifests"]) == 20
    assert len(list((canonical / "episodes").glob("*.npz"))) == 400
