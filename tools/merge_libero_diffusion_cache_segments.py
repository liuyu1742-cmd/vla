"""Merge completed LIBERO feature-cache segments into one canonical cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping
from uuid import uuid4

import numpy as np

from tools.prepare_libero_diffusion_data import (
    EXPECTED_TFRECORD_COUNTS,
    _assert_safe_cache_overwrite,
    cleanup_staging_run,
    publish_feature_cache,
)


EXPECTED_CONFIG = {
    "encoder": "resnet18_imagenet1k_v1",
    "feature_dim": 512,
    "image_size": 128,
    "max_per_instruction": 10,
    "split": "train",
    "shuffle_files": False,
    "view": "observation.image static third-person",
}
EXPECTED_TOTAL_EPISODES = 400
EXPECTED_SUITE_EPISODES = 100
EXPECTED_INSTRUCTIONS_PER_SUITE = 10
EXPECTED_EPISODES_PER_INSTRUCTION = 10


def _normalized_config(config: Mapping[str, object]) -> dict[str, object]:
    normalized = {key: config.get(key) for key in EXPECTED_CONFIG}
    if normalized != EXPECTED_CONFIG:
        raise RuntimeError(
            f"segment config differs from required shared config: {normalized}"
        )
    return normalized


def _validate_dataset(dataset_validation: Mapping[str, object]) -> dict[str, object]:
    suite_counts = dataset_validation.get("suite_counts")
    if suite_counts != EXPECTED_TFRECORD_COUNTS:
        raise RuntimeError("segment dataset validation does not contain four suite shard counts")
    if dataset_validation.get("total_tfrecords") != sum(EXPECTED_TFRECORD_COUNTS.values()):
        raise RuntimeError("segment dataset validation does not contain exactly 96 TFRecords")
    return dict(dataset_validation)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_episode_archive(
    path: Path, row: Mapping[str, object], segment_suite: str
) -> str:
    if not path.is_file():
        raise RuntimeError(f"segment episode file is missing: {path}")
    try:
        digest = _sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "features",
                "states",
                "actions",
                "instruction",
                "suite",
                "episode_index",
            }
            if not required.issubset(archive.files):
                raise RuntimeError(f"segment NPZ is missing required arrays: {path}")
            features = archive["features"]
            states = archive["states"]
            actions = archive["actions"]
            length = int(row["length"])
            if features.dtype != np.float16:
                raise RuntimeError(f"segment features must be float16: {path}")
            if states.dtype != np.float32 or actions.dtype != np.float32:
                raise RuntimeError(f"segment state/action arrays must be float32: {path}")
            if features.shape != (length, 512):
                raise RuntimeError(f"segment features have invalid shape: {path}")
            if states.shape != (length, 8) or actions.shape != (length, 7):
                raise RuntimeError(f"segment state/action arrays have invalid shape: {path}")
            if str(archive["instruction"].item()) != str(row["instruction"]):
                raise RuntimeError(f"segment instruction metadata mismatch: {path}")
            if str(archive["suite"].item()) != segment_suite:
                raise RuntimeError(f"segment suite metadata mismatch: {path}")
            if int(archive["episode_index"].item()) != int(row["episode_index"]):
                raise RuntimeError(f"segment episode index metadata mismatch: {path}")
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"segment NPZ is not SHA-256-readable and valid: {path}") from error
    return digest


def merge_cache_segments(
    manifest_paths: Iterable[Path], output_dir: Path
) -> dict:
    manifest_paths = [Path(path).resolve() for path in manifest_paths]
    if not manifest_paths:
        raise ValueError("at least one segment manifest is required")

    segments: list[dict[str, object]] = []
    common_source_commit: str | None = None
    common_git_available: bool | None = None
    common_config: dict[str, object] | None = None
    common_dataset_validation: dict[str, object] | None = None
    ranges: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)

    for index, manifest_path in enumerate(manifest_paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        segment = manifest.get("segment")
        if not isinstance(segment, dict):
            raise RuntimeError(f"segment manifest has no segment metadata: {manifest_path}")
        suite = str(segment.get("suite"))
        if suite not in EXPECTED_TFRECORD_COUNTS:
            raise RuntimeError(f"unknown suite in segment manifest: {suite}")
        start = int(segment.get("selected_start", -1))
        count = int(segment.get("selected_count", -1))
        rows = list(manifest.get("episodes", []))
        if start < 0 or count <= 0 or len(rows) != count:
            raise RuntimeError(
                f"segment count does not match manifest episodes: {manifest_path}"
            )
        normalized_config = _normalized_config(manifest.get("config", {}))
        dataset_validation = _validate_dataset(manifest.get("dataset_validation", {}))
        source_commit = manifest.get("source_commit")
        git_available = bool(manifest.get("git_available"))
        if index == 0:
            common_source_commit = source_commit
            common_git_available = git_available
            common_config = normalized_config
            common_dataset_validation = dataset_validation
        elif (
            source_commit != common_source_commit
            or git_available != common_git_available
            or normalized_config != common_config
            or dataset_validation != common_dataset_validation
        ):
            raise RuntimeError("segment provenance or shared configuration differs")
        ranges[suite].append((start, start + count, manifest_path))
        segments.append(
            {
                "manifest_path": manifest_path,
                "manifest_root": manifest_path.parent,
                "suite": suite,
                "start": start,
                "count": count,
                "rows": rows,
            }
        )

    for suite, suite_ranges in ranges.items():
        expected_start = 0
        for start, stop, manifest_path in sorted(suite_ranges):
            if start < expected_start:
                raise RuntimeError(f"segment range overlap for {suite}: {manifest_path}")
            if start > expected_start:
                raise RuntimeError(
                    f"segment range gap for {suite}: expected {expected_start}, got {start}"
                )
            expected_start = stop

    validated_rows: list[dict[str, object]] = []
    identities: set[tuple[str, str, int]] = set()
    suite_instruction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for segment in segments:
        suite = str(segment["suite"])
        manifest_root = Path(segment["manifest_root"])
        for raw_row in segment["rows"]:
            row = dict(raw_row)
            if row.get("suite") != suite:
                raise RuntimeError("segment episode suite differs from segment metadata")
            length = int(row.get("length", 0))
            if length <= 0 or row.get("boundary") != {"start": 0, "stop": length}:
                raise RuntimeError("segment manifest length/boundary is invalid")
            identity = (suite, str(row.get("instruction")), int(row.get("episode_index", -1)))
            if identity in identities:
                raise RuntimeError(f"duplicate segment episode identity: {identity}")
            identities.add(identity)
            source_path = (manifest_root / str(row.get("path"))).resolve()
            try:
                source_path.relative_to(manifest_root.resolve())
            except ValueError as error:
                raise RuntimeError("segment episode path escapes its cache root") from error
            row["source_path"] = source_path
            row["sha256"] = _validate_episode_archive(source_path, row, suite)
            validated_rows.append(row)
            suite_instruction_counts[suite][identity[1]] += 1

    if set(ranges) != set(EXPECTED_TFRECORD_COUNTS):
        raise RuntimeError("all four suites must be represented by segment manifests")
    if len(validated_rows) != EXPECTED_TOTAL_EPISODES:
        raise RuntimeError(f"final cache must contain exactly {EXPECTED_TOTAL_EPISODES} episodes")
    for suite in EXPECTED_TFRECORD_COUNTS:
        suite_count = sum(suite_instruction_counts[suite].values())
        if suite_count != EXPECTED_SUITE_EPISODES:
            raise RuntimeError(f"suite {suite} must contain exactly 100 episodes")
        instruction_counts = suite_instruction_counts[suite]
        if len(instruction_counts) != EXPECTED_INSTRUCTIONS_PER_SUITE:
            raise RuntimeError(f"suite {suite} must contain exactly 10 unique instructions")
        if any(count != EXPECTED_EPISODES_PER_INSTRUCTION for count in instruction_counts.values()):
            raise RuntimeError(f"suite {suite} instruction count must equal 10")
        suite_stop = max(stop for _start, stop, _path in ranges[suite])
        if suite_stop != EXPECTED_SUITE_EPISODES:
            raise RuntimeError(f"suite {suite} segment ranges must cover [0, 100)")

    assert common_config is not None
    assert common_dataset_validation is not None
    output_dir = Path(output_dir)
    canonical_dir = output_dir / "feature_cache"
    _assert_safe_cache_overwrite(canonical_dir, common_source_commit, common_config)
    run_dir = output_dir / ".feature_cache_merge_runs" / uuid4().hex
    candidate_dir = run_dir / "feature_cache"
    episode_dir = candidate_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=False)
    published = False
    try:
        final_rows: list[dict[str, object]] = []
        global_frame = 0
        for global_index, row in enumerate(validated_rows):
            suite = str(row["suite"])
            episode_index = int(row["episode_index"])
            filename = f"{global_index:05d}_{suite}_episode_{episode_index:05d}.npz"
            relative_path = Path("episodes") / filename
            shutil.copy2(Path(row["source_path"]), candidate_dir / relative_path)
            length = int(row["length"])
            final_rows.append(
                {
                    "path": relative_path.as_posix(),
                    "suite": suite,
                    "instruction": str(row["instruction"]),
                    "episode_index": episode_index,
                    "length": length,
                    "boundary": {"start": global_frame, "stop": global_frame + length},
                    "sha256": str(row["sha256"]),
                }
            )
            global_frame += length
        final_manifest = {
            "schema_version": 1,
            "source_commit": common_source_commit,
            "git_available": common_git_available,
            "config": common_config,
            "dataset_validation": common_dataset_validation,
            "source_segment_manifests": [str(path) for path in manifest_paths],
            "episodes": final_rows,
        }
        manifest_path = candidate_dir / "dataset_manifest.json"
        temporary_manifest = manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        publish_feature_cache(candidate_dir, canonical_dir)
        published = True
        return final_manifest
    finally:
        if not cleanup_staging_run(run_dir) and not published:
            raise RuntimeError(f"merge candidate cleanup failed: {run_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = merge_cache_segments(args.segment_manifest, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir / "feature_cache"), "episodes": len(manifest["episodes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
