"""Prepare deterministic, compact training data for the LIBERO diffusion policy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from itertools import islice
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Callable, Iterable, Iterator, Mapping
from uuid import uuid4

import numpy as np

from tools.libero_diffusion_contracts import transform_dataset_gripper


EXPECTED_TFRECORD_COUNTS = {
    "libero_spatial_no_noops": 16,
    "libero_object_no_noops": 32,
    "libero_goal_no_noops": 16,
    "libero_10_no_noops": 32,
}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
TFDS_RESOURCE_RLIMIT_NOFILE = 7
WINDOWS_NOFILE_LIMIT = 512
DECODER_SHUTDOWN_TIMEOUT_SECONDS = 5
# TensorFlow can spend several seconds releasing TFRecord and thread-pool
# resources after the completion sentinel is visible, especially for the
# longer LIBERO-10 trajectories. Normal completion gets a wider grace period;
# forced termination below remains intentionally short.
DECODER_NORMAL_SHUTDOWN_TIMEOUT_SECONDS = 60
STAGED_OPEN_RETRY_DEADLINE_SECONDS = 2.0
STAGED_OPEN_RETRY_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class EpisodeArrays:
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    instruction: str
    suite: str
    episode_index: int

    def __post_init__(self) -> None:
        lengths = {
            int(np.asarray(self.images).shape[0]),
            int(np.asarray(self.states).shape[0]),
            int(np.asarray(self.actions).shape[0]),
        }
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise ValueError("image, state, and action lengths must match and be nonzero")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if not self.suite.strip():
            raise ValueError("suite must be non-empty")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")


def select_episodes(
    episodes: Iterable[EpisodeArrays],
    max_per_instruction: int,
) -> list[EpisodeArrays]:
    """Preserve source order while capping each exact ``(suite, instruction)`` key."""

    return list(iter_selected_episodes(episodes, max_per_instruction))


def iter_selected_episodes(
    episodes: Iterable[EpisodeArrays],
    max_per_instruction: int,
) -> Iterator[EpisodeArrays]:
    """Yield selected episodes without retaining their raw images in memory."""

    if max_per_instruction <= 0:
        raise ValueError("max_per_instruction must be positive")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for episode in episodes:
        key = (episode.suite, episode.instruction)
        if counts[key] >= max_per_instruction:
            continue
        counts[key] += 1
        yield episode


def iter_selected_segment(
    episodes: Iterable[EpisodeArrays],
    *,
    suite: str,
    max_per_instruction: int,
    selected_start: int,
    selected_count: int,
) -> Iterator[EpisodeArrays]:
    """Select deterministically within one suite, then apply the selected offset."""

    if suite not in EXPECTED_TFRECORD_COUNTS:
        raise ValueError(f"unknown LIBERO suite: {suite}")
    if selected_start < 0:
        raise ValueError("selected_start must be nonnegative")
    if selected_count <= 0:
        raise ValueError("selected_count must be positive")
    suite_episodes = (episode for episode in episodes if episode.suite == suite)
    selected = iter_selected_episodes(suite_episodes, max_per_instruction)
    return islice(selected, selected_start, selected_start + selected_count)


def validate_expanded_tfrecords(dataset_root: Path) -> dict:
    """Validate the exact four-suite shard layout and reject Git LFS pointers."""

    dataset_root = Path(dataset_root)
    shards = sorted(dataset_root.rglob("*.tfrecord-*"))
    for shard in shards:
        with shard.open("rb") as stream:
            if stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                raise RuntimeError(f"Git LFS pointer was not expanded: {shard}")

    counts: dict[str, int] = {}
    for suite, expected_count in EXPECTED_TFRECORD_COUNTS.items():
        suite_root = dataset_root / suite
        if not suite_root.is_dir():
            raise RuntimeError(f"missing LIBERO suite directory: {suite}")
        count = len(list(suite_root.rglob("*.tfrecord-*")))
        if count != expected_count:
            raise RuntimeError(
                f"{suite} has {count} TFRecords; expected {expected_count}"
            )
        counts[suite] = count

    if len(shards) != sum(EXPECTED_TFRECORD_COUNTS.values()):
        raise RuntimeError(
            f"dataset has {len(shards)} TFRecords; "
            f"expected {sum(EXPECTED_TFRECORD_COUNTS.values())}"
        )
    return {
        "dataset_root": str(dataset_root.resolve()),
        "total_tfrecords": len(shards),
        "suite_counts": counts,
        "total_bytes": sum(shard.stat().st_size for shard in shards),
    }


def install_tfds_resource_compatibility() -> None:
    """Provide TFDS's import-time ``resource`` surface on Windows only.

    TFDS imports :mod:`resource` from ``core.shuffle`` even when reading with
    ``shuffle_files=False``.  The reader only needs these three symbols; the
    no-op setter deliberately avoids pretending Windows can change POSIX rlimits.
    """

    if sys.platform != "win32":
        return
    try:
        import resource  # noqa: F401
    except ModuleNotFoundError:
        resource = ModuleType("resource")
        resource.RLIMIT_NOFILE = TFDS_RESOURCE_RLIMIT_NOFILE
        resource.getrlimit = lambda _limit: (WINDOWS_NOFILE_LIMIT, WINDOWS_NOFILE_LIMIT)
        resource.setrlimit = lambda _limit, _value: None
        sys.modules["resource"] = resource


def _tfds_value(value: object) -> object:
    return value.numpy() if hasattr(value, "numpy") else value


def _instruction_text(value: object) -> str:
    value = _tfds_value(value)
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _episode_from_rlds(
    raw_episode: Mapping[str, object], suite: str, episode_index: int
) -> EpisodeArrays:
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    instruction: str | None = None
    for raw_step in raw_episode["steps"]:  # type: ignore[index]
        step = raw_step  # eager TFDS dictionaries support normal mapping access
        observation = step["observation"]
        images.append(np.asarray(_tfds_value(observation["image"]), dtype=np.uint8))
        states.append(np.asarray(_tfds_value(observation["state"]), dtype=np.float32))
        action = np.asarray(_tfds_value(step["action"]), dtype=np.float32).copy()
        action[-1] = transform_dataset_gripper(action[-1])
        actions.append(action)
        step_instruction = _instruction_text(step["language_instruction"])
        if instruction is None:
            instruction = step_instruction
        elif instruction != step_instruction:
            raise RuntimeError("RLDS episode contains inconsistent language_instruction")
    if instruction is None:
        raise RuntimeError("RLDS episode has no steps")
    return EpisodeArrays(
        images=np.stack(images),
        states=np.stack(states).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        instruction=instruction,
        suite=suite,
        episode_index=episode_index,
    )


def iter_rlds_episodes(
    dataset_root: Path,
    *,
    tfds_module: object | None = None,
    suite: str | None = None,
) -> Iterator[EpisodeArrays]:
    """Yield real LIBERO RLDS train episodes in stable builder/source order."""

    dataset_root = Path(dataset_root)
    if tfds_module is None:
        install_tfds_resource_compatibility()
        import tensorflow_datasets as tfds

        tfds_module = tfds
    suites = [suite] if suite is not None else list(EXPECTED_TFRECORD_COUNTS)
    for suite_name in suites:
        if suite_name not in EXPECTED_TFRECORD_COUNTS:
            raise ValueError(f"unknown LIBERO suite: {suite_name}")
        builder_dir = dataset_root / suite_name / "1.0.0"
        builder = tfds_module.builder_from_directory(str(builder_dir))
        dataset = builder.as_dataset(split="train", shuffle_files=False)
        for episode_index, raw_episode in enumerate(dataset):
            yield _episode_from_rlds(raw_episode, suite_name, episode_index)


def write_rlds_staging(
    episodes: Iterable[EpisodeArrays],
    staging_dir: Path,
    *,
    segment: Mapping[str, object] | None = None,
) -> int:
    """Produce atomically visible episodes and a completion sentinel."""

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for count, episode in enumerate(episodes, start=1):
        path = staging_dir / f"{count - 1:05d}.npz"
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                images=np.asarray(episode.images, dtype=np.uint8),
                states=np.asarray(episode.states, dtype=np.float32),
                actions=np.asarray(episode.actions, dtype=np.float32),
                instruction=np.asarray(episode.instruction),
                suite=np.asarray(episode.suite),
                episode_index=np.asarray(episode.episode_index, dtype=np.int64),
            )
        os.replace(temporary_path, path)
        acknowledgement = staging_dir / f"{count - 1:05d}.ack"
        while not acknowledgement.exists():
            time.sleep(0.05)
        acknowledgement.unlink()
    if count == 0:
        raise RuntimeError("RLDS selection produced no episodes")
    complete_path = staging_dir / "complete.json"
    temporary_complete = complete_path.with_suffix(".tmp")
    completion: dict[str, object] = {"episode_count": count}
    if segment is not None:
        completion["segment"] = dict(segment)
    temporary_complete.write_text(json.dumps(completion) + "\n", encoding="utf-8")
    os.replace(temporary_complete, complete_path)
    return count


def _staged_episode(path: Path) -> EpisodeArrays:
    deadline = time.monotonic() + STAGED_OPEN_RETRY_DEADLINE_SECONDS
    while True:
        try:
            with np.load(path, allow_pickle=False) as cached:
                return EpisodeArrays(
                    images=np.asarray(cached["images"], dtype=np.uint8),
                    states=np.asarray(cached["states"], dtype=np.float32),
                    actions=np.asarray(cached["actions"], dtype=np.float32),
                    instruction=str(cached["instruction"].item()),
                    suite=str(cached["suite"].item()),
                    episode_index=int(cached["episode_index"].item()),
                )
        except OSError as error:
            sharing_violation = isinstance(error, PermissionError) or getattr(
                error, "winerror", None
            ) in (32, 33)
            if not sharing_violation:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(STAGED_OPEN_RETRY_POLL_SECONDS, remaining))


def iter_live_staged_episodes(
    staging_dir: Path,
    *,
    child: object,
    poll_seconds: float = 0.05,
    expected_segment: Mapping[str, object] | None = None,
) -> Iterator[EpisodeArrays]:
    """Consume one child-produced episode at a time and remove it immediately."""

    staging_dir = Path(staging_dir)
    expected_index = 0
    while True:
        path = staging_dir / f"{expected_index:05d}.npz"
        if path.exists():
            episode = _staged_episode(path)
            yield episode
            path.unlink()
            acknowledgement = staging_dir / f"{expected_index:05d}.ack"
            acknowledgement.write_text("ack\n", encoding="utf-8")
            expected_index += 1
            continue
        complete_path = staging_dir / "complete.json"
        if complete_path.exists():
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            sentinel_segment = completion.get("segment")
            requested_segment = (
                dict(expected_segment) if expected_segment is not None else None
            )
            if sentinel_segment != requested_segment:
                raise RuntimeError(
                    "RLDS segment sentinel mismatch: "
                    f"requested {requested_segment}, decoder reported {sentinel_segment}"
                )
            expected_count = int(completion["episode_count"])
            if expected_count != expected_index:
                raise RuntimeError(
                    "RLDS staging count mismatch: "
                    f"consumed {expected_index}, decoder declared {expected_count}"
                )
            return
        poll = getattr(child, "poll")
        if poll() is not None:
            raise RuntimeError("RLDS decoder exited before writing completion sentinel")
        time.sleep(poll_seconds)


def start_rlds_decoder_subprocess(
    dataset_root: Path,
    staging_dir: Path,
    max_per_instruction: int,
    smoke_episodes: int,
    run_dir: Path,
    *,
    suite: str | None = None,
    selected_start: int = 0,
    selected_count: int | None = None,
) -> object:
    """Launch the TFDS producer with CUDA hidden and nonblocking file diagnostics."""

    command = [
        sys.executable,
        "-m",
        "tools.prepare_libero_diffusion_data",
        "--dataset-root",
        str(Path(dataset_root).resolve()),
        "--decode-rlds-to",
        str(Path(staging_dir).resolve()),
        "--max-per-instruction",
        str(max_per_instruction),
        "--smoke-episodes",
        str(smoke_episodes),
    ]
    if suite is not None:
        command.extend(
            [
                "--suite",
                suite,
                "--selected-start",
                str(selected_start),
                "--selected-count",
                str(selected_count),
            ]
        )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "decoder.stdout.log"
    stderr_path = run_dir / "decoder.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        return subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=environment,
        )


def finish_rlds_decoder_subprocess(
    child: object, stdout_path: Path, stderr_path: Path
) -> None:
    """Fail closed on a decoder error or native fault reported on stderr."""

    returncode = child.wait(timeout=DECODER_NORMAL_SHUTDOWN_TIMEOUT_SECONDS)
    stdout = Path(stdout_path).read_text(encoding="utf-8", errors="replace")
    stderr = Path(stderr_path).read_text(encoding="utf-8", errors="replace")
    diagnostics = f"decoder stdout:\n{stdout}\ndecoder stderr:\n{stderr}"
    if returncode != 0:
        raise RuntimeError(f"RLDS decoder failed with exit code {returncode}\n{diagnostics}")
    lowered = diagnostics.lower()
    if "windows fatal exception" in lowered or "access violation" in lowered:
        raise RuntimeError(f"fatal native runtime reported by RLDS decoder\n{diagnostics}")


def terminate_and_reap_decoder(child: object) -> None:
    """Stop a live decoder before its run directory is removed."""

    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=DECODER_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=DECODER_SHUTDOWN_TIMEOUT_SECONDS)


def cleanup_staging_run(run_dir: Path) -> bool:
    """Best-effort Windows cleanup with a retained diagnostic on repeated failure."""

    run_dir = Path(run_dir)
    for attempt in range(3):
        try:
            shutil.rmtree(run_dir)
            return True
        except FileNotFoundError:
            return True
        except OSError as error:
            if attempt == 2:
                try:
                    (run_dir / "cleanup_failure.txt").write_text(
                        f"cleanup failed after 3 attempts: {error}\n", encoding="utf-8"
                    )
                except OSError:
                    pass
                return False
            time.sleep(0.1 * (attempt + 1))
    return False


def publish_feature_cache(candidate_dir: Path, canonical_dir: Path) -> None:
    """Atomically replace the canonical cache only after a complete candidate."""

    candidate_dir = Path(candidate_dir)
    canonical_dir = Path(canonical_dir)
    if not (candidate_dir / "dataset_manifest.json").is_file():
        raise RuntimeError("candidate cache has no dataset_manifest.json")
    canonical_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = canonical_dir.parent / f".{canonical_dir.name}.backup-{uuid4().hex}"
    moved_existing = False
    if canonical_dir.exists():
        os.replace(canonical_dir, backup_dir)
        moved_existing = True
    try:
        os.replace(candidate_dir, canonical_dir)
    except OSError:
        if moved_existing and backup_dir.exists() and not canonical_dir.exists():
            os.replace(backup_dir, canonical_dir)
        raise
    if moved_existing and not cleanup_staging_run(backup_dir):
        raise RuntimeError(f"old cache retained after publish because cleanup failed: {backup_dir}")


def _cache_manifest_path(output_dir: Path) -> Path:
    return output_dir / "dataset_manifest.json"


def _assert_safe_cache_overwrite(
    output_dir: Path,
    source_commit: str | None,
    config: Mapping[str, object],
    segment: Mapping[str, object] | None = None,
) -> None:
    manifest_path = _cache_manifest_path(output_dir)
    if not manifest_path.exists():
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("source_commit") != source_commit:
        raise RuntimeError("refusing overwrite: source_commit differs from existing cache")
    if existing.get("config") != dict(config):
        raise RuntimeError("refusing overwrite: config differs from existing cache")
    expected_segment = dict(segment) if segment is not None else None
    if existing.get("segment") != expected_segment:
        raise RuntimeError("refusing overwrite: segment differs from existing cache")


def write_feature_cache(
    episodes: Iterable[EpisodeArrays],
    *,
    output_dir: Path,
    encoder: Callable[[np.ndarray], np.ndarray],
    source_commit: str | None,
    config: Mapping[str, object],
    dataset_validation: Mapping[str, object] | None = None,
    segment: Mapping[str, object] | None = None,
) -> dict:
    """Encode and persist one compressed, self-contained NPZ per episode."""

    output_dir = Path(output_dir)
    config = dict(config)
    feature_dim = int(config.get("feature_dim", 512))
    if feature_dim <= 0:
        raise ValueError("config feature_dim must be positive")
    _assert_safe_cache_overwrite(output_dir, source_commit, config, segment)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for cache_index, episode in enumerate(episodes):
        features = np.asarray(encoder(episode.images), dtype=np.float32)
        expected_shape = (episode.images.shape[0], feature_dim)
        if features.shape != expected_shape:
            raise ValueError(
                f"encoder returned {features.shape}; expected {expected_shape}"
            )
        filename = f"{cache_index:05d}_{episode.suite}_episode_{episode.episode_index:05d}.npz"
        relative_path = Path("episodes") / filename
        np.savez_compressed(
            output_dir / relative_path,
            features=features.astype(np.float16),
            states=np.asarray(episode.states, dtype=np.float32),
            actions=np.asarray(episode.actions, dtype=np.float32),
            instruction=np.asarray(episode.instruction),
            suite=np.asarray(episode.suite),
            episode_index=np.asarray(episode.episode_index, dtype=np.int64),
        )
        records.append(
            {
                "path": relative_path.as_posix(),
                "suite": episode.suite,
                "instruction": episode.instruction,
                "episode_index": episode.episode_index,
                "length": int(episode.images.shape[0]),
                "boundary": {"start": 0, "stop": int(episode.images.shape[0])},
            }
        )
    manifest = {
        "schema_version": 1,
        "source_commit": source_commit,
        "git_available": source_commit is not None,
        "config": config,
        "dataset_validation": dict(dataset_validation or {}),
        "episodes": records,
    }
    if segment is not None:
        manifest["segment"] = dict(segment)
    _cache_manifest_path(output_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_imagenet_resnet18_encoder(
    image_size: int, *, device_name: str = "cpu"
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the frozen ImageNet ResNet-18 feature encoder without downloading."""

    if image_size != 128:
        raise ValueError("LIBERO feature cache requires fixed image_size=128")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device_name must be 'cpu' or 'cuda'")
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint.is_file():
        raise RuntimeError(
            f"ImageNet ResNet18 weights are not cached at {checkpoint}; refusing download"
        )
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device(device_name)
    model.to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    def encode(images: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(images, dtype=np.uint8)).to(device)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            return model(tensor).cpu().numpy()

    return encode


def _source_commit(dataset_root: Path) -> tuple[str | None, bool]:
    dataset_root = Path(dataset_root).resolve()
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={dataset_root}",
                    "-C",
                    str(dataset_root),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            ).strip(),
            True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-per-instruction", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-episodes", type=int, default=0)
    parser.add_argument("--suite", choices=tuple(EXPECTED_TFRECORD_COUNTS))
    parser.add_argument("--selected-start", type=int, default=0)
    parser.add_argument("--selected-count", type=int)
    parser.add_argument("--decode-rlds-to", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke_episodes not in (0, 1):
        parser.error("--smoke-episodes supports only 0 or 1")
    if args.image_size != 128:
        parser.error("--image-size is fixed at 128 for the LIBERO feature cache")
    if args.selected_start < 0:
        parser.error("--selected-start must be nonnegative")
    if args.selected_count is not None and args.selected_count <= 0:
        parser.error("--selected-count must be positive")
    segmented = args.suite is not None or args.selected_count is not None or args.selected_start != 0
    if segmented and (args.suite is None or args.selected_count is None):
        parser.error("--suite and --selected-count are both required for segmented caching")
    segment = (
        {
            "suite": args.suite,
            "selected_start": args.selected_start,
            "selected_count": args.selected_count,
        }
        if segmented
        else None
    )
    if args.decode_rlds_to is not None:
        raw_episodes = iter_rlds_episodes(args.dataset_root, suite=args.suite)
        if segmented:
            decoded: Iterable[EpisodeArrays] = iter_selected_segment(
                raw_episodes,
                suite=args.suite,
                max_per_instruction=args.max_per_instruction,
                selected_start=args.selected_start,
                selected_count=args.selected_count,
            )
        else:
            decoded = iter_selected_episodes(raw_episodes, args.max_per_instruction)
        if args.smoke_episodes:
            decoded = islice(decoded, args.smoke_episodes)
        count = write_rlds_staging(decoded, args.decode_rlds_to, segment=segment)
        print(json.dumps({"staging_dir": str(args.decode_rlds_to), "episodes": count}))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --decode-rlds-to is used")
    validation = validate_expanded_tfrecords(args.dataset_root)
    config = {
        "encoder": "resnet18_imagenet1k_v1",
        "feature_dim": 512,
        "image_size": args.image_size,
        "max_per_instruction": args.max_per_instruction,
        "split": "train",
        "shuffle_files": False,
        "view": "observation.image static third-person",
    }
    source_commit, _git_available = _source_commit(args.dataset_root)
    cache_dir = args.output_dir / "feature_cache"
    _assert_safe_cache_overwrite(cache_dir, source_commit, config, segment)
    run_dir = args.output_dir / ".feature_cache_runs" / uuid4().hex
    staging_dir = run_dir / "staging"
    candidate_cache_dir = run_dir / "feature_cache"
    run_dir.mkdir(parents=True, exist_ok=False)
    published = False
    child: object | None = None
    try:
        decoder_arguments = {}
        if segmented:
            decoder_arguments = {
                "suite": args.suite,
                "selected_start": args.selected_start,
                "selected_count": args.selected_count,
            }
        child = start_rlds_decoder_subprocess(
            args.dataset_root,
            staging_dir,
            args.max_per_instruction,
            args.smoke_episodes,
            run_dir,
            **decoder_arguments,
        )
        manifest = write_feature_cache(
            iter_live_staged_episodes(
                staging_dir, child=child, expected_segment=segment
            ),
            output_dir=candidate_cache_dir,
            encoder=build_imagenet_resnet18_encoder(
                args.image_size, device_name=args.device
            ),
            source_commit=source_commit,
            config=config,
            dataset_validation=validation,
            segment=segment,
        )
        finish_rlds_decoder_subprocess(
            child,
            run_dir / "decoder.stdout.log",
            run_dir / "decoder.stderr.log",
        )
        publish_feature_cache(candidate_cache_dir, cache_dir)
        published = True
    except BaseException:
        if child is not None:
            terminate_and_reap_decoder(child)
        raise
    finally:
        if not cleanup_staging_run(run_dir) and not published:
            print(f"staging cleanup failed; retained diagnostics at {run_dir}", file=sys.stderr)
    print(json.dumps({"output_dir": str(cache_dir), "episodes": len(manifest["episodes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
