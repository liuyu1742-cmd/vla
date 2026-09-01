"""Resume-safe, per-episode evaluation for the four local OpenVLA baselines.

This wrapper intentionally mirrors the verified direct OpenVLA LIBERO path in
``openvla_libero_goal_pure_eval`` while adding durable episode evidence.  It
does not use expert recovery or alter LIBERO's fixed initial states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINTS = {
    "libero_spatial": REPO_ROOT / "models" / "openvla-7b-finetuned-libero-spatial-gitcode",
    "libero_object": REPO_ROOT / "models" / "openvla-7b-finetuned-libero-object",
    "libero_goal": REPO_ROOT / "models" / "openvla-7b-finetuned-libero-goal",
    "libero_10": REPO_ROOT / "models" / "openvla-7b-finetuned-libero-10-gitcode",
}
OFFICIAL_HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
EPISODE_SCHEMA = "openvla_baseline_libero_episode_v1"


class ResumeMismatchError(RuntimeError):
    """An existing episode belongs to another checkpoint or protocol."""


@dataclass
class EvalDependencies:
    set_seed: Callable[[int], None]
    load_suite: Callable[[str], Any]
    load_model: Callable[[Path], tuple[Any, Any, dict[str, Any]]]
    create_env: Callable[[Any], Any]
    preprocess_image: Callable[[dict[str, Any]], np.ndarray]
    predict_action: Callable[[Any, Any, np.ndarray, str, str], np.ndarray]
    save_image: Callable[[Path, np.ndarray], None]
    save_video: Callable[[Path, list[np.ndarray]], None]

    @classmethod
    def from_object(cls, source: Any) -> "EvalDependencies":
        return cls(**{name: getattr(source, name) for name in cls.__dataclass_fields__})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_task_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task ids must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("task ids must be unique non-negative integers")
    return result


def parse_positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def checkpoint_sha256(checkpoint: Path) -> str:
    """Hash checkpoint contents and relative names so a resume cannot mix weights."""
    digest = hashlib.sha256()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        digest.update(path.relative_to(checkpoint).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _real_dependencies() -> EvalDependencies:
    from tools import openvla_libero_goal_pure_eval as pure

    def load_suite(name: str):
        pure._configure_libero()
        from libero.libero import benchmark

        return benchmark.get_benchmark_dict()[name]()

    def load_model(checkpoint: Path):
        model, processor = pure._load_model(checkpoint)
        return model, processor, model.norm_stats

    def create_env(task):
        return pure._environment_for_task(task)

    def save_image(path: Path, image: np.ndarray) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)

    def save_video(path: Path, frames: list[np.ndarray]) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(path, fps=30) as writer:
            for frame in frames:
                writer.append_data(frame)

    return EvalDependencies(
        set_seed=pure._set_seed,
        load_suite=load_suite,
        load_model=load_model,
        create_env=create_env,
        preprocess_image=pure._libero_image,
        predict_action=pure._predict_openvla_action,
        save_image=save_image,
        save_video=save_video,
    )


def _episode_path(run_dir: Path, task_id: int, trial: int) -> Path:
    return run_dir / "episodes" / f"task_{task_id:03d}" / f"trial_{trial:03d}.json"


def _identity(
    suite: str,
    seed: int,
    checkpoint: Path,
    sha256: str | None,
    task_id: int,
    trial: int,
    horizon: int,
    num_steps_wait: int,
    *,
    max_episode_steps: int | None = None,
    effective_horizon: int | None = None,
    checkpoint_hash_status: str = "available",
) -> dict[str, Any]:
    return {
        "schema_version": EPISODE_SCHEMA,
        "suite": suite,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256,
        "checkpoint_hash_status": checkpoint_hash_status,
        "task_id": task_id,
        "trial": trial,
        "initial_state_index": trial,
        "max_episode_steps": max_episode_steps,
        "effective_horizon": effective_horizon if effective_horizon is not None else horizon,
        "horizon": horizon,
        "num_steps_wait": num_steps_wait,
        "uses_expert_recovery": False,
    }


def _existing_completed(path: Path, identity: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResumeMismatchError(f"invalid existing episode JSON: {path}") from exc
    legacy_horizon_identity = "max_episode_steps" not in saved and "effective_horizon" not in saved
    if legacy_horizon_identity and identity.get("max_episode_steps") is None:
        if saved.get("horizon") != identity.get("effective_horizon"):
            raise ResumeMismatchError(f"{path}: effective_horizon does not match current evaluation")
        saved = {**saved, "max_episode_steps": None, "effective_horizon": saved["horizon"]}
    for field, expected in identity.items():
        if field not in saved or saved[field] != expected:
            raise ResumeMismatchError(f"{path}: {field} does not match current evaluation")
    return saved.get("status") in {"completed", "error"}


def _base_episode(identity: dict[str, Any], *, horizon: int, num_steps_wait: int, task_description: str | None) -> dict[str, Any]:
    return {
        **identity,
        "started_at": utc_now(),
        "task_description": task_description,
        "done": False,
        "success": False,
        "decision_steps": 0,
        "error": None,
        "frame_paths": [],
        "video_path": None,
        "visualization_errors": [],
        "diagnostics": [],
        "checkpoint_hash_error": None,
    }


def _save_visuals(episode: dict[str, Any], frames: list[np.ndarray], episode_dir: Path, deps: EvalDependencies) -> None:
    if not frames:
        return
    selected = (("first", frames[0]), ("middle", frames[len(frames) // 2]), ("last", frames[-1]))
    for label, frame in selected:
        path = episode_dir / f"{label}.png"
        try:
            deps.save_image(path, frame)
            episode["frame_paths"].append(str(path))
        except Exception as exc:  # visual evidence must not change success semantics
            episode["visualization_errors"].append(f"PNG {label}: {type(exc).__name__}: {exc}")
    mp4_path = episode_dir / "rollout.mp4"
    try:
        deps.save_video(mp4_path, frames)
        episode["video_path"] = str(mp4_path)
    except Exception as exc:  # video is best effort only
        episode["visualization_errors"].append(f"MP4: {type(exc).__name__}: {exc}")


def _persist_error_rows(
    run_dir: Path,
    identities: list[dict[str, Any]],
    horizon: int,
    num_steps_wait: int,
    error: Exception,
    *,
    stage: str,
    additional_fields: dict[str, Any] | None = None,
) -> None:
    message = f"{type(error).__name__}: {error}"
    for identity in identities:
        path = _episode_path(run_dir, identity["task_id"], identity["trial"])
        if _existing_completed(path, identity):
            continue
        episode = _base_episode(identity, horizon=horizon, num_steps_wait=num_steps_wait, task_description=None)
        episode.update({"status": "error", "error": message, "error_stage": stage, "finished_at": utc_now()})
        if additional_fields:
            episode.update(additional_fields)
        _atomic_json(path, episode)


def _append_diagnostic(rows: list[tuple[dict[str, Any], Path]], message: str) -> None:
    for _, path in rows:
        if not path.exists():
            continue
        episode = json.loads(path.read_text(encoding="utf-8"))
        episode.setdefault("diagnostics", []).append(message)
        _atomic_json(path, episode)


def evaluate_suite(
    *, suite: str, checkpoint: Path, output_dir: Path, task_ids: list[int] | None = None, trials_per_task: int = 1,
    seed: int = 7, num_steps_wait: int = 10, max_episode_steps: int | None = None,
    dependencies: EvalDependencies | None = None,
) -> dict[str, Any]:
    """Evaluate a suite once per checkpoint, atomically persisting every episode."""
    if suite not in BASELINE_CHECKPOINTS:
        raise ValueError(f"unsupported suite: {suite}")
    if trials_per_task < 1 or num_steps_wait < 0:
        raise ValueError("trials_per_task must be positive and num_steps_wait non-negative")
    if max_episode_steps is not None and max_episode_steps < 1:
        raise ValueError("max_episode_steps must be a positive integer")
    if task_ids is not None and len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task IDs are not allowed")
    if task_ids is not None and any(task_id < 0 for task_id in task_ids):
        raise ValueError("task IDs must be non-negative")
    checkpoint = Path(checkpoint)
    suite_horizon = OFFICIAL_HORIZONS[suite]
    effective_horizon = min(suite_horizon, max_episode_steps) if max_episode_steps is not None else suite_horizon
    horizon = effective_horizon
    run_dir = Path(output_dir) / suite / f"seed_{seed}"
    deps = dependencies or _real_dependencies()
    # Every formal LIBERO suite here contains ten tasks.  Establish this before
    # opening LIBERO so a suite-construction error is still per-episode evidence.
    requested_ids = list(task_ids) if task_ids is not None else list(range(10))

    try:
        sha256 = checkpoint_sha256(checkpoint)
    except Exception as exc:
        identities = [
            _identity(
                suite,
                seed,
                checkpoint,
                None,
                task_id,
                trial,
                horizon,
                num_steps_wait,
                max_episode_steps=max_episode_steps,
                effective_horizon=effective_horizon,
                checkpoint_hash_status="unavailable",
            )
            for task_id in requested_ids
            for trial in range(trials_per_task)
        ]
        _persist_error_rows(
            run_dir,
            identities,
            horizon,
            num_steps_wait,
            exc,
            stage="checkpoint_sha256",
            additional_fields={"checkpoint_hash_error": f"{type(exc).__name__}: {exc}"},
        )
        return _write_summary(
            run_dir, suite, checkpoint, None, seed, requested_ids, trials_per_task,
            max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
        )

    identities = [
        _identity(
            suite, seed, checkpoint, sha256, task_id, trial, horizon, num_steps_wait,
            max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
        )
        for task_id in requested_ids
        for trial in range(trials_per_task)
    ]
    pending = [
        (identity, _episode_path(run_dir, identity["task_id"], identity["trial"]))
        for identity in identities
        if not _existing_completed(
            _episode_path(run_dir, identity["task_id"], identity["trial"]), identity
        )
    ]
    skipped_episodes = len(identities) - len(pending)
    if not pending:
        return _write_summary(
            run_dir,
            suite,
            checkpoint,
            sha256,
            seed,
            requested_ids,
            trials_per_task,
            max_episode_steps=max_episode_steps,
            effective_horizon=effective_horizon,
            skipped_episodes=skipped_episodes,
        )

    try:
        deps.set_seed(seed)
    except Exception as exc:
        _persist_error_rows(
            run_dir,
            [identity for identity, _ in pending],
            horizon,
            num_steps_wait,
            exc,
            stage="set_seed",
        )
        return _write_summary(
            run_dir, suite, checkpoint, sha256, seed, requested_ids, trials_per_task,
            max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
            skipped_episodes=skipped_episodes,
        )

    try:
        task_suite = deps.load_suite(suite)
    except Exception as exc:
        _persist_error_rows(
            run_dir,
            [identity for identity, _ in pending],
            horizon,
            num_steps_wait,
            exc,
            stage="load_suite",
        )
        return _write_summary(
            run_dir, suite, checkpoint, sha256, seed, requested_ids, trials_per_task,
            max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
            skipped_episodes=skipped_episodes,
        )

    if any(task_id >= task_suite.n_tasks for task_id in requested_ids):
        exc = ValueError(f"task IDs must be smaller than {task_suite.n_tasks}")
        _persist_error_rows(
            run_dir,
            [identity for identity, _ in pending],
            horizon,
            num_steps_wait,
            exc,
            stage="validate_task_ids",
        )
        return _write_summary(
            run_dir, suite, checkpoint, sha256, seed, requested_ids, trials_per_task,
            max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
            skipped_episodes=skipped_episodes,
        )

    task_contexts: list[tuple[int, Any, str, Any, list[tuple[dict[str, Any], Path]]]] = []
    for task_id in requested_ids:
        task_pending = [(identity, path) for identity, path in pending if identity["task_id"] == task_id]
        if not task_pending:
            continue
        try:
            task = task_suite.get_task(task_id)
            description = task.language
            initial_states = task_suite.get_task_init_states(task_id)
            if trials_per_task > len(initial_states):
                raise ValueError(f"task {task_id} exposes only {len(initial_states)} initial states")
        except Exception as exc:
            _persist_error_rows(
                run_dir,
                [identity for identity, _ in task_pending],
                horizon,
                num_steps_wait,
                exc,
                stage="load_task_metadata",
            )
            continue
        task_contexts.append((task_id, task, description, initial_states, task_pending))

    if not task_contexts:
        return _write_summary(
            run_dir, suite, checkpoint, sha256, seed, requested_ids, trials_per_task,
            max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
            skipped_episodes=skipped_episodes,
        )

    try:
        model, processor, norm_stats = deps.load_model(checkpoint)
        if suite not in norm_stats:
            raise KeyError(f"missing action normalization key: {suite}")
    except Exception as exc:
        viable = [identity for _, _, _, _, rows in task_contexts for identity, _ in rows]
        _persist_error_rows(run_dir, viable, horizon, num_steps_wait, exc, stage="load_model")
        return _write_summary(
            run_dir, suite, checkpoint, sha256, seed, requested_ids, trials_per_task,
            max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
            skipped_episodes=skipped_episodes,
        )

    for task_id, task, description, initial_states, task_pending in task_contexts:
        env = None
        try:
            env = deps.create_env(task)
            for identity, path in task_pending:
                episode = _base_episode(identity, horizon=horizon, num_steps_wait=num_steps_wait, task_description=description)
                frames: list[np.ndarray] = []
                try:
                    env.reset()
                    observation = env.set_init_state(initial_states[identity["trial"]])
                    policy_done = False
                    for step in range(horizon + num_steps_wait):
                        if step < num_steps_wait:
                            # Match the official evaluator: always complete the
                            # warm-up and ignore terminal signals from dummy actions.
                            observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                            continue
                        image = deps.preprocess_image(observation)
                        frames.append(image)
                        action = np.asarray(deps.predict_action(model, processor, image, description, suite), dtype=float).copy()
                        if action.shape != (7,):
                            raise ValueError(f"unexpected OpenVLA action shape: {action.shape}")
                        action[-1] = np.sign(2.0 * action[-1] - 1.0) * -1.0
                        episode["decision_steps"] += 1
                        observation, _, done, _ = env.step(action.tolist())
                        policy_done = bool(done)
                        if policy_done:
                            break
                    episode["done"] = policy_done
                    episode["success"] = policy_done
                    episode["status"] = "completed"
                except Exception as exc:
                    episode.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                _save_visuals(episode, frames, path.parent, deps)
                episode["finished_at"] = utc_now()
                _atomic_json(path, episode)
        except Exception as exc:
            _persist_error_rows(
                run_dir,
                [identity for identity, _ in task_pending],
                horizon,
                num_steps_wait,
                exc,
                stage="create_environment",
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    message = f"environment close: {type(exc).__name__}: {exc}"
                    _append_diagnostic(task_pending, message)
    return _write_summary(
        run_dir, suite, checkpoint, sha256, seed, requested_ids, trials_per_task,
        max_episode_steps=max_episode_steps, effective_horizon=effective_horizon,
        skipped_episodes=skipped_episodes,
    )


def _write_summary(
    run_dir: Path,
    suite: str,
    checkpoint: Path,
    sha256: str | None,
    seed: int,
    task_ids: list[int],
    trials_per_task: int,
    *,
    max_episode_steps: int | None = None,
    effective_horizon: int | None = None,
    skipped_episodes: int = 0,
) -> dict[str, Any]:
    rows = []
    for task_id in task_ids:
        for trial in range(trials_per_task):
            path = _episode_path(run_dir, task_id, trial)
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
        and isinstance(row.get("done"), bool)
        and isinstance(row.get("success"), bool)
    ]
    errors = [row for row in rows if row.get("status") == "error"]
    requested_episodes = len(task_ids) * trials_per_task
    integrity_status = (
        "valid"
        if len(completed) == requested_episodes and not errors and len(rows) == requested_episodes
        else "invalid"
    )
    successes = sum(row["success"] for row in completed)
    diagnostics = [item for row in rows for item in row.get("diagnostics", [])]
    summary = {
        "schema_version": "openvla_baseline_libero_suite_v1", "suite": suite, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256, "seed": seed, "task_ids": task_ids, "trials_per_task": trials_per_task,
        "max_episode_steps": max_episode_steps, "effective_horizon": effective_horizon,
        "checkpoint_hash_status": "available" if sha256 is not None else "unavailable",
        "uses_expert_recovery": False, "requested_episodes": requested_episodes,
        "completed_episodes": len(completed), "error_episodes": len(errors), "skipped_episodes": skipped_episodes,
        "successes": successes,
        "success_rate": (successes / len(completed)) if integrity_status == "valid" else None,
        "integrity_status": integrity_status,
        "diagnostics": diagnostics,
        "episode_paths": [str(_episode_path(run_dir, row["task_id"], row["trial"])) for row in rows], "finished_at": utc_now(),
    }
    _atomic_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=BASELINE_CHECKPOINTS, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the audited local baseline checkpoint for --suite.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=parse_task_ids, default=None, help="Comma-separated task IDs; default: entire suite.")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=parse_positive_int)
    args = parser.parse_args()
    summary = evaluate_suite(suite=args.suite, checkpoint=args.checkpoint or BASELINE_CHECKPOINTS[args.suite], output_dir=args.output_dir, task_ids=args.task_ids, trials_per_task=args.trials_per_task, seed=args.seed, num_steps_wait=args.num_steps_wait, max_episode_steps=args.max_episode_steps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
