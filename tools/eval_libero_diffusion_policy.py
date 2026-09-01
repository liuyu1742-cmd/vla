"""Recoverable, pure-policy evaluation for compact LIBERO diffusion policies.

The module intentionally keeps LIBERO imports at the CLI boundary so its rollout
contract can be tested with small deterministic environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from tools.libero_diffusion_contracts import ActionNormalizer, dataset_gripper_to_env
from tools.libero_diffusion_model import CompactDiffusionPolicy, DiffusionPolicyConfig
from tools.prepare_libero_diffusion_data import build_imagenet_resnet18_encoder


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "third_party" / "LIBERO"
LIBERO_CONFIG_DIR = REPO_ROOT / "outputs" / "libero_config"
NUMBA_CACHE_DIR = REPO_ROOT / "outputs" / "numba_cache"


def _normalized_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _path_is_within(path: str | os.PathLike[str], root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _assert_vendored_libero_module(module: Any) -> None:
    imported_file = getattr(module, "__file__", None)
    if imported_file is not None:
        if not _path_is_within(imported_file, LIBERO_ROOT):
            raise RuntimeError(f"foreign libero module already imported: {imported_file}")
        return
    namespace_paths = list(getattr(module, "__path__", []))
    if not namespace_paths or any(
        not _path_is_within(path, LIBERO_ROOT) for path in namespace_paths
    ):
        raise RuntimeError("foreign libero module already imported without provenance")


def _configure_libero() -> None:
    """Use the validated vendored LIBERO checkout and its local config."""

    package_init = LIBERO_ROOT / "libero" / "libero" / "__init__.py"
    config_file = LIBERO_CONFIG_DIR / "config.yaml"
    if not package_init.is_file():
        raise FileNotFoundError(f"missing vendored LIBERO package: {package_init}")
    if not config_file.is_file():
        raise FileNotFoundError(f"missing validated LIBERO config: {config_file}")

    NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)

    imported = sys.modules.get("libero")
    if imported is not None:
        _assert_vendored_libero_module(imported)
    imported_inner = sys.modules.get("libero.libero")
    if imported_inner is not None:
        _assert_vendored_libero_module(imported_inner)
        cached_config = getattr(imported_inner, "config_file", None)
        if cached_config is None or _normalized_path(cached_config) != _normalized_path(
            config_file
        ):
            raise RuntimeError(
                f"cached foreign LIBERO config: {cached_config!s}; expected {config_file}"
            )

    os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_DIR)
    libero_root = str(LIBERO_ROOT)
    normalized_root = _normalized_path(LIBERO_ROOT)
    sys.path[:] = [
        entry for entry in sys.path if _normalized_path(entry or os.curdir) != normalized_root
    ]
    sys.path.insert(0, libero_root)


SUITE_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
SETTLE_STEPS = 10
NOOP_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


@dataclass
class PolicyState:
    """History and open-loop action queue for one LIBERO episode."""

    obs_horizon: int = 2
    visual_history: deque[np.ndarray] = field(init=False)
    proprio_history: deque[np.ndarray] = field(init=False)
    action_queue: deque[np.ndarray] = field(default_factory=deque)
    policy_queries: int = 0

    def __post_init__(self) -> None:
        if self.obs_horizon <= 0:
            raise ValueError("obs_horizon must be positive")
        self.visual_history = deque(maxlen=self.obs_horizon)
        self.proprio_history = deque(maxlen=self.obs_horizon)


def _padded_history(history: deque[np.ndarray], horizon: int) -> np.ndarray:
    rows = list(history)
    if not rows:
        raise RuntimeError("observation history is empty")
    while len(rows) < horizon:
        rows.insert(0, rows[0])
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def policy_action(
    policy: Any,
    state: PolicyState,
    *,
    visual_feature: np.ndarray,
    proprio: np.ndarray,
    instruction: str,
    normalizer: ActionNormalizer,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Return one environment action, querying a new eight-step chunk as needed."""

    state.visual_history.append(np.asarray(visual_feature, dtype=np.float32))
    state.proprio_history.append(np.asarray(proprio, dtype=np.float32))
    if not state.action_queue:
        condition = {
            "visual": torch.from_numpy(_padded_history(state.visual_history, state.obs_horizon))
            .unsqueeze(0)
            .to(device),
            "state": torch.from_numpy(_padded_history(state.proprio_history, state.obs_horizon))
            .unsqueeze(0)
            .to(device),
            "instructions": [instruction],
        }
        sampled = policy.sample_actions(condition, generator=generator)
        decoded = normalizer.denormalize(sampled.detach().float().cpu().numpy()[0])
        if decoded.shape[0] != 8:
            raise RuntimeError("policy must emit exactly eight actions per query")
        state.action_queue.extend(np.asarray(action, dtype=np.float32) for action in decoded)
        state.policy_queries += 1

    action = state.action_queue.popleft().copy()
    action[-1] = dataset_gripper_to_env(np.asarray([action[-1]]))[0]
    return action


def policy_action_chunk(*args: Any, **kwargs: Any) -> np.ndarray:
    """Compatibility name for the one-action chunk consumer."""

    return policy_action(*args, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _episode_path(output_dir: Path, task_id: int, trial_index: int) -> Path:
    return output_dir / "episodes" / f"task_{task_id:03d}_trial_{trial_index:03d}.json"


def _completed_episode(
    path: Path, *, suite_name: str, checkpoint_sha256: str, seed: int, task_id: int, trial_index: int
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    matches = (
        payload.get("suite") == suite_name
        and payload.get("checkpoint_sha256") == checkpoint_sha256
        and payload.get("seed") == seed
        and payload.get("task_id") == task_id
        and payload.get("trial_index") == trial_index
        and isinstance(payload.get("success"), bool)
    )
    return payload if matches else None


def _write_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _write_video(path: Path, images: Sequence[np.ndarray]) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=30) as writer:
        for image in images:
            writer.append_data(np.asarray(image, dtype=np.uint8))


def _save_media(
    episode_stem: Path,
    frames: Sequence[np.ndarray],
    *,
    write_video: bool,
    video_writer: Callable[[Path, Sequence[np.ndarray]], None] | None,
) -> str | None:
    if not frames:
        return "no policy frames were captured"
    errors: list[str] = []
    try:
        for label, index in (("first", 0), ("middle", len(frames) // 2), ("final", len(frames) - 1)):
            _write_png(episode_stem.with_name(f"{episode_stem.name}_{label}.png"), frames[index])
    except Exception as error:  # evidence failure is recorded, not silently hidden
        errors.append(f"PNG: {error}")
    if write_video:
        try:
            (video_writer or _write_video)(episode_stem.with_suffix(".mp4"), frames)
        except Exception as error:  # video is explicitly best-effort
            errors.append(f"MP4: {error}")
    return "; ".join(errors) if errors else None


def _default_observation_adapter(observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Match the LIBERO agent-view rotation and proprio convention used by evaluation."""

    image = np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, ::-1])
    if "state" in observation:
        return image, np.asarray(observation["state"], dtype=np.float32)
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float32).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(1.0 - float(quaternion[3]) ** 2)
    axis_angle = (
        np.zeros(3, dtype=np.float32)
        if math.isclose(denominator, 0.0)
        else quaternion[:3] * (2.0 * math.acos(float(quaternion[3]))) / denominator
    )

    return image, np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            axis_angle,
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


def _run_episode(
    *,
    env: Any,
    instruction: str,
    initial_state: Any,
    policy: Any,
    normalizer: ActionNormalizer,
    encoder: Callable[[np.ndarray], np.ndarray],
    observation_adapter: Callable[[Mapping[str, Any]], tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    generator: torch.Generator,
    horizon: int,
    settle_steps: int,
) -> tuple[bool, int, float, str | None, dict[str, Any], list[np.ndarray]]:
    frames: list[np.ndarray] = []
    state = PolicyState()
    latencies: list[float] = []
    steps = 0
    try:
        observation = env.reset()
        if initial_state is not None:
            observation = env.set_init_state(initial_state)
        for _ in range(settle_steps):
            observation, _reward, done, _info = env.step(NOOP_ACTION.tolist())
            if bool(done):
                return True, steps, 0.0, None, {"policy_queries": 0, "actions_executed": 0}, frames
        for _ in range(horizon):
            image, proprio = observation_adapter(observation)
            frames.append(np.asarray(image, dtype=np.uint8))
            began = time.perf_counter()
            feature = np.asarray(encoder(np.asarray(image)[None, ...]), dtype=np.float32)[0]
            action = policy_action(
                policy, state, visual_feature=feature, proprio=proprio, instruction=instruction,
                normalizer=normalizer, device=device, generator=generator,
            )
            latencies.append(time.perf_counter() - began)
            observation, _reward, done, _info = env.step(action.tolist())
            steps += 1
            if bool(done):
                return True, steps, float(sum(latencies)), None, {
                    "policy_queries": state.policy_queries, "actions_executed": steps,
                    "last_action": action.tolist(),
                }, frames
    except Exception as error:
        return False, steps, float(sum(latencies)), f"{type(error).__name__}: {error}", {
            "policy_queries": state.policy_queries, "actions_executed": steps,
        }, frames
    return False, steps, float(sum(latencies)), None, {
        "policy_queries": state.policy_queries, "actions_executed": steps,
    }, frames


def evaluate_suite(
    *,
    suite_name: str,
    task_suite: Any,
    policy: Any,
    normalizer: ActionNormalizer,
    encoder: Callable[[np.ndarray], np.ndarray],
    checkpoint_sha256: str,
    seed: int,
    output_dir: Path,
    trials_per_task: int,
    task_ids: Sequence[int] | None = None,
    env_factory: Callable[[Any], Any] | None = None,
    observation_adapter: Callable[[Mapping[str, Any]], tuple[np.ndarray, np.ndarray]] = _default_observation_adapter,
    device: torch.device | str = "cpu",
    horizon: int | None = None,
    settle_steps: int = SETTLE_STEPS,
    write_video: bool = True,
    video_writer: Callable[[Path, Sequence[np.ndarray]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate requested official task initial states and atomically persist each result."""

    if suite_name not in SUITE_HORIZONS:
        raise ValueError(f"unsupported LIBERO suite: {suite_name}")
    if trials_per_task <= 0 or settle_steps < 0:
        raise ValueError("trials_per_task must be positive and settle_steps cannot be negative")
    device = torch.device(device)
    if env_factory is None:
        env_factory = _libero_env_factory
    output_dir = Path(output_dir)
    selected_task_ids = list(range(int(task_suite.n_tasks))) if task_ids is None else [int(value) for value in task_ids]
    rows: list[dict[str, Any]] = []
    resumed = 0
    for task_id in selected_task_ids:
        task = task_suite.get_task(task_id)
        instruction = str(task.language)
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) < trials_per_task:
            raise RuntimeError(f"task {task_id} has fewer official init states than requested trials")
        pending_trials: list[tuple[int, Path, int]] = []
        for trial_index in range(trials_per_task):
            path = _episode_path(output_dir, task_id, trial_index)
            prior = _completed_episode(path, suite_name=suite_name, checkpoint_sha256=checkpoint_sha256,
                seed=seed, task_id=task_id, trial_index=trial_index)
            if prior is not None:
                rows.append(prior)
                resumed += 1
                continue
            episode_seed = int(seed) + task_id * 100_000 + trial_index
            pending_trials.append((trial_index, path, episode_seed))
        if not pending_trials:
            continue
        try:
            env = env_factory(task)
        except Exception as error:
            for trial_index, path, episode_seed in pending_trials:
                row = {
                    "suite": suite_name,
                    "task_id": task_id,
                    "trial_index": trial_index,
                    "instruction": instruction,
                    "initial_state_index": trial_index,
                    "checkpoint_sha256": checkpoint_sha256,
                    "seed": int(seed),
                    "episode_seed": episode_seed,
                    "success": False,
                    "steps": 0,
                    "inference_latency_seconds": 0.0,
                    "error": f"{type(error).__name__}: {error}",
                    "action_summary": {"policy_queries": 0, "actions_executed": 0},
                    "media_error": "environment construction failed before frame capture",
                }
                _atomic_json(path, row)
                rows.append(row)
            continue
        try:
            for trial_index, path, episode_seed in pending_trials:
                generator = torch.Generator(device=device.type).manual_seed(episode_seed)
                success, steps, latency, error, action_summary, frames = _run_episode(
                    env=env, instruction=instruction, initial_state=initial_states[trial_index], policy=policy,
                    normalizer=normalizer, encoder=encoder, observation_adapter=observation_adapter, device=device,
                    generator=generator, horizon=horizon if horizon is not None else SUITE_HORIZONS[suite_name],
                    settle_steps=settle_steps,
                )
                row = {
                    "suite": suite_name, "task_id": task_id, "trial_index": trial_index,
                    "instruction": instruction, "initial_state_index": trial_index,
                    "checkpoint_sha256": checkpoint_sha256, "seed": int(seed), "episode_seed": episode_seed,
                    "success": bool(success), "steps": steps, "inference_latency_seconds": latency,
                    "error": error, "action_summary": action_summary, "media_error": None,
                }
                _atomic_json(path, row)
                row["media_error"] = _save_media(
                    path.with_suffix(""), frames, write_video=write_video, video_writer=video_writer
                )
                _atomic_json(path, row)
                rows.append(row)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    successes = sum(bool(row["success"]) for row in rows)
    summary = {
        "suite": suite_name, "checkpoint_sha256": checkpoint_sha256, "seed": int(seed),
        "episodes": len(rows), "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "resumed_episodes": resumed, "episode_results": rows,
        "git_available": False,
    }
    _atomic_json(output_dir / "suite_summary.json", summary)
    return summary


def load_policy_checkpoint(path: Path, *, device: torch.device | str = "cpu") -> tuple[CompactDiffusionPolicy, ActionNormalizer, str]:
    """Load the model/config/normalizer required for a self-contained evaluation."""

    device = torch.device(device)
    payload = torch.load(Path(path), map_location=device)
    for key in ("model", "model_config", "normalizer"):
        if key not in payload:
            raise RuntimeError(f"checkpoint is missing required {key!r}")
    if payload["model_config"].get("obs_horizon") != 2:
        raise RuntimeError("checkpoint model_config obs_horizon must equal 2")
    if payload["model_config"].get("action_horizon") != 8:
        raise RuntimeError("checkpoint model_config action_horizon must equal 8")
    config = DiffusionPolicyConfig(**payload["model_config"])
    model = CompactDiffusionPolicy(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    normalizer_data = payload["normalizer"]
    normalizer = ActionNormalizer(q01=np.asarray(normalizer_data["q01"], dtype=np.float32),
                                  q99=np.asarray(normalizer_data["q99"], dtype=np.float32))
    return model, normalizer, _sha256(Path(path))


def _libero_env_factory(task: Any) -> Any:
    """Construct LIBERO's official off-screen environment with a fixed env seed."""

    _configure_libero()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    _configure_libero()

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    return env


def _load_task_suite(name: str) -> Any:
    _configure_libero()
    from libero.libero import benchmark
    _configure_libero()

    return benchmark.get_benchmark_dict()[name]()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a compact diffusion policy on pure LIBERO rollouts.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=sorted(SUITE_HORIZONS), required=True)
    parser.add_argument("--task-ids", default=None, help="comma-separated official task ids (default: all)")
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    task_ids = None if args.task_ids is None else [int(value) for value in args.task_ids.split(",") if value]
    policy, normalizer, checkpoint_sha256 = load_policy_checkpoint(args.checkpoint, device=args.device)
    encoder = build_imagenet_resnet18_encoder(128, device_name=args.device)
    evaluation_dir = args.output_dir / args.suite / f"seed_{args.seed}"
    summary = evaluate_suite(suite_name=args.suite, task_suite=_load_task_suite(args.suite), policy=policy,
        normalizer=normalizer, encoder=encoder, checkpoint_sha256=checkpoint_sha256, seed=args.seed,
        output_dir=evaluation_dir, trials_per_task=args.trials_per_task, task_ids=task_ids, device=args.device,
        write_video=not args.no_video)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
