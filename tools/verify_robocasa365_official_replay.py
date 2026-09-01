"""Replay an official RoboCasa365 episode and record simulator evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def validate_episode_contract(
    states: np.ndarray, actions: np.ndarray, recorded_rewards: np.ndarray
) -> None:
    """Reject incomplete or non-successful public episodes before simulation."""
    if len(states) != len(actions) or len(actions) != len(recorded_rewards):
        raise ValueError(
            "episode length mismatch: "
            f"states={len(states)} actions={len(actions)} rewards={len(recorded_rewards)}"
        )
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError(f"expected 12D RoboCasa actions, got {actions.shape}")
    if not np.any(np.asarray(recorded_rewards) > 0):
        raise ValueError("public episode has no successful frame")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_rgb(environment, camera: str, size: int = 256) -> np.ndarray:
    return np.asarray(
        environment.sim.render(height=size, width=size, camera_name=camera)[::-1],
        dtype=np.uint8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--selection-id", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-skip", type=int, default=2)
    args = parser.parse_args()
    if args.video_skip < 1:
        raise ValueError("video-skip must be positive")

    sys.path[:0] = [
        str(ROOT / "third_party/robosuite"),
        str(ROOT / "third_party/robocasa"),
    ]
    import imageio.v2 as imageio
    import pandas as pd
    import robosuite
    from PIL import Image
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    import robocasa.utils.lerobot_utils as LU

    dataset = args.dataset.resolve()
    episode = args.episode_index
    parquet = next(dataset.glob(f"data/*/episode_{episode:06d}.parquet"))
    frame_data = pd.read_parquet(parquet)
    rewards = np.asarray(frame_data["next.reward"], dtype=np.float32)
    states = LU.get_episode_states(dataset, episode)
    actions = LU.get_episode_actions(dataset, episode)
    validate_episode_contract(states, actions, rewards)
    episode_meta = LU.get_episode_meta(dataset, episode)

    env_meta = LU.get_env_metadata(dataset)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs.update(
        env_name=env_meta["env_name"],
        has_renderer=False,
        renderer="mjviewer",
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    environment = robosuite.make(**env_kwargs)
    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / "rollout.mp4"
    first_path = args.output / "first_frame.png"
    last_path = args.output / "last_frame.png"
    success = False
    success_step = None
    max_divergence = 0.0
    executed_steps = 0
    try:
        initial_state = {
            "states": states[0],
            "model": LU.get_episode_model_xml(dataset, episode),
            "ep_meta": json.dumps(episode_meta),
        }
        reset_to(environment, initial_state)
        first_frame = render_rgb(environment, "robot0_agentview_left")
        Image.fromarray(first_frame).save(first_path)
        last_frame = first_frame
        with imageio.get_writer(video_path, fps=20) as writer:
            writer.append_data(first_frame)
            for step, action in enumerate(actions):
                _, _, done, info = environment.step(action)
                executed_steps = step + 1
                success = bool(
                    info.get("success", False) or environment._check_success()
                )
                if step + 1 < len(states):
                    actual_state = np.asarray(environment.sim.get_state().flatten())
                    max_divergence = max(
                        max_divergence,
                        float(np.linalg.norm(states[step + 1] - actual_state)),
                    )
                if step % args.video_skip == 0 or success or done:
                    last_frame = render_rgb(environment, "robot0_agentview_left")
                    writer.append_data(last_frame)
                if success:
                    success_step = step
                    break
        Image.fromarray(last_frame).save(last_path)
    finally:
        environment.close()

    report = {
        "schema_version": "robocasa365_official_replay_evidence_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "SUCCESS" if success else "FAILED",
        "success": success,
        "selection_id": args.selection_id or None,
        "validation_kind": "official_public_demonstration_action_replay",
        "pure_autonomous_vla": False,
        "counts_as_pure_policy_acceptance": False,
        "simulator_success_predicate_checked": True,
        "dataset": str(dataset),
        "episode_index": episode,
        "instruction": str(episode_meta.get("lang", "")),
        "recorded_success": bool(np.any(rewards > 0)),
        "executed_steps": executed_steps,
        "success_step": success_step,
        "max_state_divergence_l2": max_divergence,
        "source_sha256": {
            "states": sha256(dataset / f"extras/episode_{episode:06d}/states.npz"),
            "model_xml": sha256(
                dataset / f"extras/episode_{episode:06d}/model.xml.gz"
            ),
            "parquet": sha256(parquet),
        },
        "evidence": {
            "video": str(video_path.resolve()),
            "first_frame": str(first_path.resolve()),
            "last_frame": str(last_path.resolve()),
        },
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path.resolve(), flush=True)
    print(
        f"OFFICIAL_REPLAY success={success} step={success_step} "
        f"max_divergence={max_divergence:.6g}",
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
