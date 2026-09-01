"""Collect oracle corrections on states visited by the current OpenVLA policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.openvla_gym_water_cup_rollout_v2 import install_object_scale
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict
from tools.water_cup_dagger_mixing import choose_executed_action
from tools.water_cup_dagger_oracle import OracleSnapshot, WaterCupDaggerOracle


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "pick up the glass cup and place it in the cabinet"
HELD_OUT_SEED = 2


def validate_training_seed(seed: int) -> None:
    if seed == HELD_OUT_SEED:
        raise ValueError("held-out seed 2 cannot be used for DAgger training collection")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _sample_array(name: str, value: Any, samples: int, tail: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    expected = (samples, *tail)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    return result


def save_episode(
    output_dir: Path,
    *,
    seed: int,
    frames: Any,
    oracle_actions: Any,
    policy_actions: Any,
    executed_actions: Any,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist DAgger samples, with oracle actions as the canonical labels."""

    validate_training_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_array = np.asarray(frames, dtype=np.uint8)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"frames must have shape (N,H,W,3), got {frame_array.shape}")
    samples = int(frame_array.shape[0])
    oracle_array = _sample_array("oracle_actions", oracle_actions, samples, (7,)).astype(
        np.float32
    )
    policy_array = _sample_array("policy_actions", policy_actions, samples, (7,)).astype(
        np.float32
    )
    executed_array = _sample_array(
        "executed_actions", executed_actions, samples, (7,)
    ).astype(np.float32)
    round_index = int(report.get("round", 0))
    stem = f"episode_seed_{seed:03d}_round_{round_index:02d}"
    episode_path = output_dir / f"{stem}.npz"
    report_path = output_dir / f"{stem}_report.json"
    manifest_path = output_dir / f"{stem}.json"
    np.savez_compressed(
        episode_path,
        frames=frame_array,
        actions=oracle_array,
        oracle_actions=oracle_array,
        policy_actions=policy_array,
        executed_actions=executed_array,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "seed": seed,
        "round": round_index,
        "instruction": INSTRUCTION,
        "source": "dagger_recovery",
        "success": bool(report.get("success", False)),
        "samples": samples,
        "beta": float(report.get("beta", 0.0)),
        "episode": str(episode_path.resolve()),
        "report": str(report_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return episode_path, manifest_path


def cabinet_waypoints(raw: Any, depth_fraction: float = 0.5) -> tuple[np.ndarray, ...]:
    interior = next(iter(raw.cab.get_int_sites(relative=False).values()))
    p0, px, py, pz = (np.asarray(point, dtype=float) for point in interior)
    width_vector = px - p0
    depth_vector = py - p0
    height_vector = pz - p0
    depth_direction = depth_vector / np.linalg.norm(depth_vector)
    center = (
        p0
        + 0.5 * width_vector
        + depth_fraction * depth_vector
        + 0.35 * height_vector
    )
    front = p0 + 0.5 * width_vector - 0.12 * depth_direction + 0.35 * height_vector
    retreat = front - 0.15 * depth_direction
    return front, center, retreat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--object-scale", type=float, default=0.7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "water_cup_dagger",
    )
    args = parser.parse_args()
    validate_training_seed(args.seed)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be between zero and one")

    third_party = ROOT / "third_party"
    sys.path.insert(0, str(third_party / "robosuite"))
    sys.path.insert(0, str(third_party / "robocasa"))
    import gymnasium as gym
    import imageio.v3 as iio
    import robocasa  # noqa: F401
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
        PickPlaceCounterToCabinet,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_get_obj_cfgs = install_object_scale(PickPlaceCounterToCabinet, args.object_scale)
    env = None
    frames: list[np.ndarray] = []
    oracle_actions: list[np.ndarray] = []
    policy_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rng = random.Random((args.round + 1) * 1000003 + args.seed)
    success = False
    initial_object_z = 0.0
    query_path = args.output_dir / f"query_seed_{args.seed:03d}_round_{args.round:02d}.png"
    try:
        env = gym.make(
            "robocasa/PickPlaceCounterToCabinet",
            split="pretrain",
            seed=args.seed,
            obj_registries=("lightwheel",),
            obj_groups="glass_cup",
            disable_env_checker=True,
        )
        observation, _ = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        initial_object_z = float(raw.sim.data.body_xpos[object_body_id][2])
        front, center, retreat = cabinet_waypoints(raw)
        oracle = WaterCupDaggerOracle(
            front,
            center,
            retreat,
            world_to_origin=controller.world_to_origin_frame,
        )

        for step in range(args.steps):
            frame = np.asarray(observation["video.robot0_agentview_left"], dtype=np.uint8)
            iio.imwrite(query_path, frame)
            policy = np.asarray(
                predict(
                    args.host,
                    args.port,
                    {"image_path": str(query_path.resolve()), "instruction": INSTRUCTION},
                ),
                dtype=np.float32,
            )
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            decision = oracle.decide(OracleSnapshot(eef, obj, grasped, success))
            mixed = choose_executed_action(policy, decision, beta=args.beta, rng=rng)

            frames.append(frame.copy())
            oracle_actions.append(mixed.oracle_label.copy())
            policy_actions.append(policy.copy())
            executed_actions.append(mixed.executed_action.copy())
            phase_counts[decision.phase] += 1
            source_counts[mixed.source] += 1

            observation, _, _, _, info = env.step(
                to_robocasa_action(mixed.executed_action)
            )
            next_eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            next_obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            next_grasped = bool(
                raw._check_grasp(robot.gripper["right"], raw.objects["obj"])
            )
            success = bool(info.get("success", False) or raw._check_success())
            records.append(
                {
                    "step": step,
                    "phase": decision.phase,
                    "source": mixed.source,
                    "gripper_gated": mixed.gripper_gated,
                    "policy_action": policy.tolist(),
                    "oracle_action": mixed.oracle_label.tolist(),
                    "executed_action": mixed.executed_action.tolist(),
                    "grasped": next_grasped,
                    "object_z": float(next_obj[2]),
                    "object_eef_distance": float(np.linalg.norm(next_obj - next_eef)),
                    "success": success,
                }
            )
            if step % 25 == 0 or success:
                print(
                    f"step={step} phase={decision.phase} source={mixed.source} "
                    f"distance={records[-1]['object_eef_distance']:.4f} "
                    f"grasped={next_grasped} success={success}",
                    flush=True,
                )
            if success:
                break
    finally:
        if env is not None:
            env.close()
        PickPlaceCounterToCabinet._get_obj_cfgs = original_get_obj_cfgs

    heights = [record["object_z"] for record in records]
    distances = [record["object_eef_distance"] for record in records]
    report = {
        "seed": args.seed,
        "round": args.round,
        "beta": args.beta,
        "object_scale": args.object_scale,
        "instruction": INSTRUCTION,
        "executed_steps": len(records),
        "success": success,
        "ever_grasped": any(record["grasped"] for record in records),
        "initial_object_z": initial_object_z,
        "max_object_z": max(heights) if heights else None,
        "object_lift": max(heights) - initial_object_z if heights else None,
        "minimum_object_eef_distance": min(distances) if distances else None,
        "phase_counts": dict(phase_counts),
        "execution_source_counts": dict(source_counts),
        "steps": records,
    }
    episode_path, manifest_path = save_episode(
        args.output_dir,
        seed=args.seed,
        frames=frames,
        oracle_actions=oracle_actions,
        policy_actions=policy_actions,
        executed_actions=executed_actions,
        report=report,
    )
    print(
        json.dumps(
            {
                "success": success,
                "samples": len(frames),
                "episode": str(episode_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "phase_counts": dict(phase_counts),
                "execution_source_counts": dict(source_counts),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
