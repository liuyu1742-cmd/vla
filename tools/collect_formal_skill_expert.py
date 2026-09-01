"""Collect successful expert demonstrations for a formal Task-2 Skill IR."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.pick_place_oracle import PickPlaceOracle, PickPlaceSnapshot


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RELATION = "organizing::toy"
EXPECTED_ACTIONS = [
    "locate(toy)",
    "grasp(toy)",
    "move(storage)",
    "place(toy)",
]


def validate_skill_ir_for_collection(skill_ir: Mapping[str, object]) -> None:
    if skill_ir.get("relation_key") != EXPECTED_RELATION:
        raise ValueError(f"first formal collector only accepts {EXPECTED_RELATION}")
    if skill_ir.get("canonical_actions") != EXPECTED_ACTIONS:
        raise ValueError("organizing::toy canonical actions differ from Task-2 contract")
    if not isinstance(skill_ir.get("instruction"), str) or not skill_ir["instruction"]:
        raise ValueError("Skill IR instruction is required")


def build_formal_report(
    skill_ir: Mapping[str, object],
    *,
    seed: int,
    success: bool,
    executed_steps: int,
    predicates: Mapping[str, bool],
) -> dict[str, object]:
    validate_skill_ir_for_collection(skill_ir)
    return {
        "schema_version": "formal_skill_execution_report_v1",
        "evidence_scope": "formal_skill_execution",
        "relation_key": EXPECTED_RELATION,
        "counts_toward_task2_coverage": bool(success),
        "seed": seed,
        "instruction": skill_ir["instruction"],
        "canonical_actions": list(EXPECTED_ACTIONS),
        "canonical_action_progress": len(EXPECTED_ACTIONS) if success else 0,
        "success": bool(success),
        "success_predicates": dict(predicates),
        "executed_steps": int(executed_steps),
    }


def _sample_array(name: str, value: Any, samples: int, tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    expected = (samples, *tail)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    return array


def save_formal_episode(
    output_root: Path,
    *,
    seed: int,
    frames: Any,
    actions: Any,
    phases: Sequence[str],
    canonical_phases: Sequence[str],
    eef_positions: Any,
    object_positions: Any,
    grasped: Sequence[bool],
    report: Mapping[str, object],
) -> tuple[Path, Path]:
    if report.get("relation_key") != EXPECTED_RELATION:
        raise ValueError("episode report relation_key mismatch")
    frame_array = np.asarray(frames, dtype=np.uint8)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"frames must have shape (N,H,W,3), got {frame_array.shape}")
    samples = int(frame_array.shape[0])
    action_array = _sample_array("actions", actions, samples, (7,)).astype(np.float32)
    eef_array = _sample_array("eef_positions", eef_positions, samples, (3,)).astype(
        np.float32
    )
    object_array = _sample_array(
        "object_positions", object_positions, samples, (3,)
    ).astype(np.float32)
    if not all(len(values) == samples for values in (phases, canonical_phases, grasped)):
        raise ValueError("phase and grasp state lengths must match frame count")

    episode_dir = Path(output_root) / f"seed_{seed:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_path = episode_dir / "episode.npz"
    report_path = episode_dir / "report.json"
    np.savez_compressed(
        episode_path,
        frames=frame_array,
        actions=action_array,
        phases=np.asarray(phases, dtype="U48"),
        canonical_phases=np.asarray(canonical_phases, dtype="U16"),
        eef_positions=eef_array,
        object_positions=object_array,
        grasped=np.asarray(grasped, dtype=bool),
        relation_key=np.asarray(EXPECTED_RELATION),
        instruction=np.asarray(str(report.get("instruction", ""))),
    )
    report_path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_path, report_path


def cabinet_waypoints(raw: Any, depth_fraction: float = 0.5) -> tuple[np.ndarray, ...]:
    interior = next(iter(raw.cab.get_int_sites(relative=False).values()))
    p0, px, py, pz = (np.asarray(point, dtype=float) for point in interior)
    width_vector = px - p0
    depth_vector = py - p0
    height_vector = pz - p0
    depth_direction = depth_vector / np.linalg.norm(depth_vector)
    center = p0 + 0.5 * width_vector + depth_fraction * depth_vector + 0.35 * height_vector
    front = p0 + 0.5 * width_vector - 0.12 * depth_direction + 0.35 * height_vector
    retreat = front - 0.15 * depth_direction
    return front, center, retreat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-ir",
        type=Path,
        default=ROOT / "data" / "skill_coverage" / "generated" / "organizing_toy_skill_ir.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "datasets" / "formal_skills" / "organizing_toy",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed < 0 or args.frame_stride < 1 or args.max_steps < 1:
        raise ValueError("seed must be non-negative; stride and max steps must be positive")
    skill_bytes = args.skill_ir.read_bytes()
    skill_ir = json.loads(skill_bytes.decode("utf-8"))
    validate_skill_ir_for_collection(skill_ir)

    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    from tools.skill_transfer.robocasa_envs import create_formal_env

    env = None
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[str] = []
    canonical_phases: list[str] = []
    eef_positions: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    grasped_states: list[bool] = []
    diagnostics: list[dict[str, object]] = []
    phase_counts: Counter[str] = Counter()
    success = False
    executed_steps = 0
    try:
        env, environment_mapping = create_formal_env(EXPECTED_RELATION, seed=args.seed)
        observation, _ = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        front, center, retreat = cabinet_waypoints(raw)
        oracle = PickPlaceOracle(
            front,
            center,
            retreat,
            world_to_origin=controller.world_to_origin_frame,
        )

        for step in range(args.max_steps):
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            decision = oracle.decide(PickPlaceSnapshot(eef, obj, grasped, success))
            if step % args.frame_stride == 0:
                frames.append(
                    np.asarray(observation["video.robot0_agentview_left"], dtype=np.uint8).copy()
                )
                actions.append(decision.action.copy())
                phases.append(decision.phase)
                canonical_phases.append(decision.canonical_action)
                eef_positions.append(eef.copy())
                object_positions.append(obj.copy())
                grasped_states.append(grasped)

            observation, _, _, _, info = env.step(to_robocasa_action(decision.action))
            executed_steps = step + 1
            next_eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            next_obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            next_grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            success = bool(info.get("success", False) or raw._check_success())
            phase_counts[decision.phase] += 1
            diagnostics.append(
                {
                    "step": step,
                    "phase": decision.phase,
                    "canonical_action": decision.canonical_action,
                    "grasped": next_grasped,
                    "object_eef_distance": float(np.linalg.norm(next_obj - next_eef)),
                    "success": success,
                }
            )
            if step % 25 == 0 or success:
                print(
                    f"step={step} phase={decision.phase} canonical={decision.canonical_action} "
                    f"distance={diagnostics[-1]['object_eef_distance']:.4f} "
                    f"grasped={next_grasped} success={success}",
                    flush=True,
                )
            if success:
                break
    finally:
        if env is not None:
            env.close()

    predicates = {
        "simulator_success": success,
        "placed_in_storage": success,
        "gripper_released": success,
    }
    report = build_formal_report(
        skill_ir,
        seed=args.seed,
        success=success,
        executed_steps=executed_steps,
        predicates=predicates,
    )
    report.update(
        {
            "samples": len(frames),
            "frame_stride": args.frame_stride,
            "skill_ir_path": str(args.skill_ir.resolve()),
            "skill_ir_sha256": hashlib.sha256(skill_bytes).hexdigest(),
            "environment_mapping": environment_mapping,
            "phase_counts": dict(phase_counts),
            "grasp_attempts": oracle.grasp_attempts if "oracle" in locals() else 0,
            "steps": diagnostics,
        }
    )
    episode_path, report_path = save_formal_episode(
        args.output_root,
        seed=args.seed,
        frames=frames,
        actions=actions,
        phases=phases,
        canonical_phases=canonical_phases,
        eef_positions=eef_positions,
        object_positions=object_positions,
        grasped=grasped_states,
        report=report,
    )
    print(
        json.dumps(
            {
                "relation_key": EXPECTED_RELATION,
                "seed": args.seed,
                "success": success,
                "executed_steps": executed_steps,
                "samples": len(frames),
                "episode": str(episode_path.resolve()),
                "report": str(report_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
