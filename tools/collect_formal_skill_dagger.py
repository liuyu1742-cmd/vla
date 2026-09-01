"""Collect oracle recovery labels on states visited by the formal OpenVLA policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.collect_formal_skill_expert import (
    EXPECTED_RELATION,
    validate_skill_ir_for_collection,
)
from tools.collect_formal_skill_expert_safe import _safe_cabinet_waypoints
from tools.formal_skill_dagger import (
    choose_executed_action,
    save_dagger_episode,
    validate_training_seed,
)
from tools.formal_skill_policy_prompt import CANONICAL_POLICY_INSTRUCTION
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.adaptive_lift_safe_cabinet import (
    AdaptiveLiftSafeCabinetPickPlaceOracle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "datasets"
    / "formal_skills"
    / "organizing_toy"
    / "training_manifest.json"
)
DEFAULT_SKILL_IR = (
    ROOT
    / "data"
    / "skill_coverage"
    / "generated"
    / "organizing_toy_skill_ir.json"
)
DEFAULT_OUTPUT = (
    ROOT / "datasets" / "formal_skills" / "organizing_toy_dagger_r2"
)


def load_heldout_seeds(manifest_path: Path) -> set[int]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "formal_skill_training_manifest_v1":
        raise ValueError("unsupported formal skill manifest schema")
    if manifest.get("relation_key") != EXPECTED_RELATION:
        raise ValueError("formal DAgger manifest relation mismatch")
    heldout = manifest.get("held_out")
    if not isinstance(heldout, list) or not heldout:
        raise ValueError("formal DAgger manifest requires held-out entries")
    return {int(item["seed"]) for item in heldout}


def validate_collection_args(
    *,
    seed: int,
    beta: float,
    steps: int,
    heldout_seeds: Iterable[int],
) -> None:
    validate_training_seed(seed, heldout_seeds)
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must be between zero and one")
    if int(steps) < 1:
        raise ValueError("steps must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skill-ir", type=Path, default=DEFAULT_SKILL_IR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    heldout_seeds = load_heldout_seeds(args.manifest)
    validate_collection_args(
        seed=args.seed,
        beta=args.beta,
        steps=args.steps,
        heldout_seeds=heldout_seeds,
    )
    skill_bytes = args.skill_ir.read_bytes()
    skill_ir = json.loads(skill_bytes.decode("utf-8"))
    validate_skill_ir_for_collection(skill_ir)

    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    import imageio.v3 as iio
    from robocasa.utils import object_utils as OU
    from tools.skill_transfer.robocasa_envs import create_formal_env

    query_dir = Path(args.output_root) / "_queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    query_path = query_dir / (
        f"seed_{args.seed:03d}_round_{args.round:02d}.png"
    )
    rng = random.Random((args.round + 1) * 1_000_003 + args.seed)
    frames: list[np.ndarray] = []
    oracle_actions: list[np.ndarray] = []
    policy_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    phases: list[str] = []
    canonical_phases: list[str] = []
    eef_positions: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    grasped_states: list[bool] = []
    records: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    success = False
    ever_grasped = False
    ever_inside = False
    environment_mapping: dict[str, Any] = {}
    env = None
    try:
        env, environment_mapping = create_formal_env(
            EXPECTED_RELATION, seed=args.seed
        )
        observation, _ = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        object_model = raw.objects["obj"]
        front, center, retreat = _safe_cabinet_waypoints(raw)
        oracle = AdaptiveLiftSafeCabinetPickPlaceOracle(
            front,
            center,
            retreat,
            world_to_origin=controller.world_to_origin_frame,
        )

        for step in range(args.steps):
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float).copy()
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float).copy()
            grasped = bool(raw._check_grasp(robot.gripper["right"], object_model))
            inside = bool(OU.obj_inside_of(raw, "obj", raw.cab))
            simulator_success = bool(raw._check_success())
            success = bool(success or simulator_success)
            decision = oracle.decide(PickPlaceSnapshot(eef, obj, grasped, success))
            frame = np.asarray(
                observation["video.robot0_agentview_left"], dtype=np.uint8
            ).copy()
            iio.imwrite(query_path, frame)
            policy = np.asarray(
                predict(
                    args.host,
                    args.port,
                    {
                        "image_path": str(query_path.resolve()),
                        "instruction": CANONICAL_POLICY_INSTRUCTION,
                        "canonical_phase": decision.canonical_action,
                        "relation_key": EXPECTED_RELATION,
                    },
                ),
                dtype=np.float32,
            )
            mixed = choose_executed_action(
                policy, decision, beta=args.beta, rng=rng
            )

            frames.append(frame)
            oracle_actions.append(mixed.oracle_label.copy())
            policy_actions.append(policy.copy())
            executed_actions.append(mixed.executed_action.copy())
            phases.append(decision.phase)
            canonical_phases.append(decision.canonical_action)
            eef_positions.append(eef)
            object_positions.append(obj)
            grasped_states.append(grasped)
            phase_counts[decision.phase] += 1
            source_counts[mixed.source] += 1

            observation, _, _, _, info = env.step(
                to_robocasa_action(mixed.executed_action)
            )
            next_eef = np.asarray(
                raw.sim.data.site_xpos[eef_site_id], dtype=float
            )
            next_obj = np.asarray(
                raw.sim.data.body_xpos[object_body_id], dtype=float
            )
            next_grasped = bool(
                raw._check_grasp(robot.gripper["right"], object_model)
            )
            next_inside = bool(OU.obj_inside_of(raw, "obj", raw.cab))
            success = bool(
                success or info.get("success", False) or raw._check_success()
            )
            ever_grasped = bool(ever_grasped or next_grasped)
            ever_inside = bool(ever_inside or next_inside)
            records.append(
                {
                    "step": step,
                    "phase": decision.phase,
                    "canonical_phase": decision.canonical_action,
                    "source": mixed.source,
                    "gripper_gated": mixed.gripper_gated,
                    "policy_action": policy.tolist(),
                    "oracle_action": mixed.oracle_label.tolist(),
                    "executed_action": mixed.executed_action.tolist(),
                    "object_eef_distance": float(
                        np.linalg.norm(next_obj - next_eef)
                    ),
                    "grasped": next_grasped,
                    "inside": next_inside,
                    "success": success,
                }
            )
            if step % 25 == 0 or success:
                print(
                    f"step={step} phase={decision.phase} "
                    f"source={mixed.source} "
                    f"distance={records[-1]['object_eef_distance']:.4f} "
                    f"grasped={next_grasped} inside={next_inside} "
                    f"success={success}",
                    flush=True,
                )
            if success:
                break
    finally:
        if env is not None:
            env.close()

    report = {
        "schema_version": "formal_skill_dagger_report_v1",
        "relation_key": EXPECTED_RELATION,
        "seed": int(args.seed),
        "round": int(args.round),
        "beta": float(args.beta),
        "instruction": str(skill_ir["instruction"]),
        "policy_instruction": CANONICAL_POLICY_INSTRUCTION,
        "skill_ir_path": str(args.skill_ir.resolve()),
        "skill_ir_sha256": hashlib.sha256(skill_bytes).hexdigest(),
        "manifest": str(args.manifest.resolve()),
        "environment_mapping": environment_mapping,
        "executed_steps": len(records),
        "success": success,
        "ever_grasped": ever_grasped,
        "ever_inside": ever_inside,
        "phase_counts": dict(phase_counts),
        "execution_source_counts": dict(source_counts),
        "steps": records,
    }
    episode_path, report_path, sidecar_path = save_dagger_episode(
        args.output_root,
        seed=args.seed,
        heldout_seeds=heldout_seeds,
        frames=frames,
        oracle_actions=oracle_actions,
        policy_actions=policy_actions,
        executed_actions=executed_actions,
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
                "seed": args.seed,
                "success": success,
                "samples": len(frames),
                "episode": str(episode_path.resolve()),
                "report": str(report_path.resolve()),
                "sidecar": str(sidecar_path.resolve()),
                "execution_source_counts": dict(source_counts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
