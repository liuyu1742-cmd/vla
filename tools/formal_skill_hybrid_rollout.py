"""OpenVLA rollout with calibrated pickup timing and final-contact placement servo."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.formal_skill_final_placement import activate_final_placement
from tools.formal_skill_rollout import adapter_fingerprint, guarded_action
from tools.formal_skill_transport_guard import CalibratedPickPlaceGuard
from tools.formal_skill_verified_phase import VerifiedPhaseController
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


ROOT = Path(__file__).resolve().parents[1]
RELATION_KEY = "organizing::toy"


def _manifest_split(path: Path, seed: int) -> tuple[str, str]:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    train = {int(item["seed"]) for item in manifest.get("train", [])}
    held_out = {int(item["seed"]) for item in manifest.get("held_out", [])}
    if manifest.get("relation_key") != RELATION_KEY or train & held_out:
        raise ValueError("invalid formal training manifest")
    split = "held_out" if seed in held_out else "train" if seed in train else "unregistered"
    return split, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "datasets"
        / "formal_skills"
        / "organizing_toy"
        / "training_manifest.json",
    )
    parser.add_argument(
        "--skill-ir",
        type=Path,
        default=ROOT
        / "data"
        / "skill_coverage"
        / "generated"
        / "organizing_toy_skill_ir.json",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "models" / "openvla-organizing-toy-lora-balanced-r1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "formal_skill_hybrid_eval",
    )
    args = parser.parse_args()
    if args.seed < 0 or args.steps != 300 or args.action_repeat < 1:
        raise ValueError("formal hybrid acceptance requires 300 positive decisions")
    split, manifest_hash = _manifest_split(args.manifest, args.seed)
    if split != "held_out":
        raise ValueError("formal hybrid acceptance requires a held-out seed")
    skill_bytes = args.skill_ir.read_bytes()
    skill_ir = json.loads(skill_bytes.decode("utf-8"))
    instruction = str(skill_ir.get("instruction", "")).strip()
    if skill_ir.get("relation_key") != RELATION_KEY or not instruction:
        raise ValueError("invalid organizing::toy Skill IR")

    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    import imageio.v3 as iio
    from robocasa.utils import object_utils as OU
    from tools.collect_formal_skill_expert_safe import _safe_cabinet_waypoints
    from tools.skill_transfer.robocasa_envs import create_formal_env

    output_dir = args.output_root / f"seed_{args.seed:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "current_frame.png"
    report_path = output_dir / "report.json"
    video_path = output_dir / "rollout.mp4"
    trajectory: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []
    phase_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    inference_seconds: list[float] = []
    phase_controller = VerifiedPhaseController(max_grasp_decisions=20)
    execution_guard = CalibratedPickPlaceGuard(
        grasp_hold_decisions=18,
        lift_decisions=27,
        release_hold_decisions=18,
        base_guard=guarded_action,
    )
    env = None
    final_oracle = None
    final_servo_started_at = None
    success = False
    ever_grasped = False
    ever_inside = False
    final_grasped = False
    final_inside = False
    executed_decisions = 0
    executed_sim_steps = 0
    try:
        env, environment_mapping = create_formal_env(RELATION_KEY, seed=args.seed)
        observation, _ = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        object_model = raw.objects["obj"]
        front, center, retreat = _safe_cabinet_waypoints(raw)

        def state() -> tuple[np.ndarray, np.ndarray, float, bool, bool, bool]:
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float).copy()
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float).copy()
            grasped = bool(raw._check_grasp(robot.gripper["right"], object_model))
            inside = bool(OU.obj_inside_of(raw, "obj", raw.cab))
            simulator_success = bool(raw._check_success())
            return eef, obj, float(np.linalg.norm(obj - eef)), grasped, inside, simulator_success

        eef, obj, distance, grasped, inside, success = state()
        phase = phase_controller(
            "locate",
            object_eef_distance=distance,
            grasped=grasped,
            inside=inside,
            success=success,
        )
        for decision_step in range(args.steps):
            eef, obj, distance_before, grasped, inside, simulator_success = state()
            success = bool(success or simulator_success)
            if success:
                break
            snapshot = PickPlaceSnapshot(eef, obj, grasped, success)
            final_decision = final_oracle.decide(snapshot) if final_oracle else None
            request_phase = (
                final_decision.canonical_action if final_decision is not None else phase
            )
            frame = np.asarray(
                observation["video.robot0_agentview_left"], dtype=np.uint8
            ).copy()
            video_frames.append(frame)
            iio.imwrite(image_path, frame)
            started = time.perf_counter()
            raw_action = predict(
                args.host,
                args.port,
                {
                    "image_path": str(image_path.resolve()),
                    "instruction": instruction,
                    "canonical_phase": request_phase,
                    "relation_key": RELATION_KEY,
                },
            )
            inference_seconds.append(time.perf_counter() - started)
            if final_decision is None:
                action = execution_guard.apply(raw_action, phase)
                execution_mode = (
                    "openvla_guarded"
                    if np.allclose(action, guarded_action(raw_action, phase))
                    else "calibrated_pick_timing"
                )
                controller_phase = phase
            else:
                action = final_decision.action.copy()
                execution_mode = "final_contact_servo"
                controller_phase = final_decision.phase

            repeats = 0
            info: dict[str, Any] = {}
            for _ in range(args.action_repeat):
                observation, _, terminated, truncated, info = env.step(
                    to_robocasa_action(action)
                )
                repeats += 1
                executed_sim_steps += 1
                if bool(info.get("success", False) or raw._check_success()):
                    success = True
                    break
                if terminated or truncated:
                    break
            next_eef, next_obj, distance_after, next_grasped, next_inside, simulator_success = state()
            success = bool(success or simulator_success or info.get("success", False))
            ever_grasped = ever_grasped or next_grasped
            ever_inside = ever_inside or next_inside

            if final_oracle is None and next_inside and next_grasped:
                final_oracle = activate_final_placement(
                    SafeCabinetPickPlaceOracle(
                        front,
                        center,
                        retreat,
                        world_to_origin=controller.world_to_origin_frame,
                    )
                )
                final_servo_started_at = decision_step + 1
                next_phase = "move"
            elif final_oracle is not None:
                next_phase = (
                    "done"
                    if success
                    else final_oracle.CANONICAL_BY_PHASE[final_oracle.phase]
                )
            else:
                next_phase = phase_controller(
                    phase,
                    object_eef_distance=distance_after,
                    grasped=next_grasped,
                    inside=next_inside,
                    success=success,
                )
            phase_counts[request_phase] += 1
            mode_counts[execution_mode] += 1
            executed_decisions = decision_step + 1
            trajectory.append(
                {
                    "decision_step": decision_step,
                    "phase": request_phase,
                    "next_phase": next_phase,
                    "controller_phase": controller_phase,
                    "execution_mode": execution_mode,
                    "raw_action": [float(value) for value in raw_action],
                    "executed_action": action.tolist(),
                    "object_eef_distance_before": distance_before,
                    "object_eef_distance_after": distance_after,
                    "eef_position": next_eef.tolist(),
                    "object_position": next_obj.tolist(),
                    "grasped": next_grasped,
                    "inside": next_inside,
                    "success": success,
                }
            )
            if decision_step % 10 == 0 or next_phase != request_phase or success:
                print(
                    f"decision={decision_step}/{args.steps} phase={request_phase}->{next_phase} "
                    f"mode={execution_mode} distance={distance_after:.4f} "
                    f"grasped={next_grasped} inside={next_inside} success={success}",
                    flush=True,
                )
            phase = next_phase
            if success:
                break
        _, _, _, final_grasped, final_inside, simulator_success = state()
        success = bool(success or simulator_success)
    finally:
        if env is not None:
            env.close()

    if video_frames:
        iio.imwrite(video_path, np.stack(video_frames), fps=20)
    predicates = {
        "simulator_success": success,
        "placed_in_storage": final_inside,
        "gripper_released": not final_grasped,
    }
    report = {
        "schema_version": "formal_skill_hybrid_model_evaluation_v1",
        "evidence_scope": "formal_skill_hybrid_model_evaluation",
        "relation_key": RELATION_KEY,
        "seed": args.seed,
        "split": split,
        "success": success,
        "success_predicates": predicates,
        "counts_toward_task2_coverage": bool(success and all(predicates.values())),
        "counts_toward_autonomous_vla_acceptance": False,
        "max_decision_steps": args.steps,
        "action_repeat": args.action_repeat,
        "executed_decision_steps": executed_decisions,
        "executed_sim_steps": executed_sim_steps,
        "ever_grasped": ever_grasped,
        "ever_inside": ever_inside,
        "final_grasped": final_grasped,
        "final_inside": final_inside,
        "final_servo_started_at": final_servo_started_at,
        "phase_counts": dict(phase_counts),
        "execution_mode_counts": dict(mode_counts),
        "mean_inference_seconds": (
            float(np.mean(inference_seconds)) if inference_seconds else None
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "skill_ir": str(args.skill_ir.resolve()),
        "skill_ir_sha256": hashlib.sha256(skill_bytes).hexdigest(),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "adapter_sha256": adapter_fingerprint(args.adapter_dir),
        "environment_mapping": environment_mapping,
        "video": str(video_path.resolve()) if video_frames else None,
        "trajectory": trajectory,
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "relation_key",
                    "seed",
                    "success",
                    "success_predicates",
                    "executed_decision_steps",
                    "ever_grasped",
                    "ever_inside",
                    "execution_mode_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(report_path.resolve(), flush=True)
    return 0 if success and all(predicates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
