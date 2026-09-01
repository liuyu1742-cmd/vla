"""Run a 300-decision OpenVLA rollout with a verified progress safety controller."""

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

from tools.formal_skill_progress_guard import select_progress_guarded_action
from tools.formal_skill_rollout import adapter_fingerprint
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


ROOT = Path(__file__).resolve().parents[1]
RELATION_KEY = "organizing::toy"
CANONICAL_ACTIONS = ["locate(toy)", "grasp(toy)", "move(storage)", "place(toy)"]


class EvaluationSafetyOracle(SafeCabinetPickPlaceOracle):
    """Match the verified hold time while two simulator steps share one decision."""

    CLOSE_HOLD_STEPS = 18
    RELEASE_HOLD_STEPS = 18
    SETTLE_STEPS = 10


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_split(path: Path, seed: int) -> tuple[str, str]:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("relation_key") != RELATION_KEY:
        raise ValueError("evaluation manifest relation mismatch")
    train = {int(item["seed"]) for item in manifest.get("train", [])}
    held_out = {int(item["seed"]) for item in manifest.get("held_out", [])}
    if train & held_out:
        raise ValueError("evaluation manifest has overlapping splits")
    split = "held_out" if seed in held_out else "train" if seed in train else "unregistered"
    return split, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--model-weight", type=float, default=0.15)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
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
        default=ROOT / "models" / "openvla-organizing-toy-lora",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "formal_skill_guarded_eval",
    )
    args = parser.parse_args()
    if args.seed < 0 or args.steps != 300 or args.action_repeat < 1:
        raise ValueError("guarded acceptance requires a non-negative seed and 300 decisions")
    if not 0.0 <= args.model_weight <= 1.0:
        raise ValueError("model-weight must be in [0, 1]")
    split, manifest_hash = _manifest_split(args.manifest, args.seed)
    if split != "held_out":
        raise ValueError("guarded acceptance requires a held-out seed")
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
    frames: list[np.ndarray] = []
    phase_counts: Counter[str] = Counter()
    guard_counts: Counter[str] = Counter()
    inference_seconds: list[float] = []
    env = None
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
        oracle = EvaluationSafetyOracle(
            front,
            center,
            retreat,
            world_to_origin=controller.world_to_origin_frame,
        )

        def state() -> tuple[np.ndarray, np.ndarray, float, bool, bool, bool]:
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float).copy()
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float).copy()
            grasped = bool(raw._check_grasp(robot.gripper["right"], object_model))
            inside = bool(OU.obj_inside_of(raw, "obj", raw.cab))
            simulator_success = bool(raw._check_success())
            return eef, obj, float(np.linalg.norm(obj - eef)), grasped, inside, simulator_success

        for decision_step in range(args.steps):
            eef, obj, distance_before, grasped, inside, simulator_success = state()
            success = bool(success or simulator_success)
            if success:
                break
            expert = oracle.decide(PickPlaceSnapshot(eef, obj, grasped, success))
            phase = expert.canonical_action
            frame = np.asarray(
                observation["video.robot0_agentview_left"], dtype=np.uint8
            ).copy()
            frames.append(frame)
            iio.imwrite(image_path, frame)
            started = time.perf_counter()
            raw_action = predict(
                args.host,
                args.port,
                {
                    "image_path": str(image_path.resolve()),
                    "instruction": instruction,
                    "canonical_phase": phase,
                    "relation_key": RELATION_KEY,
                },
            )
            inference_seconds.append(time.perf_counter() - started)
            guard = select_progress_guarded_action(
                raw_action,
                expert.action,
                force_expert=expert.force_expert,
                model_weight=args.model_weight,
            )
            repeats = 0
            info: dict[str, Any] = {}
            for _ in range(args.action_repeat):
                observation, _, terminated, truncated, info = env.step(
                    to_robocasa_action(guard.action)
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
            phase_counts[phase] += 1
            guard_counts[guard.mode] += 1
            executed_decisions = decision_step + 1
            trajectory.append(
                {
                    "decision_step": decision_step,
                    "sim_repeats": repeats,
                    "phase": phase,
                    "oracle_phase": expert.phase,
                    "raw_action": [float(value) for value in raw_action],
                    "expert_action": expert.action.tolist(),
                    "guarded_action": guard.action.tolist(),
                    "guard_mode": guard.mode,
                    "guard_intervened": guard.intervened,
                    "translation_cosine": guard.translation_cosine,
                    "object_eef_distance_before": distance_before,
                    "object_eef_distance_after": distance_after,
                    "grasped": next_grasped,
                    "inside": next_inside,
                    "success": success,
                    "eef_position": next_eef.tolist(),
                    "object_position": next_obj.tolist(),
                }
            )
            if decision_step % 10 == 0 or success:
                print(
                    f"decision={decision_step}/{args.steps} phase={phase} "
                    f"oracle={expert.phase} guard={guard.mode} "
                    f"distance={distance_after:.4f} grasped={next_grasped} "
                    f"inside={next_inside} success={success}",
                    flush=True,
                )
            if success:
                break
        _, _, _, final_grasped, final_inside, simulator_success = state()
        success = bool(success or simulator_success)
    finally:
        if env is not None:
            env.close()

    if frames:
        iio.imwrite(video_path, np.stack(frames), fps=20)
    predicates = {
        "simulator_success": success,
        "placed_in_storage": final_inside,
        "gripper_released": not final_grasped,
    }
    interventions = int(
        guard_counts.get("forced_expert", 0) + guard_counts.get("safety_expert", 0)
    )
    report = {
        "schema_version": "formal_skill_guarded_model_evaluation_v1",
        "evidence_scope": "formal_skill_guarded_model_evaluation",
        "relation_key": RELATION_KEY,
        "seed": args.seed,
        "split": split,
        "instruction": instruction,
        "canonical_actions": CANONICAL_ACTIONS,
        "success": success,
        "success_predicates": predicates,
        "counts_toward_task2_coverage": bool(success and all(predicates.values())),
        "counts_toward_autonomous_vla_acceptance": bool(
            success and all(predicates.values()) and interventions == 0
        ),
        "max_decision_steps": args.steps,
        "action_repeat": args.action_repeat,
        "model_weight": args.model_weight,
        "executed_decision_steps": executed_decisions,
        "executed_sim_steps": executed_sim_steps,
        "ever_grasped": ever_grasped,
        "ever_inside": ever_inside,
        "final_grasped": final_grasped,
        "final_inside": final_inside,
        "phase_counts": dict(phase_counts),
        "guard_counts": dict(guard_counts),
        "safety_interventions": interventions,
        "mean_inference_seconds": (
            float(np.mean(inference_seconds)) if inference_seconds else None
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "skill_ir": str(args.skill_ir.resolve()),
        "skill_ir_sha256": _sha256(args.skill_ir),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "adapter_sha256": adapter_fingerprint(args.adapter_dir),
        "environment_mapping": environment_mapping,
        "video": str(video_path.resolve()) if frames else None,
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
                    "executed_sim_steps",
                    "ever_grasped",
                    "ever_inside",
                    "guard_counts",
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
