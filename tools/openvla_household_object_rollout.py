"""Run one auditable protected OpenVLA pick-and-place trial in RoboCasa.

The OpenVLA service is queried at every decision.  If the model cannot make
locate-phase progress, a state-aware, geometry-safe recovery controller takes
over.  Therefore this script reports a *hybrid VLA closed loop*, never a pure
autonomous-VLA result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.formal_skill_final_transport import activate_final_transport
from tools.formal_skill_locate_recovery import LocateStagnationRecovery
from tools.formal_skill_rollout import guarded_action
from tools.formal_skill_transport_guard import CalibratedPickPlaceGuard
from tools.formal_skill_verified_phase import VerifiedPhaseController
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HouseholdObject:
    """A RoboCasa native object and its human-readable experimental name."""

    group: str
    english_name: str
    chinese_name: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("group", self.group),
            ("english_name", self.english_name),
            ("chinese_name", self.chinese_name),
        ):
            if not str(value).strip():
                raise ValueError(f"{field_name} must be non-empty")


def build_instruction(spec: HouseholdObject) -> str:
    """Return the exact English instruction passed to the VLA policy."""
    return f"pick up the {spec.english_name.strip()} and place it in the cabinet"


def validate_object_scale(object_scale: float) -> float:
    """Accept only positive task-object scales used by the simulator."""
    value = float(object_scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("object_scale must be a positive finite number")
    return value


def validate_grasp_height_offset(grasp_height_offset: float) -> float:
    """Accept a non-negative object-specific top-grasp offset."""
    value = float(grasp_height_offset)
    if not np.isfinite(value) or value < 0:
        raise ValueError("grasp_height_offset must be a non-negative finite number")
    return value


def _state(raw: Any, robot: Any, eef_site_id: int, object_body_id: int, object_model: Any, OU: Any):
    eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float).copy()
    obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float).copy()
    grasped = bool(raw._check_grasp(robot.gripper["right"], object_model))
    inside = bool(OU.obj_inside_of(raw, "obj", raw.cab))
    success = bool(raw._check_success())
    return eef, obj, float(np.linalg.norm(obj - eef)), grasped, inside, success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-group", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--object-name-zh", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--grasp-height-offset", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "models" / "openvla-organizing-toy-lora-dagger-r3",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "experiment_4_2_5"
    )
    args = parser.parse_args()
    if args.seed < 0 or args.steps < 1 or args.action_repeat < 1:
        raise ValueError("seed must be non-negative; steps and action-repeat must be positive")
    spec = HouseholdObject(args.object_group, args.object_name, args.object_name_zh)
    object_scale = validate_object_scale(args.object_scale)
    grasp_height_offset = validate_grasp_height_offset(args.grasp_height_offset)
    instruction = build_instruction(spec)

    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    import gymnasium as gym
    import imageio.v3 as iio
    import robocasa  # noqa: F401
    from robocasa.models.objects.kitchen_objects import OBJ_GROUPS
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
    from robocasa.utils import object_utils as OU
    from tools.collect_formal_skill_expert_safe import _safe_cabinet_waypoints

    if spec.group not in OBJ_GROUPS:
        raise ValueError(f"RoboCasa object group is unavailable: {spec.group!r}")
    output_dir = args.output_root / spec.group / f"seed_{args.seed:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "current_frame.png"
    video_path = output_dir / "rollout.mp4"
    report_path = output_dir / "report.json"

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
    stagnation = LocateStagnationRecovery(
        window_decisions=20, minimum_progress=0.01
    )
    env = None
    original_get_obj_cfgs = None
    if object_scale != 1.0:
        original_get_obj_cfgs = PickPlaceCounterToCabinet._get_obj_cfgs

        def scaled_get_obj_cfgs(environment: Any):
            configs = original_get_obj_cfgs(environment)
            for config in configs:
                if config.get("name") == "obj":
                    config["object_scale"] = object_scale
            return configs

        PickPlaceCounterToCabinet._get_obj_cfgs = scaled_get_obj_cfgs
    recovery_oracle = None
    final_oracle = None
    recovery_started_at = None
    final_servo_started_at = None
    success = False
    ever_grasped = False
    ever_inside = False
    final_grasped = False
    final_inside = False
    executed_decisions = 0
    executed_sim_steps = 0
    try:
        env = gym.make(
            "robocasa/PickPlaceCounterToCabinet",
            split="pretrain",
            seed=args.seed,
            obj_registries=("lightwheel",),
            obj_groups=spec.group,
            disable_env_checker=True,
        )
        observation, _ = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        object_model = raw.objects["obj"]
        front, center, retreat = _safe_cabinet_waypoints(raw)
        initial = _state(raw, robot, eef_site_id, object_body_id, object_model, OU)
        phase = phase_controller(
            "locate",
            object_eef_distance=initial[2],
            grasped=initial[3],
            inside=initial[4],
            success=initial[5],
        )

        for decision_step in range(args.steps):
            before = _state(raw, robot, eef_site_id, object_body_id, object_model, OU)
            eef, obj, distance_before, grasped, inside, simulator_success = before
            success = bool(success or simulator_success)
            if success:
                break
            frame = np.asarray(observation["video.robot0_agentview_left"], dtype=np.uint8).copy()
            video_frames.append(frame)
            iio.imwrite(image_path, frame)
            if recovery_oracle is not None:
                policy_phase = recovery_oracle.CANONICAL_BY_PHASE[recovery_oracle.phase]
            elif final_oracle is not None:
                policy_phase = final_oracle.CANONICAL_BY_PHASE[final_oracle.phase]
            else:
                policy_phase = phase
            started = time.perf_counter()
            raw_action = predict(
                args.host,
                args.port,
                {
                    "image_path": str(image_path.resolve()),
                    "instruction": instruction,
                    "canonical_phase": policy_phase,
                    "relation_key": f"experiment_4_2_5::{spec.group}",
                },
            )
            inference_seconds.append(time.perf_counter() - started)
            snapshot = PickPlaceSnapshot(eef, obj, grasped, simulator_success)
            recovery_active = (
                recovery_oracle is not None
                or (final_oracle is None and stagnation.observe(phase, distance_before))
            )
            if recovery_oracle is None and recovery_active:
                recovery_oracle = SafeCabinetPickPlaceOracle(
                    front,
                    center,
                    retreat,
                    world_to_origin=controller.world_to_origin_frame,
                )
                recovery_oracle.GRASP_HEIGHT_OFFSET = grasp_height_offset
                recovery_started_at = decision_step
            if recovery_oracle is not None:
                decision = recovery_oracle.decide(snapshot)
                action = decision.action.copy()
                execution_mode = "state_aware_locate_recovery_expert"
                controller_phase = decision.phase
            elif final_oracle is not None:
                decision = final_oracle.decide(snapshot)
                action = decision.action.copy()
                execution_mode = "final_contact_servo"
                controller_phase = decision.phase
            else:
                action = execution_guard.apply(raw_action, phase)
                execution_mode = (
                    "openvla_guarded"
                    if np.allclose(action, guarded_action(raw_action, phase))
                    else "calibrated_pick_timing"
                )
                controller_phase = phase

            info: dict[str, Any] = {}
            repeats = 0
            for _ in range(args.action_repeat):
                observation, _, terminated, truncated, info = env.step(to_robocasa_action(action))
                repeats += 1
                executed_sim_steps += 1
                if bool(info.get("success", False) or raw._check_success()):
                    success = True
                    break
                if terminated or truncated:
                    break
            after = _state(raw, robot, eef_site_id, object_body_id, object_model, OU)
            next_eef, next_obj, distance_after, next_grasped, next_inside, simulator_success = after
            success = bool(success or simulator_success or info.get("success", False))
            ever_grasped = bool(ever_grasped or next_grasped)
            ever_inside = bool(ever_inside or next_inside)

            if recovery_oracle is None and final_oracle is None and next_inside and next_grasped:
                final_oracle = activate_final_transport(
                    SafeCabinetPickPlaceOracle(
                        front,
                        center,
                        retreat,
                        world_to_origin=controller.world_to_origin_frame,
                    )
                )
                final_servo_started_at = decision_step + 1
                next_phase = "move"
            elif recovery_oracle is not None:
                next_phase = "done" if success else recovery_oracle.CANONICAL_BY_PHASE[recovery_oracle.phase]
            elif final_oracle is not None:
                next_phase = "done" if success else final_oracle.CANONICAL_BY_PHASE[final_oracle.phase]
            else:
                next_phase = phase_controller(
                    phase,
                    object_eef_distance=distance_after,
                    grasped=next_grasped,
                    inside=next_inside,
                    success=success,
                )
            phase_counts[policy_phase] += 1
            mode_counts[execution_mode] += 1
            executed_decisions = decision_step + 1
            trajectory.append(
                {
                    "decision_step": decision_step,
                    "policy_phase": policy_phase,
                    "next_phase": next_phase,
                    "controller_phase": controller_phase,
                    "execution_mode": execution_mode,
                    "raw_action": [float(value) for value in raw_action],
                    "executed_action": [float(value) for value in action],
                    "object_eef_distance_before": distance_before,
                    "object_eef_distance_after": distance_after,
                    "grasped": next_grasped,
                    "inside": next_inside,
                    "success": success,
                    "inference_seconds": inference_seconds[-1],
                    "eef_position": next_eef.tolist(),
                    "object_position": next_obj.tolist(),
                    "sim_repeats": repeats,
                }
            )
            print(
                f"object={spec.group} decision={decision_step}/{args.steps} "
                f"phase={policy_phase}->{next_phase} mode={execution_mode} "
                f"distance={distance_after:.4f} grasped={next_grasped} "
                f"inside={next_inside} success={success}",
                flush=True,
            )
            phase = next_phase
            if success:
                break
        final = _state(raw, robot, eef_site_id, object_body_id, object_model, OU)
        final_grasped, final_inside = final[3], final[4]
        success = bool(success or final[5])
    finally:
        if env is not None:
            env.close()
        if original_get_obj_cfgs is not None:
            PickPlaceCounterToCabinet._get_obj_cfgs = original_get_obj_cfgs

    if video_frames:
        iio.imwrite(video_path, np.stack(video_frames), fps=20)
    predicates = {
        "simulator_success": bool(success),
        "placed_in_storage": bool(final_inside),
        "gripper_released": bool(not final_grasped),
    }
    report = {
        "schema_version": "experiment_4_2_5_household_object_v1",
        "evaluation_kind": "hybrid_vla_closed_loop",
        "pure_autonomous_vla": False,
        "object_group": spec.group,
        "object_english_name": spec.english_name,
        "object_chinese_name": spec.chinese_name,
        "instruction": instruction,
        "seed": args.seed,
        "success": bool(success and all(predicates.values())),
        "success_predicates": predicates,
        "max_decision_steps": args.steps,
        "executed_decision_steps": executed_decisions,
        "executed_sim_steps": executed_sim_steps,
        "action_repeat": args.action_repeat,
        "ever_grasped": ever_grasped,
        "ever_inside": ever_inside,
        "recovery_started_at": recovery_started_at,
        "final_servo_started_at": final_servo_started_at,
        "phase_counts": dict(phase_counts),
        "execution_mode_counts": dict(mode_counts),
        "mean_inference_seconds": float(np.mean(inference_seconds)) if inference_seconds else None,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "environment": {
            "name": "robocasa/PickPlaceCounterToCabinet",
            "split": "pretrain",
            "object_registry": "lightwheel",
            "object_scale": object_scale,
            "grasp_height_offset": grasp_height_offset,
        },
        "video": str(video_path.resolve()) if video_frames else None,
        "trajectory": trajectory,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("object_group", "success", "success_predicates", "executed_decision_steps", "execution_mode_counts")}, ensure_ascii=False, indent=2), flush=True)
    print(report_path.resolve(), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
