"""Collect oracle labels on states visited by the R4 OFT policy, then recover offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.collect_vla82_targeted_expert import drawer_waypoints
from tools.openvla_simulator_adapter import to_robocasa_action
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.open_gripper import OpenGripperPickPlaceOracle
from tools.robocasa_oft_rollout import PRIMARY_KEY, WRIST_KEY, _mapping, proprio_from_observation
from tools.robocasa_oft_tcp_client import predict_chunk


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "outputs/midterm_testing_vla82/simulator_mapping_plan.json")
    parser.add_argument("--selection-id", default="VLA82-014")
    parser.add_argument("--seed", type=int, default=824141)
    parser.add_argument("--policy-steps", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/vla82_oft_dagger")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party/robosuite"))
    sys.path.insert(0, str(ROOT / "third_party/robocasa"))
    import imageio.v3 as iio
    from tools.vla82_pure_closed_loop import _configured_environment

    mapping = _mapping(args.plan, args.selection_id)
    args.output.mkdir(parents=True, exist_ok=True)
    primary_path = args.output / "dagger_current_primary.png"
    wrist_path = args.output / "dagger_current_wrist.png"
    frames, wrists, proprios, oracle_actions, policy_actions, modes, phases = (
        [], [], [], [], [], [], []
    )
    diagnostics = []
    success = False
    with _configured_environment(mapping, args.seed) as environment:
        observation, _ = environment.reset(seed=args.seed)
        raw = environment.unwrapped.env
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        front, center, retreat = drawer_waypoints(raw.drawer)
        oracle = OpenGripperPickPlaceOracle(
            front, center, retreat, world_to_origin=controller.world_to_origin_frame
        )
        oracle.LIFT_HEIGHT = 0.14
        oracle.APPROACH_TOLERANCE = 0.055
        instruction = str(raw.get_ep_meta()["lang"])
        for step in range(args.max_steps):
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            label = oracle.decide(PickPlaceSnapshot(eef, obj, grasped, success))
            primary = np.asarray(observation[PRIMARY_KEY], dtype=np.uint8).copy()
            wrist = np.asarray(observation[WRIST_KEY], dtype=np.uint8).copy()
            proprio = proprio_from_observation(observation)
            if step < args.policy_steps:
                iio.imwrite(primary_path, primary)
                iio.imwrite(wrist_path, wrist)
                policy = predict_chunk(
                    args.host,
                    args.port,
                    {
                        "image_path": str(primary_path.resolve()),
                        "wrist_image_path": str(wrist_path.resolve()),
                        "proprio": proprio.tolist(),
                        "instruction": instruction,
                        "relation_key": args.selection_id,
                    },
                )[0]
                executed = np.clip(policy, -1.0, 1.0)
                mode = "policy_visited_oracle_labeled"
            else:
                policy = np.full(7, np.nan, dtype=np.float32)
                executed = label.action
                mode = "expert_recovery_oracle_labeled"
            frames.append(primary)
            wrists.append(wrist)
            proprios.append(proprio)
            oracle_actions.append(label.action.copy())
            policy_actions.append(policy.copy())
            modes.append(mode)
            phases.append(label.phase)
            observation, _, terminated, truncated, info = environment.step(
                to_robocasa_action(executed)
            )
            success = bool(info.get("success", False) or raw._check_success())
            diagnostics.append(
                {
                    "step": step,
                    "mode": mode,
                    "oracle_phase": label.phase,
                    "distance": float(np.linalg.norm(eef - obj)),
                    "grasped": grasped,
                    "success": success,
                }
            )
            if step % 25 == 0 or success:
                print(
                    f"step={step} mode={mode} phase={label.phase} "
                    f"distance={diagnostics[-1]['distance']:.4f} grasped={grasped} success={success}",
                    flush=True,
                )
            if success or terminated or truncated:
                break
        success = bool(success or raw._check_success())

    episode_path = args.output / f"{args.selection_id}_seed_{args.seed}_dagger.npz"
    np.savez_compressed(
        episode_path,
        primary=np.asarray(frames, dtype=np.uint8),
        wrist=np.asarray(wrists, dtype=np.uint8),
        proprio=np.asarray(proprios, dtype=np.float32),
        actions=np.asarray(oracle_actions, dtype=np.float32),
        policy_actions=np.asarray(policy_actions, dtype=np.float32),
        modes=np.asarray(modes, dtype="U48"),
        phases=np.asarray(phases, dtype="U48"),
        instruction=np.asarray(instruction),
    )
    report = {
        "status": "SUCCESS" if success else "FAILED",
        "success": success,
        "samples": len(oracle_actions),
        "policy_visited_samples": min(args.policy_steps, len(oracle_actions)),
        "expert_recovery_samples": max(0, len(oracle_actions) - args.policy_steps),
        "uses_privileged_state_for_training_labels": True,
        "expert_recovery_used_during_collection": True,
        "counts_as_pure_policy_acceptance": False,
        "episode": str(episode_path.resolve()),
        "diagnostics": diagnostics,
    }
    report_path = args.output / f"{args.selection_id}_seed_{args.seed}_dagger_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path.resolve(), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
