"""Collect a successful dual-camera simulator expert episode for offline VLA training.

Privileged object/EEF state is used only to create labels.  These episodes must
never be counted as pure-policy acceptance trials.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.collect_formal_skill_expert import cabinet_waypoints
from tools.pick_place_oracle import PickPlaceSnapshot
from tools.pick_place_oracle.adaptive_lift_safe_cabinet import (
    AdaptiveLiftSafeCabinetPickPlaceOracle,
)
from tools.pick_place_oracle.fast_transit import FastTransitPickPlaceOracle
from tools.pick_place_oracle.open_gripper import OpenGripperPickPlaceOracle
from tools.robocasa_oft_rollout import PRIMARY_KEY, WRIST_KEY, _mapping, proprio_from_observation


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROTATION_TEMPLATE_DIR = ROOT / "datasets/vla82_public_rotation_templates"
PUBLIC_ROTATION_EPISODES = {
    ("PickPlaceCabinetToCounter", "canned_food"): 3740,
    ("PickPlaceCounterToCabinet", "tupperware"): 4021,
    ("PickPlaceCounterToSink", "tongs"): 4410,
    ("PickPlaceDrawerToCounter", "tongs"): 4807,
}
DEFAULT_CABINET_SOURCE_ROTATION_TEMPLATE = (
    ROOT
    / "datasets/vla82_robocasa365_oft/PickPlaceCabinetToCounter"
    / "episode_003712/actions.npy"
)


def public_rotation_template_path(task_class: str, object_group: str) -> Path | None:
    episode = PUBLIC_ROTATION_EPISODES.get((task_class, object_group))
    if episode is not None:
        return PUBLIC_ROTATION_TEMPLATE_DIR / f"episode_{episode:06d}_actions.npy"
    if task_class == "PickPlaceCabinetToCounter":
        return DEFAULT_CABINET_SOURCE_ROTATION_TEMPLATE
    return None


def apply_rotation_template(
    action: np.ndarray, step: int, template: np.ndarray | None
) -> np.ndarray:
    """Copy only public-demo wrist rotation, retaining scene-specific motion/grip."""
    result = np.asarray(action, dtype=np.float32).copy()
    if template is not None and 0 <= step < len(template):
        result[3:6] = np.asarray(template[step, 3:6], dtype=np.float32)
    return result


def drawer_waypoints(drawer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = drawer.get_int_sites(relative=False)["int"]
    p0, px, py, pz = (np.asarray(value, dtype=float) for value in points)
    width, depth, height = px - p0, py - p0, pz - p0
    depth_direction = depth / np.linalg.norm(depth)
    # The rear half of an open top drawer is collision-limited for PandaOmron.
    # A higher 35%-depth release point is inside the official bbox and reachable.
    center = p0 + 0.5 * width + 0.35 * depth + 0.70 * height
    front = center - 0.18 * depth_direction + np.array([0.0, 0.0, 0.10])
    retreat = front - 0.12 * depth_direction
    return front, center, retreat


def box_waypoints(
    points, *, interior_height: float, approach_height: float = 0.18
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build top-entry place waypoints from four axis-aligned fixture corners."""
    p0, px, py, pz = (np.asarray(value, dtype=float) for value in points)
    width, depth, height = px - p0, py - p0, pz - p0
    center = p0 + 0.5 * width + 0.5 * depth + interior_height * height
    front = center + np.array([0.0, 0.0, approach_height])
    retreat = center + np.array([0.0, 0.0, approach_height + 0.08])
    return front, center, retreat


def destination_waypoints(raw, task_class: str, object_position: np.ndarray):
    if task_class == "PickPlaceCounterToDrawer":
        return drawer_waypoints(raw.drawer)
    if task_class == "PickPlaceCounterToCabinet":
        return cabinet_waypoints(raw, depth_fraction=0.35)
    if task_class == "PickPlaceCounterToSink":
        region = next(iter(raw.sink.get_int_sites(relative=False).values()))
        return box_waypoints(region, interior_height=0.60)
    if task_class in {
        "PickPlaceCabinetToCounter",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
    }:
        return box_waypoints(raw.counter.get_ext_sites(relative=False), interior_height=1.02)
    if task_class == "PickPlaceFridgeDrawerToShelf":
        regions = raw.fridge.get_int_sites(relative=False)
        candidates = [
            points
            for name, points in regions.items()
            if "fridge" in name and "shelf" in name and "drawer" not in name
        ]
        if not candidates:
            raise ValueError("fridge has no shelf destination region")
        candidates.sort(
            key=lambda points: float(
                np.linalg.norm(
                    box_waypoints(points, interior_height=0.35)[1] - object_position
                )
            )
        )
        return box_waypoints(candidates[0], interior_height=0.35)
    raise ValueError(f"unsupported targeted pick/place task: {task_class}")


def configure_oracle_for_mapping(oracle, task_class: str, object_group: str) -> None:
    """Apply geometry-specific tolerances measured from failed simulator runs."""
    oracle.LIFT_HEIGHT = 0.14
    oracle.APPROACH_TOLERANCE = 0.055

    if task_class == "PickPlaceCabinetToCounter":
        # The Panda EEF origin remains about 9.7 cm in front of cabinet objects
        # even when its fingers are in contact; judge the reachable finger pose.
        oracle.APPROACH_HEIGHT = 0.05
        oracle.APPROACH_TOLERANCE = 0.110
        oracle.CONTACT_TOLERANCE = 0.110

    if task_class == "PickPlaceCounterToCabinet" and object_group == "boxed_food":
        # The box top stops the EEF about 4.5 cm above the body origin. Aim at
        # that reachable surface instead of accepting a pose far from target.
        oracle.GRASP_HEIGHT_OFFSET = 0.045
        oracle.CONTACT_TOLERANCE = 0.025
        # This layout has only about 7 cm of reachable vertical clearance.
        oracle.LIFT_HEIGHT = 0.070
        if hasattr(oracle, "CLOSED_TRANSLATION_LIMIT"):
            oracle.CLOSED_TRANSLATION_LIMIT = 0.08

    if task_class == "PickPlaceCounterToSink" and object_group == "bowl":
        # Closing inside the concave bowl cannot establish a grasp; use the rim.
        oracle.GRASP_HEIGHT_OFFSET = 0.045
        oracle.CONTACT_TOLERANCE = 0.040

    if object_group == "bar_soap":
        oracle.LIFT_HEIGHT = 0.08
        if hasattr(oracle, "CLOSED_TRANSLATION_LIMIT"):
            oracle.CLOSED_TRANSLATION_LIMIT = 0.08

    if object_group == "tongs":
        oracle.LIFT_HEIGHT = 0.06
        if hasattr(oracle, "CLOSED_TRANSLATION_LIMIT"):
            oracle.CLOSED_TRANSLATION_LIMIT = 0.08

    if task_class == "PickPlaceFridgeDrawerToShelf":
        oracle.APPROACH_HEIGHT = 0.10
        oracle.LIFT_HEIGHT = 0.06
        if hasattr(oracle, "CLOSED_TRANSLATION_LIMIT"):
            oracle.CLOSED_TRANSLATION_LIMIT = 0.08


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "outputs/midterm_testing_vla82/simulator_mapping_plan.json")
    parser.add_argument("--selection-id", default="VLA82-014")
    parser.add_argument("--seed", type=int, default=824141)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/vla82_targeted_expert")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party/robosuite"))
    sys.path.insert(0, str(ROOT / "third_party/robocasa"))
    from tools.vla82_pure_closed_loop import _configured_environment

    mapping = _mapping(args.plan, args.selection_id)
    args.output.mkdir(parents=True, exist_ok=True)
    frames, wrists, proprios, actions, phases = [], [], [], [], []
    diagnostics = []
    success = False
    rotation_template = None
    rotation_template_path = public_rotation_template_path(
        mapping["task_class"], mapping["object_group"]
    )
    if rotation_template_path is not None:
        public_actions = np.load(rotation_template_path, allow_pickle=False)
        close_indices = np.flatnonzero(np.asarray(public_actions)[:, 6] > 0)
        template_end = int(close_indices[0]) if len(close_indices) else len(public_actions)
        rotation_template = np.asarray(public_actions[:template_end], dtype=np.float32)
    with _configured_environment(mapping, args.seed) as environment:
        observation, _ = environment.reset(seed=args.seed)
        raw = environment.unwrapped.env
        robot = raw.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        initial_object = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
        front, center, retreat = destination_waypoints(
            raw, mapping["task_class"], initial_object
        )
        oracle_class = (
            AdaptiveLiftSafeCabinetPickPlaceOracle
            if mapping["task_class"] == "PickPlaceCounterToCabinet"
            else (
                OpenGripperPickPlaceOracle
                if mapping["task_class"] == "PickPlaceCounterToDrawer"
                else FastTransitPickPlaceOracle
            )
        )
        oracle = oracle_class(
            front, center, retreat, world_to_origin=controller.world_to_origin_frame
        )
        configure_oracle_for_mapping(
            oracle, mapping["task_class"], mapping["object_group"]
        )
        instruction = str(raw.get_ep_meta()["lang"])
        for step in range(args.max_steps):
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float)
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float)
            grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
            decision = oracle.decide(PickPlaceSnapshot(eef, obj, grasped, success))
            action = apply_rotation_template(decision.action, step, rotation_template)
            frames.append(np.asarray(observation[PRIMARY_KEY], dtype=np.uint8).copy())
            wrists.append(np.asarray(observation[WRIST_KEY], dtype=np.uint8).copy())
            proprios.append(proprio_from_observation(observation))
            actions.append(action.copy())
            phases.append(decision.phase)
            observation, _, terminated, truncated, info = environment.step(
                to_robocasa_action(action)
            )
            success = bool(info.get("success", False) or raw._check_success())
            diagnostics.append(
                {
                    "step": step,
                    "phase": decision.phase,
                    "grasped": grasped,
                    "eef": eef.tolist(),
                    "object": obj.tolist(),
                    "distance": float(np.linalg.norm(eef - obj)),
                    "success": success,
                }
            )
            if step % 25 == 0 or success:
                print(
                    f"step={step} phase={decision.phase} distance={diagnostics[-1]['distance']:.4f} "
                    f"grasped={grasped} success={success}",
                    flush=True,
                )
            if success or terminated or truncated:
                break
        final_success = bool(raw._check_success())
        success = bool(success or final_success)

    episode_path = args.output / f"{args.selection_id}_seed_{args.seed}.npz"
    np.savez_compressed(
        episode_path,
        primary=np.asarray(frames, dtype=np.uint8),
        wrist=np.asarray(wrists, dtype=np.uint8),
        proprio=np.asarray(proprios, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        phases=np.asarray(phases, dtype="U48"),
        instruction=np.asarray(
            instruction
        ),
    )
    report = {
        "status": "SUCCESS" if success else "FAILED",
        "success": success,
        "selection_id": args.selection_id,
        "seed": args.seed,
        "samples": len(actions),
        "task_class": mapping["task_class"],
        "object_group": mapping["object_group"],
        "uses_privileged_state_for_training_labels": True,
        "counts_as_pure_policy_acceptance": False,
        "public_rotation_template": (
            str(rotation_template_path.resolve())
            if rotation_template_path is not None
            else None
        ),
        "episode": str(episode_path.resolve()),
        "front": front.tolist(),
        "center": center.tolist(),
        "retreat": retreat.tolist(),
        "diagnostics": diagnostics,
    }
    report_path = args.output / f"{args.selection_id}_seed_{args.seed}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path.resolve(), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
