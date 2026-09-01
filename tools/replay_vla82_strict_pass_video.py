"""Replay one strict VLA82 PASS through public steps with the audit camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_vla82_full_simulation import _load_compiled_specs, _scene_for_spec
from tools.vla82_full_sim.environment import (
    _raw_environment,
    build_physics_capture_contract,
    make_environment,
    snapshot_from_environment,
)
from tools.vla82_full_sim.predicates import evaluate_operation
from tools.vla82_full_sim.expert import CleaningCoverageTracker, held_for_cleaning_credit


def camera_position_in_parent_frame(
    *,
    world_position: np.ndarray,
    parent_position: np.ndarray,
    parent_rotation: np.ndarray,
) -> np.ndarray:
    """Express a fixed world camera position in its moving parent frame."""
    return np.asarray(parent_rotation, dtype=float).T @ (
        np.asarray(world_position, dtype=float) - np.asarray(parent_position, dtype=float)
    )


def lock_camera_to_world(
    raw: object,
    *,
    camera: str,
    world_position: np.ndarray,
    focus: np.ndarray,
) -> None:
    """Aim a camera at the active operation while keeping its world pose fixed."""
    from robosuite.utils import transform_utils as transform

    camera_id = raw.sim.model.camera_name2id(camera)
    parent_id = int(raw.sim.model.cam_bodyid[camera_id])
    parent_position = np.asarray(raw.sim.data.body_xpos[parent_id], dtype=float)
    parent_rotation = np.asarray(raw.sim.data.body_xmat[parent_id], dtype=float).reshape(3, 3)
    position = np.asarray(world_position, dtype=float)
    forward = np.asarray(focus, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(right, forward), -forward))
    raw.sim.model.cam_pos[camera_id] = camera_position_in_parent_frame(
        world_position=position,
        parent_position=parent_position,
        parent_rotation=parent_rotation,
    )
    raw.sim.model.cam_quat[camera_id] = transform.convert_quat(
        transform.mat2quat(parent_rotation.T @ rotation), to="wxyz",
    )
    # Camera model fields are consumed directly by MuJoCo's renderer. Calling
    # sim.forward() here would also rebuild contact constraints and erase the
    # solver warm-start state, causing an otherwise deterministic grasp replay
    # to diverge from its captured public-step trajectory.


def presentation_occluder_names(selection_id: str) -> tuple[str, ...]:
    """Return render-only cabinet geometries that hide VLA82-004's operation."""
    selected = str(selection_id)
    if selected == "VLA82-004":
        return (
            "cab_1_right_group_left_door_g0",
            "cab_1_right_group_left_door_g1",
            "cab_1_right_group_right_door_g0",
            "cab_1_right_group_right_door_g1",
        )
    if selected == "VLA82-018":
        return (
            "cab_2_left_group_left_door_g0",
            "cab_2_left_group_left_door_g1",
            "cab_2_left_group_right_door_g0",
            "cab_2_left_group_right_door_g1",
        )
    return ()


def presentation_world_camera_position(
    selection_id: str, *, default: object,
) -> np.ndarray:
    """Place the cabinet-to-counter replay in front of the open cabinet."""
    if str(selection_id) == "VLA82-018":
        return np.array((1.40, -3.20, 2.20), dtype=float)
    return np.asarray(default, dtype=float)


def apply_presentation_occluder_mask(raw: object, selection_id: str) -> tuple[str, ...]:
    """Hide specified replay-only visual occluders without changing physics."""
    names = presentation_occluder_names(selection_id)
    for name in names:
        geom_id = raw.sim.model.geom_name2id(name)
        raw.sim.model.geom_rgba[geom_id, 3] = 0.0
    raw.sim.forward()
    return names


@dataclass(frozen=True)
class StrictPassSource:
    selection_id: str
    seed: int
    npz_path: Path
    json_path: Path
    report: dict[str, object]


def replay_coverage_tracker(spec: object, contract: object) -> CleaningCoverageTracker | None:
    if not (set(getattr(spec, "phases", ())) & {"wipe", "scrub"}):
        return None
    return CleaningCoverageTracker(next(iter(getattr(contract, "target_geometries"))))


def replay_coverage_credit(phase_index: int) -> bool:
    """Match capture semantics: only wipe/scrub-labelled public steps earn cells."""
    return int(phase_index) == 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_pass_source(npz_path: Path) -> StrictPassSource:
    npz_path = Path(npz_path).resolve()
    json_path = npz_path.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError("strict PASS replay requires matching NPZ and JSON")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("predicate_success") is not True:
        raise ValueError("refusing to replay a non-strict PASS episode")
    return StrictPassSource(
        selection_id=str(report["selection_id"]),
        seed=int(report["seed"]),
        npz_path=npz_path,
        json_path=json_path,
        report=report,
    )


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("strict replay produced no frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = map(int, frames[0].shape[:2])
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"empty strict replay video: {path}")


def replay_strict_pass(
    source: StrictPassSource,
    output: Path,
    *,
    fps: float = 12.0,
    frame_stride: int = 1,
    width: int = 1280,
    height: int = 720,
) -> Path:
    specs = {spec.selection_id: spec for spec in _load_compiled_specs()}
    spec = specs[source.selection_id]
    scene = _scene_for_spec(spec)
    with np.load(source.npz_path, allow_pickle=False) as data:
        actions = np.asarray(data["actions"], dtype=np.float32)
        phase_values = np.asarray(data["phases"], dtype=np.int16)
    environment = make_environment(scene, seed=source.seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".inprogress.mp4")
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    snapshots = []
    try:
        observation, _ = environment.reset(seed=source.seed)
        presentation_masked_geometries = apply_presentation_occluder_mask(
            _raw_environment(environment), source.selection_id
        )
        contract = build_physics_capture_contract(environment, scene)
        coverage_tracker = replay_coverage_tracker(spec, contract)
        initial = snapshot_from_environment(environment, 0, contract=contract)
        snapshots.append(initial)
        camera = scene.camera_names[0]
        # ``robot0_agentview_left`` is attached to the mobile base.  For a
        # cabinet extraction, keep the approved audit viewpoint in world space
        # so base motion cannot sweep the camera behind the cabinet door.
        world_locked_camera_position = None
        if source.selection_id in {"VLA82-004", "VLA82-018"}:
            world_locked_camera_position = presentation_world_camera_position(
                source.selection_id,
                default=environment.audit_camera["position"],
            )
        for index, action in enumerate(actions, start=1):
            observation, _, terminated, truncated, _ = environment.step(action)
            snapshot = snapshot_from_environment(environment, index, contract=contract)
            if coverage_tracker is not None:
                raw = _raw_environment(environment)
                object_geoms = tuple(getattr(contract, "object_geom_names", {}).get("obj", ()))
                try:
                    raw_grasp = bool(raw._check_grasp(raw.robots[0].gripper["right"], raw.objects["obj"]))
                    from tools.vla82_full_sim.expert import _selected_geom_has_two_finger_contacts
                    held_tool = held_for_cleaning_credit(
                        raw_grasp=raw_grasp,
                        two_pad_contact=_selected_geom_has_two_finger_contacts(raw, object_geoms),
                    )
                except (AttributeError, KeyError, TypeError):
                    held_tool = False
                snapshot = coverage_tracker.derive(
                    snapshot,
                    credit=held_tool and replay_coverage_credit(int(phase_values[index - 1])),
                )
            snapshots.append(snapshot)
            if (index - 1) % max(int(frame_stride), 1) == 0:
                raw = _raw_environment(environment)
                if world_locked_camera_position is not None:
                    target_object = raw.objects["obj"]
                    target_body = raw.sim.model.body_name2id(target_object.root_body)
                    target = np.asarray(raw.sim.data.body_xpos[target_body], dtype=float)
                    robot = raw.robots[0]
                    eef = np.asarray(raw.sim.data.site_xpos[robot.eef_site_id["right"]], dtype=float)
                    lock_camera_to_world(
                        raw,
                        camera=camera,
                        world_position=world_locked_camera_position,
                        focus=.5 * (target + eef),
                    )
                frame = np.asarray(raw.sim.render(
                    width=int(width), height=int(height), camera_name=camera,
                )[::-1], dtype=np.uint8)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(temporary_output), cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps), (int(width), int(height)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer: {temporary_output}")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1
            if terminated or truncated:
                raise RuntimeError(f"strict replay terminated at public step {index}")
        predicate = evaluate_operation(spec, initial, snapshots[1:], snapshots[-1])
        if not bool(predicate.success):
            coverage = None if coverage_tracker is None else {
                "covered_cells": coverage_tracker.covered_cells,
                "evidence_steps": len(coverage_tracker.evidence),
            }
            raise RuntimeError(f"strict replay predicate failed: {predicate.errors}; coverage={coverage}")
        if writer is None or frame_count == 0:
            raise RuntimeError("strict replay produced no frames")
        writer.release()
        writer = None
        temporary_output.replace(output)
        sidecar = Path(output).with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "selection_id": source.selection_id,
                    "seed": source.seed,
                    "status": "PASS",
                    "predicate_success": True,
                    "source_strict_pass_json": str(source.json_path),
                    "source_strict_pass_json_sha256": _sha256(source.json_path),
                    "source_actions_npz": str(source.npz_path),
                    "source_actions_npz_sha256": _sha256(source.npz_path),
                    "render_mode": "same simulator public-step replay; configured audit camera (world-locked for mobile cabinet extraction); render-only cabinet occluder mask where required",
                    "presentation_masked_geometries": presentation_masked_geometries,
                    "camera": camera,
                    "frames": frame_count,
                    "physics_steps": len(actions),
                    "fps": float(fps),
                    "width": int(width),
                    "height": int(height),
                    "friction_evidence": getattr(environment, "friction_evidence", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return Path(output).resolve()
    finally:
        if writer is not None:
            writer.release()
        if temporary_output.exists():
            temporary_output.unlink()
        environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    source = load_strict_pass_source(args.episode)
    print(replay_strict_pass(
        source, args.output, fps=args.fps, frame_stride=args.frame_stride,
        width=args.width, height=args.height,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
