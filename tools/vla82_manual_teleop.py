"""Manually operate any authoritative VLA82 scene with a RoboSuite keyboard."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RECORD_ROOT = PROJECT_ROOT / "outputs" / "manual_vla82_teleop"
DEFAULT_CAMERA = "robot0_agentview_left"


def resolve_selection(selection_id: str, specs: Sequence[Any]) -> Any:
    """Return one source-registered VLA82 specification by identifier."""
    normalized = str(selection_id).strip().upper()
    for spec in specs:
        if str(spec.selection_id).upper() == normalized:
            return spec
    raise ValueError(f"unknown VLA82 selection: {selection_id}")


def build_episode_metadata(
    *,
    selection_id: str,
    seed: int,
    task_class: str,
    camera: str,
    action_count: int,
    request: Any,
    fingerprint: Mapping[str, Any],
    friction_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe a manual capture without assigning an acceptance result."""
    return {
        "status": "manual_unverified",
        "selection_id": str(selection_id),
        "seed": int(seed),
        "task_class": str(task_class),
        "camera": str(camera),
        "action_count": int(action_count),
        "manipulated_objects": list(getattr(request, "manipulated_objects", ())),
        "source_fixture": str(getattr(request, "source_fixture", "")),
        "target_fixture": str(getattr(request, "target_fixture", "")),
        "target_relation": str(getattr(request, "target_relation", "")),
        "scene_sha256": str(fingerprint.get("sha256", "")),
        "runtime_scene_fingerprint": dict(fingerprint),
        "friction_evidence": dict(friction_evidence),
        "controls": {
            "b": "toggle arm/base mode",
            "arrows": "arm XY or mobile-base XY in base mode",
            "semicolon_period": "arm Z; torso Z in base mode when available",
            "o_p": "yaw; mobile-base yaw in base mode",
            "space": "toggle gripper",
            "q": "finish and save the current manual capture",
        },
    }


def save_episode(
    directory: Path,
    actions: Sequence[np.ndarray],
    timestamps: Sequence[float],
    metadata: Mapping[str, Any],
    model_xml: str,
) -> Path:
    """Write public action vectors and scene provenance for one manual capture."""
    if not actions:
        raise ValueError("manual capture requires at least one action")
    if len(actions) != len(timestamps):
        raise ValueError("manual capture action and timestamp counts differ")
    action_array = np.stack([np.asarray(action, dtype=np.float32).reshape(-1) for action in actions])
    if action_array.ndim != 2 or not np.isfinite(action_array).all():
        raise ValueError("manual capture actions must be finite equal-length vectors")
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "actions.npz",
        actions=action_array,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        step_indexes=np.arange(action_array.shape[0], dtype=np.int32),
    )
    (output / "episode.json").write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "model.xml").write_text(str(model_xml), encoding="utf-8")
    return output


def _episode_directory(record_root: Path, selection_id: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(record_root) / str(selection_id).upper() / f"{stamp}-manual"


def _model_xml(raw: Any) -> str:
    get_xml = getattr(getattr(raw, "model", None), "get_xml", None)
    return str(get_xml()) if callable(get_xml) else ""


def _keyboard_action(raw: Any, device: Any, previous_gripper_actions: Sequence[dict[str, np.ndarray]]) -> np.ndarray | None:
    """Mirror RoboCasa's public teleoperation action assembly for PandaOmron."""
    input_action = device.input2action(mirror_actions=True)
    if input_action is None:
        return None
    active_robot = raw.robots[device.active_robot]
    action_dict = deepcopy(input_action)
    for arm in active_robot.arms:
        input_type = active_robot.part_controllers[arm].input_type
        if input_type == "delta":
            action_dict[arm] = input_action[f"{arm}_delta"]
        elif input_type == "absolute":
            action_dict[arm] = input_action[f"{arm}_abs"]
        else:
            raise RuntimeError(f"unsupported controller input type: {input_type}")
    actions = [robot.create_action_vector(previous_gripper_actions[index]) for index, robot in enumerate(raw.robots)]
    actions[device.active_robot] = active_robot.create_action_vector(action_dict)
    return np.concatenate(actions).astype(np.float32, copy=False)


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        return
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = map(int, np.asarray(frames[0]).shape[:2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def run_manual_episode(
    scene: Any,
    *,
    seed: int,
    record_root: Path,
    camera: str | None = None,
    record_video: bool = False,
    max_steps: int = 4_000,
    pos_sensitivity: float = 1.0,
    rot_sensitivity: float = 1.0,
) -> Path:
    """Open a visible VLA82 scene, collect keyboard control, and save the episode."""
    from robosuite.devices import Keyboard
    from tools.vla82_full_sim.environment import _raw_environment, make_environment

    selected_camera = str(camera or scene.camera_names[0])
    if selected_camera not in scene.camera_names:
        raise ValueError(f"camera is unavailable for {scene.selection_id}: {selected_camera}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    environment = make_environment(scene, int(seed), has_renderer=True, renderer="mjviewer")
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    frames: list[np.ndarray] = []
    try:
        environment.reset(seed=int(seed))
        raw = _raw_environment(environment)
        device = Keyboard(env=raw, pos_sensitivity=float(pos_sensitivity), rot_sensitivity=float(rot_sensitivity))
        previous_gripper_actions = [
            {
                f"{arm}_gripper": np.zeros(robot.gripper[arm].dof, dtype=np.float32)
                for arm in robot.arms
                if robot.gripper[arm].dof > 0
            }
            for robot in raw.robots
        ]
        raw.render()
        device.start_control()
        print("Manual control ready. Press b for base mode; q saves and exits.", flush=True)
        started = time.monotonic()
        for _ in range(int(max_steps)):
            action = _keyboard_action(raw, device, previous_gripper_actions)
            if action is None:
                break
            action = np.clip(action, environment.action_space.low, environment.action_space.high)
            environment.step(action)
            actions.append(action.copy())
            timestamps.append(time.monotonic() - started)
            raw.render()
            if record_video:
                frame = np.asarray(raw.sim.render(width=1280, height=720, camera_name=selected_camera)[::-1], dtype=np.uint8)
                frames.append(frame)
            time.sleep(1.0 / 30.0)
        if not actions:
            raise RuntimeError("no actions were captured; press a movement, gripper, or base-control key before q")
        metadata = build_episode_metadata(
            selection_id=scene.selection_id,
            seed=int(seed),
            task_class=scene.task_class,
            camera=selected_camera,
            action_count=len(actions),
            request=scene,
            fingerprint=getattr(environment, "runtime_scene_fingerprint", {}),
            friction_evidence=getattr(environment, "friction_evidence", {}),
        )
        output = save_episode(
            _episode_directory(record_root, scene.selection_id), actions, timestamps, metadata, _model_xml(raw)
        )
        if record_video:
            _write_video(output / "manual.mp4", frames, fps=30.0)
        print(f"Manual episode saved: {output}", flush=True)
        return output
    finally:
        environment.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-id", required=True, help="selection ID present in the installed VLA82 registry")
    parser.add_argument("--seed", type=int, default=None, help="scene seed; defaults to the numeric VLA82 identifier")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--camera", default=None, help=f"camera for optional video; default: {DEFAULT_CAMERA}")
    parser.add_argument("--record-video", action="store_true", help="also save a 1280x720 MP4")
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from tools.run_vla82_full_simulation import _load_compiled_specs, _scene_for_spec

    spec = resolve_selection(args.selection_id, _load_compiled_specs())
    scene = _scene_for_spec(spec)
    seed = int(args.seed) if args.seed is not None else int(scene.selection_id.rsplit("-", 1)[1]) + 820_000
    run_manual_episode(
        scene,
        seed=seed,
        record_root=args.record_dir,
        camera=args.camera,
        record_video=bool(args.record_video),
        max_steps=int(args.max_steps),
        pos_sensitivity=float(args.pos_sensitivity),
        rot_sensitivity=float(args.rot_sensitivity),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
