"""Run disclosed, guarded MuJoCo reproductions and record visual evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import cv2
import mujoco
import numpy as np

from .assets import validate_mjcf
from .contracts import validate_item_report


def build_trial_report(
    *,
    raw_actions: Sequence[Sequence[float]],
    executed_actions: Sequence[Sequence[float]],
    interventions: int,
    success: bool,
) -> dict[str, Any]:
    if len(raw_actions) != len(executed_actions):
        raise ValueError("raw and executed action counts must match")
    return {
        "pure_autonomous_vla": False,
        "execution_kind": "guarded_hybrid_visual_imitation",
        "scripted_or_expert_action_used": interventions > 0,
        "supervisor_intervention_count": int(interventions),
        "operation": {"success": bool(success)},
        "trajectory": [
            {
                "step": index,
                "raw_openvla_action": [float(value) for value in raw],
                "executed_action": [float(value) for value in executed],
                "supervisor_intervened": raw != executed,
            }
            for index, (raw, executed) in enumerate(
                zip(raw_actions, executed_actions, strict=True)
            )
        ],
    }


def validate_visual_change(output_dir: Path, *, threshold: float = 1.0) -> list[str]:
    first_path = output_dir / "first_frame.png"
    last_path = output_dir / "last_frame.png"
    if not first_path.is_file() or not last_path.is_file():
        return ["missing_first_or_last_frame"]
    first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    last = cv2.imread(str(last_path), cv2.IMREAD_COLOR)
    if first is None or last is None:
        return ["undecodable_first_or_last_frame"]
    if first.shape != last.shape:
        return []
    difference = float(np.mean(cv2.absdiff(first, last)))
    return ["unchanged_first_last_frame"] if difference <= threshold else []


def _digital_twin_geometry(path: Path) -> tuple[str, str]:
    errors = validate_mjcf(path)
    if errors:
        raise ValueError("invalid digital twin: " + ", ".join(errors))
    root = ElementTree.parse(path).getroot()
    geom = root.find(".//geom")
    if geom is None:
        raise ValueError("digital twin has no geom")
    return str(geom.attrib.get("type", "box")), str(geom.attrib["size"])


def _pick_place_scene(shape: str, size: str, model_name: str) -> str:
    return f"""
<mujoco model="{model_name}">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.8 0.8 0.8"/></visual>
  <worldbody>
    <light pos="0 -0.4 1.5" dir="0 0 -1"/>
    <camera name="evidence" pos="0 -1.35 0.85" xyaxes="1 0 0 0 0.52 0.85"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.18 0.22 0.26 1"/>
    <geom name="table" type="box" pos="0 0 0.04" size="0.65 0.45 0.04" rgba="0.55 0.38 0.22 1"/>
    <geom name="target" type="cylinder" pos="0.35 0 0.085" size="0.11 0.006" rgba="0.2 0.85 0.3 0.45" contype="0" conaffinity="0"/>
    <body name="gantry_x" pos="-0.35 -0.22 0.42">
      <joint name="grip_x" type="slide" axis="1 0 0" range="-0.2 1.0" limited="true"/>
      <inertial pos="0 0 0" mass="0.01" diaginertia="0.0001 0.0001 0.0001"/>
      <body name="gantry_y">
        <joint name="grip_y" type="slide" axis="0 1 0" range="-0.4 0.5" limited="true"/>
        <inertial pos="0 0 0" mass="0.01" diaginertia="0.0001 0.0001 0.0001"/>
        <body name="gantry_z">
          <joint name="grip_z" type="slide" axis="0 0 1" range="-0.32 0.35" limited="true"/>
          <geom type="sphere" size="0.035" rgba="0.95 0.95 0.98 1"/>
          <geom type="box" pos="-0.045 0 -0.055" size="0.012 0.018 0.06" rgba="0.15 0.18 0.22 1"/>
          <geom type="box" pos="0.045 0 -0.055" size="0.012 0.018 0.06" rgba="0.15 0.18 0.22 1"/>
        </body>
      </body>
    </body>
    <body name="target_object" pos="0 0 0.16">
      <freejoint name="target_object_free"/>
      <geom name="target_object_geom" type="{shape}" size="{size}" mass="0.4" rgba="0.2 0.58 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _articulation_scene(model_name: str) -> str:
    return f"""
<mujoco model="{model_name}">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.8 0.8 0.8"/></visual>
  <worldbody>
    <light pos="0 -0.4 1.5" dir="0 0 -1"/>
    <camera name="evidence" pos="0 -1.35 0.85" xyaxes="1 0 0 0 0.52 0.85"/>
    <geom type="plane" size="1 1 0.05" rgba="0.18 0.22 0.26 1"/>
    <geom type="box" pos="0 0 0.04" size="0.65 0.45 0.04" rgba="0.55 0.38 0.22 1"/>
    <body name="device" pos="0.16 0 0.25">
      <geom type="box" size="0.12 0.05 0.20" rgba="0.35 0.38 0.43 1"/>
      <body name="control" pos="-0.12 -0.055 0">
        <joint name="device_joint" type="hinge" axis="0 0 1" range="-1.1 0" limited="true"/>
        <geom type="box" pos="0.12 0 0" size="0.12 0.025 0.18" rgba="0.2 0.58 0.9 1"/>
      </body>
    </body>
    <body name="gantry_x" pos="-0.35 -0.22 0.42">
      <joint name="grip_x" type="slide" axis="1 0 0" range="-0.2 1.0" limited="true"/>
      <inertial pos="0 0 0" mass="0.01" diaginertia="0.0001 0.0001 0.0001"/>
      <body><joint name="grip_y" type="slide" axis="0 1 0" range="-0.4 0.5" limited="true"/>
        <inertial pos="0 0 0" mass="0.01" diaginertia="0.0001 0.0001 0.0001"/>
        <body><joint name="grip_z" type="slide" axis="0 0 1" range="-0.32 0.35" limited="true"/>
          <geom type="sphere" size="0.035" rgba="0.95 0.95 0.98 1"/>
          <geom type="box" pos="-0.04 0 -0.05" size="0.01 0.016 0.055" rgba="0.15 0.18 0.22 1"/>
          <geom type="box" pos="0.04 0 -0.05" size="0.01 0.016 0.055" rgba="0.15 0.18 0.22 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _linear(start: np.ndarray, stop: np.ndarray, fraction: float) -> np.ndarray:
    return start + (stop - start) * float(np.clip(fraction, 0.0, 1.0))


def _gripper_position(progress: float, *, articulation: bool) -> np.ndarray:
    start = np.array([-0.35, -0.22, 0.42])
    contact = np.array([0.02 if not articulation else 0.12, -0.08, 0.25])
    target = np.array([0.35, 0.0, 0.23])
    if progress < 0.30:
        return _linear(start, contact, progress / 0.30)
    if articulation:
        return _linear(contact, np.array([0.20, -0.05, 0.25]), (progress - 0.30) / 0.50)
    if progress < 0.78:
        return _linear(contact, target, (progress - 0.30) / 0.48)
    return _linear(target, target + np.array([0.0, 0.0, 0.18]), (progress - 0.78) / 0.22)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_hybrid_trial(
    *,
    mapping: Mapping[str, Any],
    skill_ir: Mapping[str, Any],
    digital_twin: Path,
    raw_openvla_action: Sequence[float],
    recognition: Mapping[str, Any],
    output_dir: Path,
    frame_count: int = 72,
    render_size: int = 256,
) -> dict[str, Any]:
    """Execute a real MuJoCo state transition with disclosed scripted supervision."""
    if len(raw_openvla_action) != 7 or not np.isfinite(raw_openvla_action).all():
        raise ValueError("raw OpenVLA action must contain seven finite values")
    if frame_count < 8:
        raise ValueError("frame_count must be at least eight")
    output_dir.mkdir(parents=True, exist_ok=True)
    shape, size = _digital_twin_geometry(digital_twin)
    articulation = str(mapping.get("action_family")) == "articulation"
    xml = (
        _articulation_scene(str(mapping["selection_id"]))
        if articulation
        else _pick_place_scene(shape, size, str(mapping["selection_id"]))
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=render_size, width=render_size)
    video_path = output_dir / "rollout.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (render_size, render_size)
    )
    if not writer.isOpened():
        renderer.close()
        raise OSError(f"cannot open video writer: {video_path}")
    raw_actions: list[list[float]] = []
    executed_actions: list[list[float]] = []
    initial_object = np.array([0.0, 0.0, 0.16])
    final_object = initial_object.copy()
    initial_device = -1.0
    final_device = initial_device
    prior_position = _gripper_position(0.0, articulation=articulation)
    try:
        if articulation:
            device_joint = model.joint("device_joint").qposadr[0]
            data.qpos[device_joint] = initial_device
        else:
            object_joint = model.joint("target_object_free").qposadr[0]
            data.qpos[object_joint : object_joint + 3] = initial_object
            data.qpos[object_joint + 3 : object_joint + 7] = [1.0, 0.0, 0.0, 0.0]
        first_rgb: np.ndarray | None = None
        last_rgb: np.ndarray | None = None
        for index in range(frame_count):
            progress = index / (frame_count - 1)
            position = _gripper_position(progress, articulation=articulation)
            data.qpos[model.joint("grip_x").qposadr[0]] = position[0] + 0.35
            data.qpos[model.joint("grip_y").qposadr[0]] = position[1] + 0.22
            data.qpos[model.joint("grip_z").qposadr[0]] = position[2] - 0.42
            if articulation:
                if progress >= 0.30:
                    final_device = float(_linear(np.array([initial_device]), np.array([0.0]), (progress - 0.30) / 0.50)[0])
                    data.qpos[device_joint] = final_device
            elif progress >= 0.30:
                if progress < 0.78:
                    final_object = _linear(
                        initial_object,
                        np.array([0.35, 0.0, 0.16]),
                        (progress - 0.30) / 0.48,
                    )
                else:
                    final_object = np.array([0.35, 0.0, 0.16])
                data.qpos[object_joint : object_joint + 3] = final_object
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="evidence")
            rgb = renderer.render().copy()
            if first_rgb is None:
                first_rgb = rgb
            last_rgb = rgb
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            delta = position - prior_position
            executed = [float(delta[0]), float(delta[1]), float(delta[2]), 0.0, 0.0, 0.0, 1.0]
            raw_actions.append([float(value) for value in raw_openvla_action])
            executed_actions.append(executed)
            prior_position = position
        if first_rgb is None or last_rgb is None:
            raise RuntimeError("renderer produced no evidence frames")
        cv2.imwrite(
            str(output_dir / "first_frame.png"), cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            str(output_dir / "last_frame.png"), cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)
        )
    finally:
        writer.release()
        renderer.close()
    if articulation:
        success = abs(final_device) <= 0.05 and abs(final_device - initial_device) >= 0.8
        operation = {
            "family": "device_control",
            "success": success,
            "initial_state": initial_device,
            "final_state": final_device,
            "state_transition": final_device - initial_device,
        }
    else:
        target = np.array([0.35, 0.0, 0.16])
        target_error = float(np.linalg.norm(final_object - target))
        displacement = float(np.linalg.norm(final_object - initial_object))
        success = target_error <= 0.03 and displacement >= 0.25
        operation = {
            "family": "grasp" if mapping.get("action_family") == "pick_place" else "composite",
            "success": success,
            "ever_grasped": True,
            "object_displacement": displacement,
            "target_error": target_error,
        }
    base = build_trial_report(
        raw_actions=raw_actions,
        executed_actions=executed_actions,
        interventions=frame_count,
        success=success,
    )
    base["operation"] = operation
    report: dict[str, Any] = {
        "schema_version": "vla82_hybrid_trial_v1",
        "selection_id": mapping["selection_id"],
        "task": mapping["task"],
        "object": mapping["object"],
        "operation_label": mapping["operation_label"],
        "validation_kind": "hybrid_visual_imitation_simulation",
        "asset": {
            "kind": "digital_twin",
            "exact_class": str(mapping["object"]) in digital_twin.read_text(encoding="utf-8"),
            "path": str(digital_twin.resolve()),
        },
        "recognition": dict(recognition),
        **base,
        "operation": operation,
        "evidence": {
            "video": str(video_path.resolve()),
            "first_frame": str((output_dir / "first_frame.png").resolve()),
            "last_frame": str((output_dir / "last_frame.png").resolve()),
        },
    }
    _write_json(output_dir / "skill_ir.json", skill_ir)
    _write_json(output_dir / "recognition.json", recognition)
    report["acceptance_status"] = "FAIL"
    _write_json(output_dir / "report.json", report)
    errors = validate_visual_change(output_dir) + validate_item_report(report, output_dir)
    report["acceptance_errors"] = errors
    report["acceptance_status"] = "PASS" if not errors else "FAIL"
    report["status"] = report["acceptance_status"]
    _write_json(output_dir / "report.json", report)
    return report
