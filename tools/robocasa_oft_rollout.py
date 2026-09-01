"""Pure-policy RoboCasa rollout using 8-step OpenVLA-OFT action chunks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.robocasa_oft_contract import extract_proprio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "midterm_testing_vla82" / "oft_r1_representative"
PRIMARY_KEY = "video.robot0_agentview_left"
WRIST_KEY = "video.robot0_eye_in_hand"


def validate_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    values = np.asarray(action_chunk, dtype=np.float32)
    if values.shape != (8, 7):
        raise ValueError(f"action chunk must have shape (8, 7), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("action chunk must be finite")
    return values


def proprio_from_observation(observation: dict[str, Any]) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(observation["state.base_position"], dtype=np.float32),
            np.asarray(observation["state.base_rotation"], dtype=np.float32),
            np.asarray(observation["state.end_effector_position_relative"], dtype=np.float32),
            np.asarray(observation["state.end_effector_rotation_relative"], dtype=np.float32),
            np.asarray(observation["state.gripper_qpos"], dtype=np.float32),
        ]
    )
    return extract_proprio(state)


def _rgb(observation: dict[str, Any], key: str) -> np.ndarray:
    image = np.asarray(observation[key], dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(f"invalid RGB observation {key}: {image.shape}")
    return image.copy()


def _mapping(plan_path: Path, selection_id: str) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matches = [item for item in plan["mappings"] if item["selection_id"] == selection_id]
    if len(matches) != 1:
        raise ValueError(f"selection ID must map exactly once: {selection_id}")
    return matches[0]


def run_trial(
    mapping: dict[str, Any],
    *,
    output_root: Path,
    host: str,
    port: int,
    max_decisions: int,
    execute_per_chunk: int,
    translation_gain: float,
    rotation_gain: float,
    gripper_threshold: float | None,
    diagnostic_state: bool,
    seed: int,
) -> dict[str, Any]:
    import imageio.v3 as iio
    from tools.robocasa_oft_tcp_client import predict_chunk
    from tools.vla82_pure_closed_loop import _configured_environment

    trial_dir = output_root / mapping["selection_id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    primary_path = trial_dir / "current_primary.png"
    wrist_path = trial_dir / "current_wrist.png"
    first_path = trial_dir / "first_frame.png"
    last_path = trial_dir / "last_frame.png"
    video_path = trial_dir / "rollout.mp4"
    report_path = trial_dir / "report.json"
    frames: list[np.ndarray] = []
    chunks: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    success = False
    terminated = truncated = False

    with _configured_environment(mapping, seed) as environment:
        observation, reset_info = environment.reset(seed=seed)
        raw = environment.unwrapped.env
        instruction = str(raw.get_ep_meta().get("lang", "")).strip()
        if not instruction:
            raise RuntimeError("RoboCasa returned an empty instruction")
        frames.append(_rgb(observation, PRIMARY_KEY))
        iio.imwrite(first_path, frames[0])
        executed = 0
        request_number = 0
        while executed < max_decisions and not (success or terminated or truncated):
            primary = _rgb(observation, PRIMARY_KEY)
            wrist = _rgb(observation, WRIST_KEY)
            iio.imwrite(primary_path, primary)
            iio.imwrite(wrist_path, wrist)
            proprio = proprio_from_observation(observation)
            started = time.perf_counter()
            chunk = predict_chunk(
                host,
                port,
                {
                    "image_path": str(primary_path.resolve()),
                    "wrist_image_path": str(wrist_path.resolve()),
                    "proprio": proprio.tolist(),
                    "instruction": instruction,
                    "relation_key": mapping["selection_id"],
                },
            )
            latency = time.perf_counter() - started
            chunks.append(
                {
                    "request": request_number,
                    "at_executed_step": executed,
                    "inference_seconds": latency,
                    "predicted_action_chunk": chunk.tolist(),
                }
            )
            request_number += 1
            for chunk_index, row in enumerate(chunk[:execute_per_chunk]):
                if executed >= max_decisions:
                    break
                scaled = np.asarray(row, dtype=np.float32).copy()
                scaled[:3] *= translation_gain
                scaled[3:6] *= rotation_gain
                if gripper_threshold is not None:
                    scaled[6] = 1.0 if row[6] >= gripper_threshold else -1.0
                bounded = np.clip(scaled, -1.0, 1.0).astype(np.float32)
                observation, _, terminated, truncated, info = environment.step(
                    to_robocasa_action(bounded)
                )
                executed += 1
                success = bool(info.get("success", False) or raw._check_success())
                frames.append(_rgb(observation, PRIMARY_KEY))
                trajectory.append(
                    {
                        "executed_step": executed,
                        "chunk_request": request_number - 1,
                        "chunk_index": chunk_index,
                        "predicted_action": [float(value) for value in row],
                        "scaled_action": [float(value) for value in scaled],
                        "executed_bounded_action": [float(value) for value in bounded],
                        "success": success,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        **(
                            {
                                "diagnostic_eef_position": np.asarray(
                                    raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]],
                                    dtype=float,
                                ).tolist(),
                                "diagnostic_object_position": np.asarray(
                                    raw.sim.data.body_xpos[raw.obj_body_id["obj"]], dtype=float
                                ).tolist(),
                                "diagnostic_grasped": bool(
                                    raw._check_grasp(
                                        raw.robots[0].gripper["right"], raw.objects["obj"]
                                    )
                                ),
                            }
                            if diagnostic_state
                            else {}
                        ),
                    }
                )
                if success or terminated or truncated:
                    break
            print(
                f"{mapping['selection_id']} actions={executed}/{max_decisions} "
                f"requests={request_number} latency={latency:.3f}s success={success}",
                flush=True,
            )
        success = bool(success or raw._check_success())

    iio.imwrite(last_path, frames[-1])
    iio.imwrite(video_path, np.stack(frames), fps=20)
    result = {
        "schema_version": "robocasa_oft_pure_rollout_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "SUCCESS" if success else "TASK_FAILED",
        "selection_id": mapping["selection_id"],
        "source_table": mapping["source_table"],
        "task": mapping["task"],
        "object": mapping["object"],
        "task_class": mapping["task_class"],
        "object_group": mapping.get("object_group"),
        "mapping_mode": mapping["mapping_mode"],
        "proxy_disclosed": mapping["proxy_disclosed"],
        "instruction": instruction,
        "seed": seed,
        "success": success,
        "actual_closed_loop": True,
        "pure_autonomous_vla": True,
        "oracle_or_scripted_action_used": False,
        "expert_recovery_used": False,
        "simulator_success_predicate_checked": True,
        "diagnostic_privileged_state_logged": diagnostic_state,
        "counts_as_acceptance_trial": not diagnostic_state,
        "executed_decision_steps": len(trajectory),
        "inference_requests": len(chunks),
        "action_chunk_size": 8,
        "execute_per_chunk": execute_per_chunk,
        "fixed_action_adapter": {
            "translation_gain": translation_gain,
            "rotation_gain": rotation_gain,
            "gripper_threshold": gripper_threshold,
            "uses_privileged_state": False,
        },
        "max_decision_steps": max_decisions,
        "reset_info": {str(key): str(value) for key, value in reset_info.items()},
        "video": str(video_path.resolve()),
        "first_frame": str(first_path.resolve()),
        "last_frame": str(last_path.resolve()),
        "predicted_chunks": chunks,
        "trajectory": trajectory,
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["report"] = str(report_path.resolve())
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--selection-id", default="VLA82-014")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--execute-per-chunk", type=int, default=8)
    parser.add_argument("--translation-gain", type=float, default=1.0)
    parser.add_argument("--rotation-gain", type=float, default=1.0)
    parser.add_argument("--gripper-threshold", type=float)
    parser.add_argument("--diagnostic-state", action="store_true")
    parser.add_argument("--seed", type=int, default=824141)
    args = parser.parse_args(argv)
    if (
        args.max_decisions < 1
        or not 1 <= args.execute_per_chunk <= 8
        or min(args.translation_gain, args.rotation_gain) <= 0
        or (
            args.gripper_threshold is not None
            and not -1.0 <= args.gripper_threshold <= 1.0
        )
    ):
        raise ValueError("max-decisions must be positive and execute-per-chunk must be 1..8")
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_trial(
        _mapping(args.plan, args.selection_id),
        output_root=args.output,
        host=args.host,
        port=args.port,
        max_decisions=args.max_decisions,
        execute_per_chunk=args.execute_per_chunk,
        translation_gain=args.translation_gain,
        rotation_gain=args.rotation_gain,
        gripper_threshold=args.gripper_threshold,
        diagnostic_state=args.diagnostic_state,
        seed=args.seed,
    )
    manifest = {
        "completed_count": 1,
        "success_count": int(result["success"]),
        "failure_count": int(not result["success"]),
        "all_tasks_successful": bool(result["success"]),
        "results": [result],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print((args.output / "manifest.json").resolve(), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
