"""Run a 300-decision RoboCasa closed loop with the formal OpenVLA service."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.openvla_simulator_adapter import to_robocasa_action
from tools.openvla_tcp_client import predict


ROOT = Path(__file__).resolve().parents[1]
RELATION_KEY = "organizing::toy"
CANONICAL_ACTIONS = ["locate(toy)", "grasp(toy)", "move(storage)", "place(toy)"]
PHASES = frozenset({"locate", "grasp", "move", "place"})


def next_canonical_phase(
    current: str,
    *,
    object_eef_distance: float,
    grasped: bool,
    inside: bool,
    success: bool,
) -> str:
    if current not in PHASES:
        raise ValueError(f"unsupported current canonical phase: {current!r}")
    if success:
        return "done"
    if inside:
        return "place"
    if grasped:
        return "move"
    if current in {"grasp", "move"} and object_eef_distance > 0.06:
        return "locate"
    if object_eef_distance <= 0.04:
        return "grasp"
    return "locate"


def guarded_action(raw_action: Sequence[float], phase: str) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (7,):
        raise ValueError("formal OpenVLA action must contain seven values")
    if not np.isfinite(action).all():
        raise ValueError("formal OpenVLA action must be finite")
    if phase not in PHASES:
        raise ValueError(f"unsupported canonical phase: {phase!r}")
    translation_limit = {"locate": 1.0, "grasp": 0.25, "move": 0.30, "place": 0.50}[phase]
    result = np.zeros(7, dtype=np.float32)
    result[:3] = np.clip(action[:3], -translation_limit, translation_limit)
    result[6] = 1.0 if phase in {"grasp", "move"} else -1.0
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_fingerprint(adapter_dir: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in Path(adapter_dir).rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"formal adapter has no files: {adapter_dir}")
    for path in files:
        digest.update(path.relative_to(adapter_dir).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _manifest_split(manifest_path: Path, seed: int) -> tuple[str, str]:
    raw = Path(manifest_path).read_bytes()
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
        "--output-root", type=Path, default=ROOT / "outputs" / "formal_skill_eval"
    )
    parser.add_argument("--allow-non-heldout", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.steps < 1 or args.action_repeat < 1:
        raise ValueError("seed must be non-negative; steps and action-repeat must be positive")
    split, manifest_hash = _manifest_split(args.manifest, args.seed)
    if split != "held_out" and not args.allow_non_heldout:
        raise ValueError(
            f"seed {args.seed} belongs to split {split!r}; formal evaluation requires held_out"
        )
    skill_bytes = args.skill_ir.read_bytes()
    skill_ir = json.loads(skill_bytes.decode("utf-8"))
    if skill_ir.get("relation_key") != RELATION_KEY:
        raise ValueError("Skill IR relation mismatch")
    instruction = str(skill_ir.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("Skill IR instruction is empty")

    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    import imageio.v3 as iio
    from robocasa.utils import object_utils as OU
    from tools.skill_transfer.robocasa_envs import create_formal_env

    output_dir = args.output_root / f"seed_{args.seed:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "current_frame.png"
    video_path = output_dir / "rollout.mp4"
    report_path = output_dir / "report.json"
    env = None
    trajectory: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []
    phase_counts: Counter[str] = Counter()
    inference_seconds: list[float] = []
    success = False
    ever_grasped = False
    ever_inside = False
    executed_decisions = 0
    executed_sim_steps = 0
    final_grasped = False
    final_inside = False
    try:
        env, environment_mapping = create_formal_env(RELATION_KEY, seed=args.seed)
        observation, _ = env.reset(seed=args.seed)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        object_body_id = raw.obj_body_id["obj"]
        object_model = raw.objects["obj"]

        def state() -> tuple[np.ndarray, np.ndarray, float, bool, bool, bool]:
            eef = np.asarray(raw.sim.data.site_xpos[eef_site_id], dtype=float).copy()
            obj = np.asarray(raw.sim.data.body_xpos[object_body_id], dtype=float).copy()
            distance = float(np.linalg.norm(obj - eef))
            grasped = bool(raw._check_grasp(robot.gripper["right"], object_model))
            inside = bool(OU.obj_inside_of(raw, "obj", raw.cab))
            simulator_success = bool(raw._check_success())
            return eef, obj, distance, grasped, inside, simulator_success

        _, _, distance, grasped, inside, success = state()
        phase = next_canonical_phase(
            "locate",
            object_eef_distance=distance,
            grasped=grasped,
            inside=inside,
            success=success,
        )
        for decision_step in range(args.steps):
            if phase == "done":
                break
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
                    "canonical_phase": phase,
                    "relation_key": RELATION_KEY,
                },
            )
            inference_time = time.perf_counter() - started
            inference_seconds.append(inference_time)
            action = guarded_action(raw_action, phase)
            before = state()
            repeats = 0
            info: dict[str, Any] = {}
            for _ in range(args.action_repeat):
                observation, _, terminated, truncated, info = env.step(
                    to_robocasa_action(action)
                )
                repeats += 1
                executed_sim_steps += 1
                after = state()
                if after[-1] or terminated or truncated:
                    break
            eef, obj, distance, grasped, inside, success = after
            success = bool(success or info.get("success", False))
            ever_grasped = ever_grasped or grasped
            ever_inside = ever_inside or inside
            next_phase = next_canonical_phase(
                phase,
                object_eef_distance=distance,
                grasped=grasped,
                inside=inside,
                success=success,
            )
            phase_counts[phase] += 1
            executed_decisions = decision_step + 1
            trajectory.append(
                {
                    "decision_step": decision_step,
                    "sim_repeats": repeats,
                    "phase": phase,
                    "next_phase": next_phase,
                    "raw_action": [float(value) for value in raw_action],
                    "guarded_action": action.tolist(),
                    "object_eef_distance_before": before[2],
                    "object_eef_distance_after": distance,
                    "grasped": grasped,
                    "inside": inside,
                    "success": success,
                    "inference_seconds": inference_time,
                    "eef_position": eef.tolist(),
                    "object_position": obj.tolist(),
                }
            )
            if decision_step % 10 == 0 or next_phase != phase or success:
                print(
                    f"decision={decision_step}/{args.steps} phase={phase}->{next_phase} "
                    f"distance={distance:.4f} grasped={grasped} inside={inside} "
                    f"success={success} inference={inference_time:.3f}s",
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
        "schema_version": "formal_skill_model_evaluation_v1",
        "evidence_scope": "formal_skill_model_evaluation",
        "relation_key": RELATION_KEY,
        "counts_toward_task2_coverage": bool(success and all(predicates.values())),
        "seed": args.seed,
        "split": split,
        "instruction": instruction,
        "canonical_actions": CANONICAL_ACTIONS,
        "success": success,
        "success_predicates": predicates,
        "max_decision_steps": args.steps,
        "action_repeat": args.action_repeat,
        "executed_decision_steps": executed_decisions,
        "executed_sim_steps": executed_sim_steps,
        "ever_grasped": ever_grasped,
        "ever_inside": ever_inside,
        "final_grasped": final_grasped,
        "final_inside": final_inside,
        "phase_counts": dict(phase_counts),
        "mean_inference_seconds": (
            float(np.mean(inference_seconds)) if inference_seconds else None
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "skill_ir": str(args.skill_ir.resolve()),
        "skill_ir_sha256": hashlib.sha256(skill_bytes).hexdigest(),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "adapter_sha256": _adapter_fingerprint(args.adapter_dir),
        "environment_mapping": environment_mapping,
        "video": str(video_path.resolve()) if video_frames else None,
        "trajectory": trajectory,
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    print(json.dumps({key: report[key] for key in (
        "relation_key", "seed", "split", "success", "success_predicates",
        "executed_decision_steps", "executed_sim_steps", "ever_grasped", "ever_inside"
    )}, ensure_ascii=False, indent=2), flush=True)
    print(report_path.resolve(), flush=True)
    return 0 if report["counts_toward_task2_coverage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
