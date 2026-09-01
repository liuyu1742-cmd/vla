"""Reset and audit the real RoboCasa ``organizing::storage_box`` mapping."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELATION_KEY = "organizing::storage_box"
CAMERA_KEY = "video.robot0_agentview_left"
HELD_OUT_SEEDS = {201, 202, 203}


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def native_action_shape(action_spec: object) -> list[int]:
    if not isinstance(action_spec, tuple) or len(action_spec) != 2:
        raise ValueError("robosuite action_spec must contain lower and upper bounds")
    lower = np.asarray(action_spec[0])
    upper = np.asarray(action_spec[1])
    if lower.ndim != 1 or lower.shape != upper.shape:
        raise ValueError("robosuite action_spec bounds must have the same flat shape")
    return [int(lower.shape[0])]


def validate_reset_audit(report: Mapping[str, object]) -> dict[str, object]:
    """Validate reset evidence without promoting it to execution evidence."""

    if report.get("schema_version") != "formal_env_mapping_check_v1":
        raise ValueError("unsupported reset audit schema")
    if report.get("relation_key") != RELATION_KEY:
        raise ValueError("reset audit relation must be organizing::storage_box")
    if report.get("reset_completed") is not True:
        raise ValueError("RoboCasa reset did not complete")
    if report.get("task_success_at_reset") is True:
        raise ValueError("task cannot already be successful at reset")

    mapping = report.get("mapping")
    if not isinstance(mapping, Mapping):
        raise ValueError("mapping metadata is missing")
    if (
        mapping.get("object_group") != "tupperware"
        or mapping.get("simulator_object_proxy") != "tupperware"
        or mapping.get("semantic_asset_proxy") is not True
    ):
        raise ValueError("mapping must disclose the native tupperware proxy")

    action_space = report.get("action_space")
    if not isinstance(action_space, Mapping) or action_space.get("shape") != [12]:
        raise ValueError("RoboCasa mapping requires the native 12-dimensional action space")
    camera = report.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("camera metadata is missing")
    camera_shape = camera.get("shape")
    if (
        camera.get("key") != CAMERA_KEY
        or not isinstance(camera_shape, list)
        or len(camera_shape) != 3
        or camera_shape[-1] != 3
    ):
        raise ValueError("required RGB virtual camera is invalid")

    obj = report.get("object")
    if not isinstance(obj, Mapping):
        raise ValueError("object metadata is missing")
    model_path = str(obj.get("model_path", "")).lower()
    if obj.get("category") != "tupperware" or "tupperware" not in model_path:
        raise ValueError("sampled object must be a tupperware asset")

    predicates = report.get("initial_predicates")
    if not isinstance(predicates, Mapping):
        raise ValueError("initial predicates are missing")
    if predicates.get("object_on_counter") is not True:
        raise ValueError("target object must start on the counter")
    if predicates.get("object_inside_cabinet") is not False:
        raise ValueError("target object must start outside the cabinet")

    validated = dict(report)
    validated["simulator_mapping_ready"] = True
    return validated


def audit_seed(seed: int, output_root: Path) -> dict[str, object]:
    if seed in HELD_OUT_SEEDS:
        raise ValueError(f"held-out seed {seed} cannot be used for mapping tuning")

    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))

    import imageio.v3 as iio
    from robocasa.utils import object_utils as OU

    from tools.skill_transfer.robocasa_envs import create_formal_env

    seed_dir = Path(output_root) / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env, mapping = create_formal_env(RELATION_KEY, seed=seed)
        observation, reset_info = env.reset(seed=seed)
        raw = env.unwrapped.env
        if CAMERA_KEY not in observation:
            available = sorted(
                str(key) for key in observation if str(key).startswith("video.")
            )
            raise RuntimeError(
                f"required virtual camera {CAMERA_KEY!r} is absent; available={available}"
            )
        frame = np.asarray(observation[CAMERA_KEY], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"virtual camera produced invalid shape {frame.shape}")
        frame_path = seed_dir / "reset_frame.png"
        iio.imwrite(frame_path, frame)

        obj = raw.objects["obj"]
        body_id = raw.obj_body_id["obj"]
        object_position = np.asarray(raw.sim.data.body_xpos[body_id], dtype=float)
        info = raw.object_cfgs[0].get("info", {})
        category = info.get("cat") if isinstance(info, Mapping) else None
        object_on_counter = bool(
            OU.check_obj_fixture_contact(raw, "obj", raw.counter)
        )
        object_inside_cabinet = bool(OU.obj_inside_of(raw, "obj", raw.cab))
        gripper_far = bool(OU.gripper_obj_far(raw, "obj"))
        report = {
            "schema_version": "formal_env_mapping_check_v1",
            "relation_key": RELATION_KEY,
            "seed": seed,
            "reset_completed": True,
            "simulator_mapping_ready": True,
            "task_success_at_reset": bool(raw._check_success()),
            "mapping": mapping,
            "action_space": {"shape": native_action_shape(raw.action_spec)},
            "camera": {
                "key": CAMERA_KEY,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "frame": str(frame_path.resolve()),
            },
            "initial_predicates": {
                "object_on_counter": object_on_counter,
                "object_inside_cabinet": object_inside_cabinet,
                "gripper_far": gripper_far,
            },
            "object": {
                "category": category,
                "model_path": str(Path(obj.mjcf_path).resolve()),
                "size": np.asarray(obj.size, dtype=float).tolist(),
                "position": object_position.tolist(),
            },
            "target": {"canonical": "storage", "fixture": "cabinet"},
            "reset_info": {
                str(key): _json_ready(value) for key, value in reset_info.items()
            },
        }
        validated = validate_reset_audit(report)
        report_path = seed_dir / "report.json"
        report_path.write_text(
            json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return validated
    finally:
        if env is not None:
            env.close()


def run_audits(seeds: Sequence[int], output_root: Path) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one seed is required")
    reports = [audit_seed(seed, output_root) for seed in seeds]
    manifest = {
        "schema_version": "formal_env_mapping_batch_v1",
        "relation_key": RELATION_KEY,
        "seeds": list(seeds),
        "held_out_seeds": sorted(HELD_OUT_SEEDS),
        "held_out_overlap": sorted(set(seeds) & HELD_OUT_SEEDS),
        "report_count": len(reports),
        "all_simulator_mappings_ready": all(
            report.get("simulator_mapping_ready") is True for report in reports
        ),
        "execution_success_claimed": False,
        "reports": [
            str((Path(output_root) / f"seed_{seed:03d}" / "report.json").resolve())
            for seed in seeds
        ],
    }
    manifest_path = Path(output_root) / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "organizing_storage_box_env_check",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_audits(args.seeds, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print((args.output / "manifest.json").resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
