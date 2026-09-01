"""RoboCasa VLA mapping audit with the complete official object registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools import audit_vla_midterm_simulator_mapping_v1 as legacy


OBJECT_REGISTRIES = ("objaverse", "lightwheel", "aigen")
install_target_object_group = legacy.install_target_object_group
audit_relation = legacy.audit_relation
run_audit = legacy.run_audit
build_parser = legacy.build_parser


def _one_reset(
    relation: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    import gymnasium as gym
    import imageio.v3 as iio
    import robocasa  # noqa: F401
    from robocasa.models.objects.kitchen_objects import OBJ_GROUPS

    task_class = str(relation["task_class"])
    object_group = str(relation["object_group"])
    if object_group not in OBJ_GROUPS:
        raise ValueError(f"RoboCasa object group is unavailable: {object_group}")
    cls = legacy._task_class(task_class)
    original = install_target_object_group(cls, object_group)
    env = None
    try:
        env = gym.make(
            f"robocasa/{task_class}",
            split="pretrain",
            seed=seed,
            obj_registries=OBJECT_REGISTRIES,
            obj_groups=object_group,
            camera_names=["robot0_agentview_left"],
            disable_env_checker=True,
        )
        observation, reset_info = env.reset(seed=seed)
        raw = env.unwrapped.env
        frame = np.asarray(observation[legacy.CAMERA_KEY], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError(f"invalid RGB frame shape: {frame.shape}")
        frame_path = output_dir / "reset_frame.png"
        iio.imwrite(frame_path, frame)

        actual = legacy._actual_object_metadata(raw)
        allowed_categories = list(OBJ_GROUPS[object_group])
        actual_category = actual["category"]
        if actual_category not in allowed_categories:
            raise RuntimeError(
                f"sampled category {actual_category!r} not in "
                f"OBJ_GROUPS[{object_group!r}]={allowed_categories!r}"
            )
        task_success_at_reset = bool(raw._check_success())
        if task_success_at_reset:
            raise RuntimeError("task is already successful at reset")
        lower, upper = raw.action_spec
        action_shape = list(np.asarray(lower).shape)
        if action_shape != [12] or np.asarray(upper).shape != (12,):
            raise RuntimeError(
                f"expected native RoboCasa action shape [12], got {action_shape}"
            )

        episode_meta = raw.get_ep_meta()
        instruction = str(episode_meta.get("lang", ""))
        expected_object = str(relation["manipulated_object_text"])
        if expected_object not in instruction.lower():
            raise RuntimeError(
                f"reset instruction does not name {expected_object!r}: "
                f"{instruction!r}"
            )
        return {
            "status": "PASS",
            "task_class": task_class,
            "object_group": object_group,
            "seed": seed,
            "instruction": instruction,
            "source_instruction": relation["example_instruction"],
            "object_registries": list(OBJECT_REGISTRIES),
            "actual_object": actual,
            "allowed_categories": allowed_categories,
            "camera": {
                "key": legacy.CAMERA_KEY,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "frame": str(frame_path.resolve()),
            },
            "action_shape": action_shape,
            "task_success_at_reset": task_success_at_reset,
            "reset_info": {
                str(key): legacy._json_ready(value)
                for key, value in reset_info.items()
            },
        }
    finally:
        if env is not None:
            env.close()
        cls._get_obj_cfgs = original


legacy._one_reset = _one_reset
main = legacy.main


if __name__ == "__main__":
    raise SystemExit(main())
