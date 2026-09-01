"""Formal Task-2 relation to RoboCasa environment mappings.

This package is the active compatibility implementation while the Windows editor
helper cannot update the original module in place. Target toys come from the local
registry; standard lightwheel objects remain available for task distractors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from tools.skill_transfer.robocasa_assets import LOCAL_REGISTRY, register_local_toy_assets


OBJECT_REGISTRIES = (LOCAL_REGISTRY, "lightwheel")
ORGANIZING_TOY = {
    "relation_key": "organizing::toy",
    "robocasa_task": "PickPlaceCounterToCabinet",
    "gym_id": "robocasa/PickPlaceCounterToCabinet",
    "object_group": "toy",
    "target_object": "toy",
    "target_region": "cabinet",
    "canonical_target": "storage",
    "success_predicate_version": "organizing_toy_v1",
    "object_registries": OBJECT_REGISTRIES,
}
ORGANIZING_STORAGE_BOX = {
    "relation_key": "organizing::storage_box",
    "robocasa_task": "PickPlaceCounterToCabinet",
    "gym_id": "robocasa/PickPlaceCounterToCabinet",
    "object_group": "tupperware",
    "target_object": "storage_box",
    "simulator_object_proxy": "tupperware",
    "semantic_asset_proxy": True,
    "target_region": "cabinet",
    "canonical_target": "storage",
    "success_predicate_version": "organizing_storage_box_v1",
    "object_scale": 0.45,
    "object_rotation": math.pi / 2.0,
    "object_registries": ("lightwheel",),
}
FORMAL_ENV_SPECS = {
    str(ORGANIZING_TOY["relation_key"]): ORGANIZING_TOY,
    str(ORGANIZING_STORAGE_BOX["relation_key"]): ORGANIZING_STORAGE_BOX,
}


def formal_env_spec(relation_key: str) -> dict[str, object]:
    if relation_key not in FORMAL_ENV_SPECS:
        raise ValueError(f"formal relation environment is not implemented: {relation_key}")
    return dict(FORMAL_ENV_SPECS[relation_key])


def install_object_scale(environment_class: type, object_scale: float) -> Callable:
    if object_scale <= 0:
        raise ValueError("object_scale must be positive")
    original_get_obj_cfgs = environment_class._get_obj_cfgs

    def scaled_get_obj_cfgs(environment: Any):
        configs = original_get_obj_cfgs(environment)
        for config in configs:
            if config.get("name") == "obj":
                config["object_scale"] = object_scale
        return configs

    environment_class._get_obj_cfgs = scaled_get_obj_cfgs
    return original_get_obj_cfgs


def install_instance_object_config(
    environment: Any,
    *,
    object_scale: float,
    object_rotation: float,
) -> Callable:
    if object_scale <= 0:
        raise ValueError("object_scale must be positive")
    original_get_obj_cfgs = environment._get_obj_cfgs

    def configured_get_obj_cfgs():
        configs = original_get_obj_cfgs()
        for config in configs:
            if config.get("name") == "obj":
                config["object_scale"] = object_scale
                config.setdefault("placement", {})["rotation"] = object_rotation
        return configs

    environment._get_obj_cfgs = configured_get_obj_cfgs
    return original_get_obj_cfgs


def install_instance_object_scale(environment: Any, object_scale: float) -> Callable:
    """Backward-compatible scale-only instance patch."""
    return install_instance_object_config(
        environment,
        object_scale=object_scale,
        object_rotation=0.0,
    )


def create_formal_env(
    relation_key: str,
    *,
    seed: int,
    camera_name: str = "robot0_agentview_left",
    gym_make: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, object]]:
    spec = formal_env_spec(relation_key)
    if relation_key == ORGANIZING_TOY["relation_key"]:
        register_local_toy_assets()
    object_registries = tuple(spec["object_registries"])
    using_default_gym = gym_make is None
    if gym_make is None:
        import gymnasium as gym
        import robocasa  # noqa: F401

        gym_make = gym.make
    env = gym_make(
        str(spec["gym_id"]),
        split="pretrain",
        seed=seed,
        obj_registries=object_registries,
        obj_groups=str(spec["object_group"]),
        camera_names=[camera_name],
        disable_env_checker=True,
    )
    if using_default_gym and relation_key == ORGANIZING_STORAGE_BOX["relation_key"]:
        install_instance_object_config(
            env.unwrapped.env,
            object_scale=float(spec["object_scale"]),
            object_rotation=float(spec["object_rotation"]),
        )
    metadata = dict(spec)
    metadata.update(
        {
            "seed": seed,
            "camera_name": camera_name,
            "object_registries": list(object_registries),
        }
    )
    return env, metadata


__all__ = [
    "OBJECT_REGISTRIES",
    "FORMAL_ENV_SPECS",
    "ORGANIZING_STORAGE_BOX",
    "ORGANIZING_TOY",
    "create_formal_env",
    "formal_env_spec",
    "install_instance_object_config",
    "install_instance_object_scale",
    "install_object_scale",
]
