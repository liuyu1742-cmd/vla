"""Formal Task-2 relation to RoboCasa environment mappings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .robocasa_assets import LOCAL_REGISTRY, register_local_toy_assets


ORGANIZING_TOY = {
    "relation_key": "organizing::toy",
    "robocasa_task": "PickPlaceCounterToCabinet",
    "gym_id": "robocasa/PickPlaceCounterToCabinet",
    "object_group": "toy",
    "target_object": "toy",
    "target_region": "cabinet",
    "canonical_target": "storage",
    "success_predicate_version": "organizing_toy_v1",
}


def formal_env_spec(relation_key: str) -> dict[str, object]:
    """Return a copy of the implemented formal environment mapping."""
    if relation_key != ORGANIZING_TOY["relation_key"]:
        raise ValueError(f"formal relation environment is not implemented: {relation_key}")
    return dict(ORGANIZING_TOY)


def create_formal_env(
    relation_key: str,
    *,
    seed: int,
    camera_name: str = "robot0_agentview_left",
    gym_make: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, object]]:
    """Create the relation's real RoboCasa environment and mapping metadata."""
    spec = formal_env_spec(relation_key)
    register_local_toy_assets()
    if gym_make is None:
        import gymnasium as gym
        import robocasa  # noqa: F401

        gym_make = gym.make
    env = gym_make(
        str(spec["gym_id"]),
        split="pretrain",
        seed=seed,
        obj_registries=(LOCAL_REGISTRY,),
        obj_groups=str(spec["object_group"]),
        camera_names=[camera_name],
        disable_env_checker=True,
    )
    metadata = dict(spec)
    metadata.update({"seed": seed, "camera_name": camera_name})
    return env, metadata


__all__ = ["ORGANIZING_TOY", "create_formal_env", "formal_env_spec"]
