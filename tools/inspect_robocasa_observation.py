"""Print observation keys for the project water-cup RoboCasa task."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))

import gymnasium as gym
import robocasa  # noqa: F401
from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
    PickPlaceCounterToCabinet,
)


def main() -> int:
    original = PickPlaceCounterToCabinet._get_obj_cfgs

    def scaled(environment):
        configurations = original(environment)
        for configuration in configurations:
            if configuration.get("name") == "obj":
                configuration["object_scale"] = 0.7
        return configurations

    PickPlaceCounterToCabinet._get_obj_cfgs = scaled
    environment = gym.make(
        "robocasa/PickPlaceCounterToCabinet",
        split="pretrain",
        seed=0,
        obj_registries=("lightwheel",),
        obj_groups="glass_cup",
        disable_env_checker=True,
    )
    try:
        observation, _ = environment.reset(seed=0)
        for key, value in observation.items():
            print(f"{key}: shape={getattr(value, 'shape', None)} value={value}")
    finally:
        environment.close()
        PickPlaceCounterToCabinet._get_obj_cfgs = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
