"""Print every collision decision made while resetting one drawer-source task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from robocasa.utils import env_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_vla82_full_simulation import _load_compiled_specs, _scene_for_spec
from tools.vla82_full_sim.environment import make_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-id", default="VLA82-037")
    args = parser.parse_args()
    original = env_utils.detect_robot_collision
    calls = 0

    def traced(environment):
        nonlocal calls
        calls += 1
        base, _ = environment.robots[0].composite_controller.part_controllers["base"].get_base_pose()
        collided = bool(original(environment))
        print(f"collision_call={calls:02d} base={np.round(base, 5).tolist()} collided={collided}", flush=True)
        if not collided and hasattr(environment, "object_placements"):
            drawer_points = np.asarray(environment.drawer.get_bbox_points(), dtype=float)
            object_body = int(environment.sim.model.body_name2id(environment.objects["obj"].root_body))
            print(
                "safe_context="
                f"object={np.round(environment.object_placements['obj'][0], 5).tolist()} "
                f"live_object={np.round(environment.sim.data.body_xpos[object_body], 5).tolist()} "
                f"drawer={np.round(environment.drawer.pos, 5).tolist()} "
                f"bbox_low={np.round(drawer_points.min(axis=0), 5).tolist()} "
                f"bbox_high={np.round(drawer_points.max(axis=0), 5).tolist()}",
                flush=True,
            )
        return collided

    env_utils.detect_robot_collision = traced
    spec = next(item for item in _load_compiled_specs() if item.selection_id == args.selection_id)
    environment = make_environment(_scene_for_spec(spec), seed=2000)
    environment.close()


if __name__ == "__main__":
    main()
